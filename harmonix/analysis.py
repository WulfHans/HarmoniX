from __future__ import annotations
import threading
from collections import deque

import numpy as np
import scipy.signal
import librosa
import soundfile as sf
import sounddevice as sd

from harmonix.model import Overtone, Tone

SAMPLE_RATE = 22050
WINDOW_DURATION = 0.5          # seconds
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
FFT_SIZE = 4096
MAX_OVERTONES = 16
MIN_VOLUME_FACTOR = 0.02       # discard partials quieter than 2% of the base
MAX_VOLUME_FACTOR = 10.0       # partials louder than 10x the base are most likely detection glitches
_BASE_TOLERANCE = 0.15         # peaks within [1.0, 1.0 + tol) count as the base, not an overtone

YIN_FMIN = 80.0                # lower F0 bound — keeps YIN above the 50 Hz mains hum
_HPS_K = 5                     # harmonics evaluated in the HPS product
_HPS_F_CAP = 1000.0            # highest plausible fundamental (Hz)


def _interp_magnitude(freq: float, magnitude: np.ndarray, sample_rate: int) -> float:
    """Linearly interpolate the FFT magnitude at an arbitrary frequency."""
    idx = freq * FFT_SIZE / sample_rate
    lo = int(idx)
    hi = min(lo + 1, len(magnitude) - 1)
    frac = idx - lo
    return float(magnitude[lo] * (1 - frac) + magnitude[hi] * frac)


def _hps_refine_f0(f0: float, magnitude: np.ndarray, sample_rate: int) -> float:
    """Select the best F0 among {f0, f0×2, f0×3} via the Harmonic Product Score.

    HPS = product of interpolated magnitudes at k=1..K harmonics.  A genuine
    fundamental requires all harmonics to be simultaneously present; a single
    missing harmonic collapses the product to near zero, which reliably rejects
    sub-octave mis-detections.
    """
    candidates = [f0 * m for m in range(1, 4) if 0 < f0 * m <= _HPS_F_CAP]
    if not candidates:
        return f0

    def score(f_cand: float) -> float:
        p = 1.0
        for k in range(1, _HPS_K + 1):
            p *= _interp_magnitude(f_cand * k, magnitude, sample_rate)
        return p

    return max(candidates, key=score)


def analyze_frame(
    frame: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    use_hps: bool = False,
) -> Tone | None:
    """Extract a Tone from a single audio frame (numpy float32/64 array)."""
    if len(frame) < 512:
        return None

    frame = frame.astype(np.float32)
    max_amp = float(np.max(np.abs(frame)))
    if max_amp < 1e-6:
        return None
    frame_norm = frame / max_amp

    windowed = frame_norm * np.hanning(len(frame_norm))

    # F0 via YIN — returns one value per analysis frame; take the median of voiced frames
    try:
        f0_series = librosa.yin(windowed, fmin=YIN_FMIN, fmax=4000.0, sr=sample_rate)
        voiced = f0_series[(f0_series > 0) & np.isfinite(f0_series)]
        if len(voiced) == 0:
            return None
        f0 = float(np.median(voiced))
    except Exception:
        return None

    if f0 <= 0:
        return None

    # Magnitude spectrum
    magnitude = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)

    # Optional HPS refinement: if YIN landed on a sub-harmonic, the true
    # fundamental (f0×2 or f0×3) will have a higher harmonic product score.
    if use_hps:
        f0 = _hps_refine_f0(f0, magnitude, sample_rate)

    # Amplitude at F0
    f0_bin = int(round(f0 * FFT_SIZE / sample_rate))
    f0_bin = int(np.clip(f0_bin, 1, len(magnitude) - 1))
    f0_amp = float(magnitude[f0_bin])
    if f0_amp < 1e-9:
        return None

    # Spectral peak detection — minimum distance = ~0.4 × F0 to avoid sub-harmonic noise
    min_dist = max(1, int(f0 * 0.4 * FFT_SIZE / sample_rate))
    peaks, _ = scipy.signal.find_peaks(
        magnitude,
        height=f0_amp * MIN_VOLUME_FACTOR,
        distance=min_dist,
    )

    overtones: list[Overtone] = []
    for p in peaks:
        freq = float(freqs[p])
        if freq < 20.0:
            continue
        factor = freq / f0
        vol_factor = float(magnitude[p]) / f0_amp
        if factor < 1.0 + _BASE_TOLERANCE:
            # Either the base partial itself, or a sub-harmonic/undertone —
            # by definition an "overtone" lies above the base frequency.
            continue
        if vol_factor > MAX_VOLUME_FACTOR:
            # An overtone louder than 10x the base is most likely a detection
            # glitch (e.g. a misplaced F0 deflating f0_amp), not a real partial.
            continue
        overtones.append(Overtone(
            frequency_factor=round(factor, 4),
            volume_factor=round(vol_factor, 4),
        ))

    # Keep loudest partials only
    overtones.sort(key=lambda o: -o.volume_factor)
    overtones = overtones[:MAX_OVERTONES]

    return Tone(
        base_frequency=round(f0, 2),
        base_volume=round(max_amp, 4),
        overtones=overtones,
    )


def _read_center_window(path: str, sample_rate: int, window_samples: int) -> np.ndarray:
    """Read roughly `window_samples` (at `sample_rate`) of audio from around
    the middle of the file.

    Reads only a bounded chunk via `soundfile`, rather than decoding the whole
    file with `librosa.load`. This matters because some encoders (e.g.
    streaming-mode FLAC) write an unknown total-sample count (reported by
    libsndfile as the int64 sentinel `2**63 - 1`); asking such a file to
    allocate an array sized from that count raises "array is too big" / can
    crash. By bounding the read ourselves we never hit that path, and as a
    bonus we avoid decoding the entire file just to analyze 500 ms of it.
    """
    with sf.SoundFile(path) as f:
        native_sr = f.samplerate
        native_needed = int(window_samples * native_sr / sample_rate) + native_sr

        total = f.frames
        # Treat anything over ~24h as a bogus/unknown length and just read
        # from the start instead of trying to seek to a fabricated "center".
        if 0 < total < native_sr * 60 * 60 * 24:
            start = max(0, total // 2 - native_needed // 2)
            f.seek(start)

        raw = f.read(frames=native_needed, dtype="float32", always_2d=True)

    if raw.size == 0:
        return np.array([], dtype=np.float32)

    mono = raw.mean(axis=1).astype(np.float32)
    if native_sr != sample_rate:
        mono = librosa.resample(mono, orig_sr=native_sr, target_sr=sample_rate)
    return mono


def analyze_file(path: str, *, use_hps: bool = False) -> Tone | None:
    """Load an audio file and analyze a 500 ms window from the center."""
    audio = _read_center_window(path, SAMPLE_RATE, WINDOW_SAMPLES)
    if len(audio) == 0:
        return None
    center = len(audio) // 2
    half = WINDOW_SAMPLES // 2
    start = max(0, center - half)
    end = min(len(audio), start + WINDOW_SAMPLES)
    return analyze_frame(audio[start:end], SAMPLE_RATE, use_hps=use_hps)


class MicCapture:
    """Captures microphone audio into a rolling 500 ms circular buffer."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._buffer: deque[float] = deque(maxlen=WINDOW_SAMPLES)
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            self._buffer.extend(indata[:, 0].tolist())

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            if len(self._buffer) < WINDOW_SAMPLES // 2:
                return None
            return np.array(self._buffer, dtype=np.float32)

    def analyze(self, *, use_hps: bool = False) -> Tone | None:
        frame = self.get_frame()
        if frame is None:
            return None
        return analyze_frame(frame, self.sample_rate, use_hps=use_hps)
