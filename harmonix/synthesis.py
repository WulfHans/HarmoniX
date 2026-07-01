from __future__ import annotations

import numpy as np
import sounddevice as sd

from harmonix.model import Tone

SAMPLE_RATE = 44100
# Larger blocksize reduces the chance of ALSA underrun warnings on Linux
_BLOCKSIZE = 2048


def build_waveform(
    tone: Tone,
    base_frequency: float | None = None,
    base_volume: float | None = None,
    duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Construct an additive synthesis waveform from a Tone."""
    freq = base_frequency if base_frequency is not None else tone.base_frequency
    vol = base_volume if base_volume is not None else min(tone.base_volume, 1.0)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2.0 * np.pi * freq * t)
    for ot in tone.overtones:
        wave += ot.volume_factor * np.sin(2.0 * np.pi * freq * ot.frequency_factor * t)

    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak
    wave = (wave * vol).astype(np.float32)
    return wave


def build_combined_waveform(
    parts: list[tuple[Tone, float, float]],
    duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix several independently-pitched/volumed tones into one waveform.

    `parts` is a list of (tone, effective_frequency_hz, volume_scale) — each
    tone is rendered via `build_waveform` at its own frequency/volume override
    and the results are summed. Only normalized down if the sum would clip,
    so the relative balance set by `volume_scale` is preserved.
    """
    n = int(sample_rate * duration)
    combined = np.zeros(n, dtype=np.float32)
    for tone, freq, vol in parts:
        combined += build_waveform(tone, base_frequency=freq, base_volume=vol,
                                     duration=duration, sample_rate=sample_rate)

    peak = np.max(np.abs(combined))
    if peak > 1.0:
        combined = combined / peak
    return combined.astype(np.float32)


class ToneSynthesizer:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def play(
        self,
        tone: Tone,
        base_frequency: float | None = None,
        base_volume: float | None = None,
        duration: float = 2.0,
    ) -> None:
        # sd.play() is non-blocking — portaudio drives playback in its own
        # internal thread, so we never need a Python thread here.
        sd.stop()
        wave = build_waveform(tone, base_frequency, base_volume, duration, self.sample_rate)
        sd.play(wave, samplerate=self.sample_rate, blocksize=_BLOCKSIZE)

    def play_multi(self, parts: list[tuple[Tone, float, float]], duration: float = 2.0) -> None:
        sd.stop()
        wave = build_combined_waveform(parts, duration, self.sample_rate)
        sd.play(wave, samplerate=self.sample_rate, blocksize=_BLOCKSIZE)

    def stop(self) -> None:
        sd.stop()
