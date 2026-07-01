# HarmoniX — Tone Analysis & Synthesis Application

## Context

Building a Python GUI application that models any tone as a **base frequency + a list of overtones** (each with a relative frequency factor and relative volume). Two main features: (1) analyze audio into this structure, (2) synthesize and play a tone from it. A named-tone database is stored/loaded as a JSON file. The GUI has two tabs: Analyze and Edit (Edit spec TBD).

---

## Research: Is the Discrete Overtone Model Viable?

**Yes — it is the industry standard** (sinusoidal model / additive synthesis), used in SPEAR, Loris, SMS (Xavier Serra). A kernel/continuous spectrum would **not** give a significant improvement for musical tones: spectral peaks are only a few Hz wide, so a Gaussian kernel is perceptually identical to a discrete spike. The non-integer frequency factor already handles inharmonic instruments. A meaningful future extension would be an optional `noise_floor_db` field (SMS-style stochastic residual), not a kernel.

---

## Data Model (`harmonix/model.py`)

```python
@dataclass
class Overtone:
    frequency_factor: float  # ratio to base frequency (2.0 = one octave above)
    volume_factor: float     # ratio to base volume (1.0 = same level)

@dataclass
class Tone:
    base_frequency: float    # Hz
    base_volume: float       # 0.0–1.0 linear
    overtones: list[Overtone]

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> 'Tone': ...

@dataclass
class ToneDatabase:
    tones: dict[str, Tone]   # name → Tone

    def save(self, path: str) -> None: ...       # JSON
    @classmethod
    def load(cls, path: str) -> 'ToneDatabase': ...
    def add(self, name: str, tone: Tone) -> None: ...
    def remove(self, name: str) -> None: ...
```

---

## Library Stack

| Purpose | Library |
|---|---|
| File I/O (WAV/FLAC) | `soundfile` |
| File I/O (MP3) | `librosa` (needs system `ffmpeg`) |
| Mic capture | `sounddevice` |
| F0 detection | `librosa.yin()` |
| FFT + peak detection | `numpy`, `scipy.signal.find_peaks` |
| Synthesis + playback | `sounddevice` + `numpy` |
| GUI | `PyQt5` |
| Real-time spectrum plot | `pyqtgraph` |

**`requirements.txt`:** `numpy scipy librosa soundfile sounddevice PyQt5 pyqtgraph`

---

## Project Structure

```
harmonix/
├── __init__.py
├── model.py              # Tone, Overtone, ToneDatabase dataclasses + JSON I/O
├── analysis.py           # ToneAnalyzer: audio → Tone
├── synthesis.py          # ToneSynthesizer: Tone → audio → playback
└── gui/
    ├── __init__.py
    ├── app.py             # QMainWindow, two-tab layout, database load/save
    ├── analyze_tab.py     # Analyze tab widget
    ├── tone_plot.py       # Bar-chart widget (log freq axis, relative scale)
    └── edit_tab.py        # Edit tab widget (spec TBD)
main.py
requirements.txt
```

---

## Analyze Tab Layout (`harmonix/gui/analyze_tab.py`)

```
[ Load MP3 ]  [ Start Recording ]  [ Stop Recording ]

┌─────────────────────────────────────────────────────┐
│  Tone spectrum plot (pyqtgraph)                     │
│  • X-axis: frequency factor (log scale)             │
│    - Base always anchored at factor = 1.0 (center)  │
│    - Overtones appear at their factor position      │
│  • Y-axis: relative volume (base always = 1.0)      │
│  • Each partial shown as a vertical bar             │
│  • Base bar highlighted differently from overtones  │
└─────────────────────────────────────────────────────┘

Name: [_________________________]  [ Save ]  [ Clear ]

[ Play ]  [ Stop ]
```

**Plot details:**
- X-axis is log-scale frequency factor. Since all values are relative, x=1.0 is the base. Integer harmonics (1, 2, 3, 4...) fall at log-evenly-spaced positions.
- The base bar is always drawn at x=1, height=1 (reference). Overtone bars are drawn at their `frequency_factor`, height = `volume_factor`.
- Actual Hz label shown in a tooltip or secondary annotation (base_freq × factor).

**Behavior:**
- **Load MP3:** file dialog → analyze → update plot + populate name field with filename stem
- **Start Recording:** begin sounddevice InputStream, rolling 500 ms buffer; plot updates ~10 Hz via QTimer
- **Stop Recording:** stop stream, freeze last analyzed Tone in plot
- **Save:** add current Tone to ToneDatabase under the given name; persist JSON
- **Clear:** reset plot and name field; discard current Tone
- **Play / Stop:** additive synthesis of current Tone at its original base_frequency + base_volume; plays in background thread

---

## Edit Tab

Spec TBD. Placeholder tab in the initial implementation.

---

## Analysis Pipeline (`harmonix/analysis.py`)

**Shared frame analysis:**
1. Apply Hann window
2. `numpy.fft.rfft` → magnitude spectrum
3. `librosa.yin(frame, fmin=50, fmax=2000, sr=sample_rate)` → F0
4. `scipy.signal.find_peaks(magnitude, height=noise_threshold, distance=min_bin_distance)` → partial peaks
5. `frequency_factor = peak_hz / F0`, `volume_factor = peak_amp / F0_amp`
6. Sort by volume_factor descending; keep top N (configurable, default 16)
7. Return `Tone(base_frequency=F0, base_volume=normalized_F0_amp, overtones=[...])`

**File source:** `librosa.load(path, sr=22050)` → slice center 500 ms → frame analysis

**Mic source:** `sounddevice.InputStream` callback → circular buffer → QTimer snapshot → frame analysis

---

## Synthesis Pipeline (`harmonix/synthesis.py`)

```python
t = np.linspace(0, duration, int(sample_rate * duration))
wave = base_volume * np.sin(2π * base_frequency * t)
for ot in tone.overtones:
    wave += base_volume * ot.volume_factor * np.sin(2π * base_frequency * ot.frequency_factor * t)
wave = wave / np.max(np.abs(wave))  # normalize
sounddevice.play(wave.astype(np.float32), samplerate=sample_rate)
```

Playback uses the Tone's stored `base_frequency` and `base_volume`. Runs in a daemon thread so the GUI stays responsive. `sounddevice.stop()` handles the Stop button.

---

## Verification

1. `pip install -r requirements.txt` + `sudo apt install ffmpeg`
2. `python main.py` → window opens with Analyze and Edit tabs
3. **Synthesis check:** Record silence → hum a steady pitch → plot shows a bar at x=1 and harmonic bars to the right; save tone; press Play → reproduced pitch audible
4. **File check:** Load an MP3 of a known note → base_frequency matches expected pitch; plot shows harmonics
5. **Database check:** Save two tones with different names → close app → reopen → both names visible in Edit tab (once implemented)
6. **Round-trip:** Analyze → Save → Play → sounds similar to original
