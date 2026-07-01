import os

# Must be set before any Qt/PyQt5 import happens (including pytest-qt's own
# QApplication setup) — this lets the GUI test suite run without a display,
# e.g. over SSH or in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from harmonix.model import Overtone, Tone

SAMPLE_RATE = 22050
FLAC_DIR = os.path.join(os.path.dirname(__file__), "..", "flac")


@pytest.fixture
def simple_tone() -> Tone:
    """A small, hand-built Tone for round-trip / synthesis / plotting tests."""
    return Tone(
        base_frequency=220.0,
        base_volume=0.6,
        overtones=[
            Overtone(frequency_factor=2.0, volume_factor=0.5),
            Overtone(frequency_factor=3.0, volume_factor=0.25),
        ],
    )


def make_harmonic_signal(f0: float, partials: dict, sample_rate: int = SAMPLE_RATE,
                         duration: float = 0.5) -> np.ndarray:
    """Build a synthetic signal as a sum of sine waves at known relative
    factors and amplitudes — `partials` maps frequency_factor -> amplitude,
    and the base (factor 1.0, amplitude 1.0) is always included."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2.0 * np.pi * f0 * t)
    for factor, amplitude in partials.items():
        signal = signal + amplitude * np.sin(2.0 * np.pi * f0 * factor * t)
    return signal.astype(np.float32)


@pytest.fixture
def synthetic_harmonic_signal():
    """A 220 Hz tone with a clean ×2 and ×3 harmonic series of known amplitude.

    Returns (signal, f0, expected_partials) where expected_partials maps
    frequency_factor -> amplitude for the overtones (excluding the base).
    """
    f0 = 220.0
    partials = {2.0: 0.5, 3.0: 0.25}
    return make_harmonic_signal(f0, partials), f0, partials


@pytest.fixture
def flac_files():
    """Paths to the example FLAC fixtures (streaming-encoded, unknown length —
    these reproduce the 'array is too big' regression)."""
    names = ["a3_a.flac", "a3_e.flac", "a3_i.flac"]
    paths = [os.path.join(FLAC_DIR, n) for n in names]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        pytest.skip(f"FLAC fixtures not found: {missing}")
    return paths
