import os
import numpy as np
import pytest

from harmonix.analysis import analyze_frame, analyze_file, SAMPLE_RATE, MAX_VOLUME_FACTOR
from tests.conftest import make_harmonic_signal

FLAC_DIR = os.path.join(os.path.dirname(__file__), "..", "flac")

# Files that previously produced wrong F0 with fmin=50 Hz, and their expected
# fundamental frequencies (Hz, ±10% tolerance).
_PREVIOUSLY_FAILING = [
    ("a3_o.flac", 210.0),   # YIN was pulled to ~53 Hz by 50 Hz mains hum
    ("d3_a.flac", 145.0),   # YIN found the sub-octave (73.8 Hz)
    ("d3_o.flac", 145.0),   # YIN locked onto the hum at 50 Hz
]


@pytest.fixture
def flac_files_fixed():
    """The three FLAC files that previously failed with fmin=50 Hz."""
    paths = [(os.path.join(FLAC_DIR, name), hz) for name, hz in _PREVIOUSLY_FAILING]
    missing = [p for p, _ in paths if not os.path.isfile(p)]
    if missing:
        pytest.skip(f"FLAC fixtures not found: {missing}")
    return paths


def test_analyze_frame_extracts_known_harmonics(synthetic_harmonic_signal):
    signal, f0, expected_partials = synthetic_harmonic_signal

    tone = analyze_frame(signal, SAMPLE_RATE)

    assert tone is not None
    assert tone.base_frequency == pytest.approx(f0, rel=0.02)

    found = {round(o.frequency_factor): o.volume_factor for o in tone.overtones}
    for factor, amplitude in expected_partials.items():
        assert round(factor) in found, f"expected partial x{factor} not detected"
        assert found[round(factor)] == pytest.approx(amplitude, rel=0.15)


def test_analyze_frame_rejects_silence():
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    assert analyze_frame(silence, SAMPLE_RATE) is None


def test_analyze_frame_rejects_short_frames():
    tiny = np.ones(64, dtype=np.float32)
    assert analyze_frame(tiny, SAMPLE_RATE) is None


def test_analyze_frame_filters_sub_base_partials():
    """Regression: peaks below the base (e.g. sub-harmonics / detection noise)
    must never show up as 'overtones' — by definition an overtone lies above
    the base frequency."""
    f0 = 220.0
    # Inject an artificial sub-harmonic at x0.22 alongside a real x2 overtone
    signal = make_harmonic_signal(f0, {0.22: 0.4, 2.0: 0.5})

    tone = analyze_frame(signal, SAMPLE_RATE)

    assert tone is not None
    assert all(o.frequency_factor >= 1.0 for o in tone.overtones)
    assert not any(o.frequency_factor < 1.15 for o in tone.overtones)


def test_analyze_frame_filters_glitch_volume_factors(mocker):
    """Regression: a volume_factor far above the base (>10x) signals a
    detection glitch -- e.g. YIN locking onto a frequency where the spectrum
    has almost no energy while a real, unrelated peak sits nearby, producing
    an absurd ratio -- and must be dropped rather than shown as an overtone."""
    f_real = 1100.0          # the only real energy in this signal
    bogus_f0 = 220.0         # F0 is fooled into landing far from any energy
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    signal = np.sin(2.0 * np.pi * f_real * t).astype(np.float32)

    mocker.patch("harmonix.analysis.librosa.yin", return_value=np.array([bogus_f0]))

    tone = analyze_frame(signal, SAMPLE_RATE)

    assert tone is not None
    assert tone.base_frequency == pytest.approx(bogus_f0)
    assert all(o.volume_factor <= MAX_VOLUME_FACTOR for o in tone.overtones)


def test_analyze_file_handles_streaming_flac_with_unknown_length(flac_files):
    """Regression for 'array is too big': these FLAC files were encoded in
    streaming mode with an unknown total-sample count (libsndfile reports the
    int64 sentinel 2**63 - 1 for `frames`). Loading must not crash, and must
    return a sensible Tone."""
    for path in flac_files:
        tone = analyze_file(path)
        assert tone is not None, f"expected a Tone from {path}"
        assert 50.0 < tone.base_frequency < 2000.0
        assert all(o.frequency_factor >= 1.0 for o in tone.overtones)


def test_analyze_file_missing_path_raises():
    with pytest.raises(Exception):
        analyze_file("does/not/exist.flac")


# ---------------------------------------------------------------------------
# YIN-80 regression: files that previously produced wrong F0 with fmin=50 Hz
# ---------------------------------------------------------------------------

def test_yin80_corrects_previously_failing_files(flac_files_fixed):
    """Raising fmin to 80 Hz excludes the 50 Hz mains hum from YIN's search
    space; all three formerly-failing recordings should now land within 10% of
    their true fundamental."""
    for path, expected_hz in flac_files_fixed:
        tone = analyze_file(path)
        assert tone is not None, f"expected a Tone from {path}"
        assert tone.base_frequency == pytest.approx(expected_hz, rel=0.10), (
            f"{path}: expected ~{expected_hz} Hz, got {tone.base_frequency:.1f} Hz"
        )


# ---------------------------------------------------------------------------
# HPS regression: use_hps=True should also detect correct F0
# ---------------------------------------------------------------------------

def test_hps_mode_corrects_previously_failing_files(flac_files_fixed):
    """With HPS enabled the system must still find the correct fundamental on
    all formerly-failing files (HPS cannot regress what YIN-80 already fixed)."""
    for path, expected_hz in flac_files_fixed:
        tone = analyze_file(path, use_hps=True)
        assert tone is not None, f"expected a Tone from {path} with HPS"
        assert tone.base_frequency == pytest.approx(expected_hz, rel=0.10), (
            f"{path}: HPS expected ~{expected_hz} Hz, got {tone.base_frequency:.1f} Hz"
        )


def test_hps_does_not_regress_always_working_files(flac_files):
    """HPS must not break files that were already correctly analyzed."""
    for path in flac_files:
        tone_default = analyze_file(path)
        tone_hps = analyze_file(path, use_hps=True)
        assert tone_hps is not None, f"HPS returned None for {path}"
        if tone_default is not None:
            assert tone_hps.base_frequency == pytest.approx(
                tone_default.base_frequency, rel=0.10
            ), (
                f"{path}: HPS ({tone_hps.base_frequency:.1f} Hz) diverged from "
                f"YIN-80 ({tone_default.base_frequency:.1f} Hz)"
            )


def test_analyze_frame_use_hps_flag_is_accepted():
    """Smoke test: use_hps keyword arg must be accepted without TypeError."""
    signal = make_harmonic_signal(220.0, {2.0: 0.5, 3.0: 0.3})
    tone = analyze_frame(signal, SAMPLE_RATE, use_hps=True)
    # Result may or may not be detected on a synthetic signal; we only care
    # that the call itself does not raise.
    assert tone is None or tone.base_frequency > 0
