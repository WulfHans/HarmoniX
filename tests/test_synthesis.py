import numpy as np
import pytest

from harmonix.model import Overtone, Tone
from harmonix.synthesis import build_waveform, build_combined_waveform, ToneSynthesizer


SR = 22050


def _spectrum(wave: np.ndarray, sample_rate: int = SR):
    magnitude = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / sample_rate)
    return freqs, magnitude


def _amplitude_near(freqs, magnitude, target_hz, bandwidth_hz=5.0):
    mask = np.abs(freqs - target_hz) <= bandwidth_hz
    return float(magnitude[mask].max()) if mask.any() else 0.0


def test_build_waveform_contains_base_and_overtones(simple_tone):
    wave = build_waveform(simple_tone, duration=1.0, sample_rate=SR)
    freqs, magnitude = _spectrum(wave, SR)

    base_amp = _amplitude_near(freqs, magnitude, simple_tone.base_frequency)
    assert base_amp > 0

    for ot in simple_tone.overtones:
        target_hz = simple_tone.base_frequency * ot.frequency_factor
        amp = _amplitude_near(freqs, magnitude, target_hz)
        ratio = amp / base_amp
        assert ratio == pytest.approx(ot.volume_factor, rel=0.1), (
            f"x{ot.frequency_factor}: expected amplitude ratio "
            f"~{ot.volume_factor}, got {ratio:.3f}"
        )


def test_build_waveform_respects_overridden_frequency_and_volume(simple_tone):
    wave = build_waveform(simple_tone, base_frequency=440.0, base_volume=0.3,
                          duration=0.5, sample_rate=SR)
    freqs, magnitude = _spectrum(wave, SR)

    # Energy should now be at 440 Hz, not the Tone's stored 220 Hz
    assert _amplitude_near(freqs, magnitude, 440.0) > _amplitude_near(freqs, magnitude, 220.0)
    assert np.max(np.abs(wave)) == pytest.approx(0.3, rel=0.05)


def test_build_waveform_never_clips():
    loud_tone = Tone(base_frequency=300.0, base_volume=1.0, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=5.0),
        Overtone(frequency_factor=3.0, volume_factor=5.0),
    ])
    wave = build_waveform(loud_tone, duration=0.2, sample_rate=SR)
    assert np.max(np.abs(wave)) <= 1.0 + 1e-6


def test_build_waveform_dtype_and_length():
    tone = Tone(base_frequency=100.0, base_volume=0.5, overtones=[])
    wave = build_waveform(tone, duration=0.25, sample_rate=SR)
    assert wave.dtype == np.float32
    assert len(wave) == int(SR * 0.25)


# --- ToneSynthesizer: must drive sounddevice without spawning Python threads ---
# (regression for the double-free crash: sd.play()/sd.stop() raced against a
# Python thread running sd.wait())

def test_synthesizer_play_calls_sounddevice_play_directly(simple_tone, mocker):
    mock_play = mocker.patch("harmonix.synthesis.sd.play")
    mock_stop = mocker.patch("harmonix.synthesis.sd.stop")

    synth = ToneSynthesizer(sample_rate=SR)
    synth.play(simple_tone, duration=0.1)

    mock_stop.assert_called_once()   # stops any prior playback first
    mock_play.assert_called_once()
    args, kwargs = mock_play.call_args
    assert kwargs.get("samplerate") == SR
    assert isinstance(args[0], np.ndarray)


def test_synthesizer_stop_calls_sounddevice_stop(mocker):
    mock_stop = mocker.patch("harmonix.synthesis.sd.stop")
    synth = ToneSynthesizer(sample_rate=SR)
    synth.stop()
    mock_stop.assert_called_once()


def test_build_combined_waveform_contains_all_voice_frequencies():
    tone_a = Tone(base_frequency=220.0, base_volume=1.0, overtones=[])
    tone_b = Tone(base_frequency=440.0, base_volume=1.0, overtones=[])
    parts = [(tone_a, 220.0, 1.0), (tone_b, 440.0, 1.0)]

    wave = build_combined_waveform(parts, duration=1.0, sample_rate=SR)
    freqs, magnitude = _spectrum(wave, SR)

    assert _amplitude_near(freqs, magnitude, 220.0) > 0
    assert _amplitude_near(freqs, magnitude, 440.0) > 0


def test_build_combined_waveform_never_clips():
    loud_tone = Tone(base_frequency=300.0, base_volume=1.0, overtones=[])
    parts = [(loud_tone, 300.0, 1.0), (loud_tone, 450.0, 1.0), (loud_tone, 600.0, 1.0)]

    wave = build_combined_waveform(parts, duration=0.2, sample_rate=SR)
    assert np.max(np.abs(wave)) <= 1.0 + 1e-6


def test_build_combined_waveform_dtype_and_length():
    tone = Tone(base_frequency=100.0, base_volume=0.5, overtones=[])
    wave = build_combined_waveform([(tone, 100.0, 0.5)], duration=0.25, sample_rate=SR)
    assert wave.dtype == np.float32
    assert len(wave) == int(SR * 0.25)


def test_synthesizer_play_multi_calls_sounddevice_play_directly(simple_tone, mocker):
    mock_play = mocker.patch("harmonix.synthesis.sd.play")
    mock_stop = mocker.patch("harmonix.synthesis.sd.stop")

    synth = ToneSynthesizer(sample_rate=SR)
    synth.play_multi([(simple_tone, 220.0, 1.0), (simple_tone, 440.0, 0.5)], duration=0.1)

    mock_stop.assert_called_once()
    mock_play.assert_called_once()
    args, kwargs = mock_play.call_args
    assert kwargs.get("samplerate") == SR
    assert isinstance(args[0], np.ndarray)


def test_synthesis_module_does_not_use_threading():
    """Regression: a Python thread running sd.wait() raced against sd.stop()
    called from the GUI thread, corrupting portaudio's internal state
    ('double free or corruption'). sd.play() is already non-blocking —
    portaudio drives playback in its own internal thread — so no Python
    threading is needed at all."""
    import harmonix.synthesis as synthesis_module
    assert not hasattr(synthesis_module, "threading")
