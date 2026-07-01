import pytest

from harmonix.model import Overtone, Tone
from harmonix.gui.analyze_tab import AnalyzeTab


@pytest.fixture
def tab(qtbot, mocker):
    """An AnalyzeTab with hardware/file dependencies mocked out — no real
    microphone, audio file, or sound output is ever touched."""
    mocker.patch("harmonix.gui.analyze_tab.MicCapture")
    mocker.patch("harmonix.gui.analyze_tab.ToneSynthesizer")

    saved = []
    widget = AnalyzeTab(on_save_tone=lambda name, tone: saved.append((name, tone)))
    widget._saved = saved              # stash for assertions
    widget._mic.analyze.return_value = None  # keep the live-update timer inert
    qtbot.addWidget(widget)
    return widget


# --- Save -----------------------------------------------------------------

def test_save_requires_a_tone(tab):
    tab._name_edit.setText("name")
    tab._on_save()
    assert tab._saved == []
    assert "No tone" in tab._status.text()


def test_save_requires_a_name(tab):
    tab._current_tone = Tone(220.0, 0.5, [])
    tab._name_edit.setText("")
    tab._on_save()
    assert tab._saved == []
    assert "name" in tab._status.text().lower()


def test_save_invokes_callback_with_name_and_tone(tab):
    tone = Tone(220.0, 0.5, [Overtone(2.0, 0.5)])
    tab._current_tone = tone
    tab._name_edit.setText("my-tone")

    tab._on_save()

    assert tab._saved == [("my-tone", tone)]
    assert "my-tone" in tab._status.text()


# --- Clear -----------------------------------------------------------------

def test_clear_resets_tone_and_name(tab):
    tab._current_tone = Tone(220.0, 0.5, [])
    tab._name_edit.setText("something")

    tab._on_clear()

    assert tab._current_tone is None
    assert tab._name_edit.text() == ""


# --- Playback --------------------------------------------------------------

def test_play_passes_current_tone_to_synthesizer(tab):
    tone = Tone(220.0, 0.5, [])
    tab._current_tone = tone

    tab._on_play()

    tab._synth.play.assert_called_once()
    args, _ = tab._synth.play.call_args
    assert args[0] is tone


def test_play_without_a_tone_does_not_touch_synthesizer(tab):
    tab._current_tone = None
    tab._on_play()
    tab._synth.play.assert_not_called()


def test_play_stop_button_stops_synthesizer(tab):
    tab._on_play_stop()
    tab._synth.stop.assert_called_once()


# --- File loading -----------------------------------------------------------

def test_load_resets_plot_view_and_populates_tone_and_name(tab, mocker):
    fake_tone = Tone(330.0, 0.4, [Overtone(2.0, 0.3)])
    mocker.patch("harmonix.gui.analyze_tab.analyze_file", return_value=fake_tone)
    mocker.patch(
        "harmonix.gui.analyze_tab.QFileDialog.getOpenFileName",
        return_value=("/tmp/example.wav", ""),
    )
    reset_spy = mocker.spy(tab._plot, "reset_view")

    tab._on_load()

    reset_spy.assert_called_once()
    assert tab._current_tone == fake_tone
    assert tab._name_edit.text() == "example"
    assert "example.wav" in tab._status.text()


def test_load_cancelled_dialog_changes_nothing(tab, mocker):
    mocker.patch(
        "harmonix.gui.analyze_tab.QFileDialog.getOpenFileName",
        return_value=("", ""),
    )
    reset_spy = mocker.spy(tab._plot, "reset_view")

    tab._on_load()

    reset_spy.assert_not_called()
    assert tab._current_tone is None


def test_load_with_undetectable_pitch_reports_status(tab, mocker):
    mocker.patch("harmonix.gui.analyze_tab.analyze_file", return_value=None)
    mocker.patch(
        "harmonix.gui.analyze_tab.QFileDialog.getOpenFileName",
        return_value=("/tmp/silence.wav", ""),
    )

    tab._on_load()

    assert tab._current_tone is None
    assert "pitch" in tab._status.text().lower()


# --- Recording ---------------------------------------------------------------

def test_start_recording_resets_view_starts_mic_and_toggles_buttons(tab, mocker):
    reset_spy = mocker.spy(tab._plot, "reset_view")

    tab._on_start_recording()

    reset_spy.assert_called_once()
    tab._mic.start.assert_called_once()
    assert not tab._btn_start.isEnabled()
    assert tab._btn_stop.isEnabled()

    tab._on_stop_recording()

    tab._mic.stop.assert_called_once()
    assert tab._btn_start.isEnabled()
    assert not tab._btn_stop.isEnabled()
