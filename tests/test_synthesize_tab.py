import pytest

from harmonix.model import Overtone, Tone, ToneDatabase
from harmonix.gui.synthesize_tab import SynthesizeTab


@pytest.fixture
def db():
    return ToneDatabase(tones={
        "tone-a": Tone(base_frequency=220.0, base_volume=0.6, overtones=[Overtone(2.0, 0.5)]),
        "tone-b": Tone(base_frequency=330.0, base_volume=0.4, overtones=[Overtone(2.0, 0.3)]),
    })


@pytest.fixture
def tab(qtbot, mocker, db):
    mocker.patch("harmonix.gui.synthesize_tab.ToneSynthesizer")
    widget = SynthesizeTab(db=db)
    qtbot.addWidget(widget)
    return widget


# --- Initialization ---------------------------------------------------------

def test_first_row_auto_selected_on_init(tab, db):
    row0 = tab._rows[0]
    assert row0.checkbox.isChecked()
    assert row0.selected_tone_name() == "tone-a"


def test_other_rows_start_inactive(tab):
    for row in tab._rows[1:]:
        assert not row.checkbox.isChecked()
        assert row.selected_tone_name() is None


def test_combos_populated_with_db_tone_names(tab):
    for row in tab._rows:
        names = [row.combo.itemText(i) for i in range(row.combo.count())]
        assert names == ["—", "tone-a", "tone-b"]


# --- Plot updates ------------------------------------------------------------

def test_changing_checkbox_updates_plot(tab, mocker):
    spy = mocker.spy(tab._plot, "update_voices")

    tab._rows[1].combo.setCurrentText("tone-b")
    tab._rows[1].checkbox.setChecked(True)

    assert spy.call_count >= 1
    entries = spy.call_args[0][0]
    assert len(entries) == 2


def test_unchecking_row_removes_it_from_plot(tab, mocker):
    tab._rows[1].combo.setCurrentText("tone-b")
    tab._rows[1].checkbox.setChecked(True)

    spy = mocker.spy(tab._plot, "update_voices")
    tab._rows[1].checkbox.setChecked(False)

    entries = spy.call_args[0][0]
    assert len(entries) == 1


def test_pitch_slider_changes_voice_pitch_factor(tab, mocker):
    tab._rows[1].combo.setCurrentText("tone-b")
    tab._rows[1].checkbox.setChecked(True)

    spy = mocker.spy(tab._plot, "update_voices")
    tab._rows[1].pitch_slider.setValue(1200)  # +1 octave => factor 2.0

    entries = spy.call_args[0][0]
    voice2 = next(e for e in entries if e.tone is tab._db.tones["tone-b"])
    assert voice2.pitch_factor == pytest.approx(2.0, rel=1e-3)


def test_first_voice_pitch_factor_is_always_one_in_diagram(tab):
    tab._rows[0].pitch_slider.setValue(1200)  # +1 octave
    entries = []
    tab._plot.update_voices = lambda e: entries.extend(e)
    tab._update_plot()

    voice1 = next(e for e in entries if e.tone is tab._db.tones["tone-a"])
    assert voice1.pitch_factor == pytest.approx(1.0)


# --- Volume slider ------------------------------------------------------------

def test_volume_slider_updates_label_and_scale(tab):
    row = tab._rows[0]
    row.volume_slider.setValue(50)
    assert row.volume_label.text() == "50%"
    assert row.volume_scale == pytest.approx(0.5)


# --- Playback ------------------------------------------------------------------

def test_play_passes_active_voices_to_synth(tab):
    tab._rows[1].combo.setCurrentText("tone-b")
    tab._rows[1].checkbox.setChecked(True)

    tab._on_play()

    tab._synth.play_multi.assert_called_once()
    args, _ = tab._synth.play_multi.call_args
    parts = args[0]
    assert len(parts) == 2


def test_play_with_no_active_voices_does_not_touch_synth(tab):
    tab._rows[0].checkbox.setChecked(False)

    tab._on_play()

    tab._synth.play_multi.assert_not_called()


def test_play_stop_calls_synth_stop(tab):
    tab._on_play_stop()
    tab._synth.stop.assert_called_once()


def test_reference_frequency_uses_first_voice_pitch(tab):
    tab._rows[0].pitch_slider.setValue(1200)  # ×2
    ref = tab._reference_frequency()
    assert ref == pytest.approx(220.0 * 2.0, rel=1e-3)


def test_reference_frequency_none_when_first_voice_unselected(tab):
    tab._rows[0].combo.setCurrentIndex(0)  # "—"
    assert tab._reference_frequency() is None


def test_second_voice_pitch_relative_to_first_voice_effective_frequency(tab):
    tab._rows[0].pitch_slider.setValue(1200)  # tone-a effective = 440 Hz
    tab._rows[1].combo.setCurrentText("tone-b")
    tab._rows[1].checkbox.setChecked(True)
    tab._rows[1].pitch_slider.setValue(0)  # ×1 relative to tone 1

    tab._on_play()

    args, _ = tab._synth.play_multi.call_args
    parts = args[0]
    tone_b_part = next(p for p in parts if p[0] is tab._db.tones["tone-b"])
    assert tone_b_part[1] == pytest.approx(440.0, rel=1e-3)


# --- Refresh ------------------------------------------------------------------

def test_refresh_updates_combo_options(tab, mocker):
    new_db = ToneDatabase(tones={
        "tone-a": Tone(base_frequency=220.0, base_volume=0.6, overtones=[]),
        "tone-c": Tone(base_frequency=550.0, base_volume=0.5, overtones=[]),
    })

    tab.refresh(new_db)

    names = [tab._rows[0].combo.itemText(i) for i in range(tab._rows[0].combo.count())]
    assert names == ["—", "tone-a", "tone-c"]
