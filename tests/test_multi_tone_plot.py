import numpy as np
import pyqtgraph as pg

from harmonix.model import Overtone, Tone
from harmonix.gui.multi_tone_plot import MultiTonePlotWidget, VoiceEntry, VOICE_COLORS


def _bar_items(plot_widget):
    return [item for item in plot_widget._plot.items() if isinstance(item, pg.BarGraphItem)]


def test_update_voices_places_bars_at_pitch_relative_factors(qtbot):
    widget = MultiTonePlotWidget()
    qtbot.addWidget(widget)

    tone1 = Tone(base_frequency=220.0, base_volume=1.0, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=0.5),
    ])
    tone2 = Tone(base_frequency=330.0, base_volume=1.0, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=0.4),
    ])

    entries = [
        VoiceEntry(tone=tone1, pitch_factor=1.0, volume_scale=1.0, color=VOICE_COLORS[0]),
        VoiceEntry(tone=tone2, pitch_factor=2.0, volume_scale=1.0, color=VOICE_COLORS[1]),
    ]
    widget.update_voices(entries)

    bars = _bar_items(widget)
    # base bars + overtone bars for each voice = 4 BarGraphItems
    assert len(bars) == 4

    all_x = []
    for bar in bars:
        all_x.extend(bar.opts["x"])

    expected_x = [
        np.log10(1.0),   # tone1 base, pitch_factor=1.0
        np.log10(2.0),   # tone1 overtone, 1.0 * 2.0
        np.log10(2.0),   # tone2 base, pitch_factor=2.0
        np.log10(4.0),   # tone2 overtone, 2.0 * 2.0
    ]

    for expected in expected_x:
        assert any(abs(x - expected) < 1e-9 for x in all_x)


def test_update_voices_uses_distinct_colors_per_voice(qtbot):
    widget = MultiTonePlotWidget()
    qtbot.addWidget(widget)

    tone1 = Tone(base_frequency=220.0, base_volume=1.0, overtones=[])
    tone2 = Tone(base_frequency=330.0, base_volume=1.0, overtones=[])

    entries = [
        VoiceEntry(tone=tone1, pitch_factor=1.0, volume_scale=1.0, color=VOICE_COLORS[0]),
        VoiceEntry(tone=tone2, pitch_factor=1.5, volume_scale=1.0, color=VOICE_COLORS[1]),
    ]
    widget.update_voices(entries)

    bars = _bar_items(widget)
    assert len(bars) == 2

    colors = {bar.opts["brush"].color().getRgb()[:3] for bar in bars}
    assert colors == {VOICE_COLORS[0], VOICE_COLORS[1]}


def test_update_voices_does_not_add_text_labels(qtbot):
    widget = MultiTonePlotWidget()
    qtbot.addWidget(widget)

    tone1 = Tone(base_frequency=220.0, base_volume=1.0, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=0.5),
        Overtone(frequency_factor=3.0, volume_factor=0.3),
    ])

    entries = [
        VoiceEntry(tone=tone1, pitch_factor=1.0, volume_scale=1.0, color=VOICE_COLORS[0]),
    ]
    widget.update_voices(entries)

    text_items = [item for item in widget._plot.items() if isinstance(item, pg.TextItem)]
    assert text_items == []


def test_update_voices_empty_list_clears_plot(qtbot):
    widget = MultiTonePlotWidget()
    qtbot.addWidget(widget)

    tone1 = Tone(base_frequency=220.0, base_volume=1.0, overtones=[])
    widget.update_voices([VoiceEntry(tone=tone1, pitch_factor=1.0, volume_scale=1.0, color=VOICE_COLORS[0])])
    assert len(_bar_items(widget)) == 1

    widget.update_voices([])
    assert _bar_items(widget) == []
