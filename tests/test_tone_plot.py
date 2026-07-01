import numpy as np
import pyqtgraph as pg
import pytest

from harmonix.model import Overtone, Tone
from harmonix.gui.tone_plot import (
    TonePlotWidget, _BAR_WIDTH, _X_MIN_FACTOR, _DEFAULT_X_MAX_HZ, _DEFAULT_Y_MAX,
    _Y_MAX_CAP,
)


@pytest.fixture
def plot(qtbot):
    widget = TonePlotWidget()
    qtbot.addWidget(widget)
    return widget


def _x_range_factors(plot):
    (x0, x1), _ = plot._plot.getViewBox().viewRange()
    return 10 ** x0, 10 ** x1


def _y_range(plot):
    _, (y0, y1) = plot._plot.getViewBox().viewRange()
    return y0, y1


def test_bar_width_is_20_percent_of_original():
    # Original width was 0.03 — it must now be considerably smaller (~20%).
    assert _BAR_WIDTH == pytest.approx(0.03 * 0.2)


def test_default_axis_bounds_are_5khz_and_2x(plot, simple_tone):
    plot.update_tone(simple_tone)

    x_min, x_max = _x_range_factors(plot)
    assert x_min == pytest.approx(_X_MIN_FACTOR)
    assert x_max * simple_tone.base_frequency == pytest.approx(_DEFAULT_X_MAX_HZ)

    y_min, y_max = _y_range(plot)
    assert y_min == 0.0
    assert y_max == pytest.approx(_DEFAULT_Y_MAX)


def test_x_axis_lower_bound_is_always_point_nine_times_base(plot, simple_tone):
    """Nothing below the base is ever shown — overtones never occur there."""
    plot.update_tone(simple_tone)
    x_min, _ = _x_range_factors(plot)
    assert x_min == pytest.approx(0.9)


def test_x_axis_expands_for_overtones_above_5khz(plot):
    tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=40.0, volume_factor=0.1),  # 40*220 = 8800 Hz
    ])
    plot.update_tone(tone)

    _, x_max = _x_range_factors(plot)
    assert x_max * tone.base_frequency >= 8800.0


def test_y_axis_expands_for_volume_above_2x(plot):
    tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=3.5),
    ])
    plot.update_tone(tone)

    _, y_max = _y_range(plot)
    assert y_max >= 3.5


def test_y_axis_autoscaling_is_capped(plot):
    """Volumes beyond _Y_MAX_CAP (4x) are most likely analysis glitches —
    the axis must not stretch to accommodate them."""
    tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=50.0),
    ])
    plot.update_tone(tone)

    _, y_max = _y_range(plot)
    assert y_max == pytest.approx(_Y_MAX_CAP)
    assert plot._y_max == pytest.approx(_Y_MAX_CAP)


def test_axis_expansion_is_sticky_across_updates(plot, simple_tone):
    """Bounds must not shrink back on the next update — that would make the
    view jitter every ~100 ms during live mic recording."""
    loud_tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=4.0),
    ])
    plot.update_tone(loud_tone)
    _, expanded_y_max = _y_range(plot)
    assert expanded_y_max > _DEFAULT_Y_MAX

    plot.update_tone(simple_tone)  # a "quiet" tone, well within defaults
    _, y_max_after = _y_range(plot)
    assert y_max_after == expanded_y_max


def test_reset_view_restores_defaults_after_expansion(plot, simple_tone):
    loud_tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=4.0),
    ])
    plot.update_tone(loud_tone)
    assert plot._y_max > _DEFAULT_Y_MAX

    plot.reset_view()
    assert plot._y_max == pytest.approx(_DEFAULT_Y_MAX)
    assert plot._x_max_hz == pytest.approx(_DEFAULT_X_MAX_HZ)

    # And the next tone is rendered against the restored defaults
    plot.update_tone(simple_tone)
    _, x_max = _x_range_factors(plot)
    assert x_max * simple_tone.base_frequency == pytest.approx(_DEFAULT_X_MAX_HZ)


def _bar_items(plot):
    # PlotWidget.items() also returns persistent chrome (axes, viewbox) that
    # clear() correctly leaves alone — only check the bars we draw ourselves.
    return [item for item in plot._plot.items() if isinstance(item, pg.BarGraphItem)]


def _label_items(plot):
    return [item for item in plot._plot.items() if isinstance(item, pg.TextItem)]


def test_overlapping_overtone_labels_are_staggered_vertically(plot):
    """Two overtones close together on the log axis would otherwise produce
    overlapping text labels — they must be shifted apart in the y-direction."""
    tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=0.5),
        Overtone(frequency_factor=2.02, volume_factor=0.5),
    ])
    plot.update_tone(tone)

    labels = [item for item in _label_items(plot) if "×2.0" in item.textItem.toPlainText()]
    assert len(labels) == 2

    y_positions = sorted(item.pos().y() for item in labels)
    assert y_positions[1] - y_positions[0] > 0.1


def test_well_separated_overtone_labels_are_not_staggered(plot):
    """Labels for bars that are far apart should sit at the same base offset —
    staggering should only kick in for genuinely close neighbors."""
    tone = Tone(base_frequency=220.0, base_volume=0.5, overtones=[
        Overtone(frequency_factor=2.0, volume_factor=0.5),
        Overtone(frequency_factor=8.0, volume_factor=0.5),
    ])
    plot.update_tone(tone)

    labels = [item for item in _label_items(plot) if "Hz" in item.textItem.toPlainText()
              and "base" not in item.textItem.toPlainText()]
    assert len(labels) == 2

    y_positions = [item.pos().y() for item in labels]
    assert y_positions[0] == pytest.approx(y_positions[1])


def test_update_tone_with_none_clears_plot(plot, simple_tone):
    plot.update_tone(simple_tone)
    assert len(_bar_items(plot)) > 0

    plot.update_tone(None)
    assert len(_bar_items(plot)) == 0
