from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from harmonix.model import Tone
from harmonix.gui.tone_plot import _BAR_WIDTH, _Y_MAX_CAP

# One color per voice slot (base bars). Overtone bars use the same hue at a
# lower alpha so a tone's whole spectrum reads as "one color".
VOICE_COLORS: list[tuple[int, int, int]] = [
    (52, 152, 219),   # blue
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (241, 196, 15),   # yellow
]
_BASE_ALPHA = 220
_OVERTONE_ALPHA = 130

_DEFAULT_X_MIN_FACTOR = 0.2
_DEFAULT_X_MAX_FACTOR = 5.0
_DEFAULT_Y_MAX = 2.0

# Tick factors spanning both directions from ×1 (pitch range is ×0.25–×4,
# overtones extend further up).
_TICK_FACTORS = sorted(set(
    [0.125, 0.1667, 0.2, 0.25, 0.3333, 0.5, 0.6667, 0.75] +
    [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0]
))


@dataclass
class VoiceEntry:
    tone: Tone
    pitch_factor: float            # this voice's base, relative to Tone 1's effective frequency
    volume_scale: float            # 0.0-1.0 attenuation applied to bar heights
    color: tuple[int, int, int]


class MultiTonePlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(background="#1a1a2e")
        self._plot.setLabel("left", "Relative Volume")
        self._plot.setLabel("bottom", "Frequency (relative to Tone 1, log scale)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.enableAutoRange(False)

        self._x_min_factor = _DEFAULT_X_MIN_FACTOR
        self._x_max_factor = _DEFAULT_X_MAX_FACTOR
        self._y_max = _DEFAULT_Y_MAX

        self._plot.setXRange(np.log10(self._x_min_factor), np.log10(self._x_max_factor), padding=0)
        self._plot.setYRange(0.0, self._y_max, padding=0)

        layout.addWidget(self._plot)

    # ------------------------------------------------------------------
    def reset_view(self) -> None:
        """Reset the axis bounds to their defaults (see TonePlotWidget.reset_view)."""
        self._x_min_factor = _DEFAULT_X_MIN_FACTOR
        self._x_max_factor = _DEFAULT_X_MAX_FACTOR
        self._y_max = _DEFAULT_Y_MAX

    # ------------------------------------------------------------------
    def update_voices(self, entries: list[VoiceEntry]) -> None:
        self._plot.clear()

        if entries:
            x_min = self._x_min_factor
            x_max = self._x_max_factor
            y_max = self._y_max

            for entry in entries:
                base_x = np.log10(entry.pitch_factor)
                base_h = 1.0 * entry.volume_scale

                base_bar = pg.BarGraphItem(
                    x=[base_x], height=[base_h], width=_BAR_WIDTH,
                    brush=pg.mkBrush(*entry.color, _BASE_ALPHA),
                    pen=pg.mkPen(None),
                )
                self._plot.addItem(base_bar)

                x_min = min(x_min, entry.pitch_factor)
                x_max = max(x_max, entry.pitch_factor)
                y_max = max(y_max, base_h)

                if entry.tone.overtones:
                    factors = [entry.pitch_factor * o.frequency_factor for o in entry.tone.overtones]
                    xs = [np.log10(f) for f in factors]
                    ys = [o.volume_factor * entry.volume_scale for o in entry.tone.overtones]

                    ot_bars = pg.BarGraphItem(
                        x=xs, height=ys, width=_BAR_WIDTH,
                        brush=pg.mkBrush(*entry.color, _OVERTONE_ALPHA),
                        pen=pg.mkPen(None),
                    )
                    self._plot.addItem(ot_bars)

                    x_min = min(x_min, min(factors))
                    x_max = max(x_max, max(factors))
                    y_max = max(y_max, max(ys))

            # Sticky bounds: only ever grow until reset_view() is called
            self._x_min_factor = x_min
            self._x_max_factor = x_max
            self._y_max = min(y_max, _Y_MAX_CAP)

            tick_factors = [f for f in _TICK_FACTORS if self._x_min_factor <= f <= self._x_max_factor]
            ticks = [(np.log10(f), f"×{f:g}") for f in tick_factors]
            self._plot.getAxis("bottom").setTicks([ticks])

        self._plot.setXRange(np.log10(self._x_min_factor), np.log10(self._x_max_factor), padding=0)
        self._plot.setYRange(0.0, self._y_max, padding=0)
