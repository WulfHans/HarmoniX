from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtGui import QFont

from harmonix.model import Tone

_BASE_COLOR = (52, 152, 219, 220)
_OVERTONE_COLOR = (231, 76, 60, 200)
_LABEL_COLOR = (200, 200, 200)
_BAR_WIDTH = 0.006   # width in log10 space (20% of the original 0.03)

_X_MIN_FACTOR = 0.9       # nothing below the base is ever shown
_DEFAULT_X_MAX_HZ = 5000.0
_DEFAULT_Y_MAX = 2.0
_Y_MAX_CAP = 4.0          # autoscaling never expands the y-axis past this — higher is most likely a glitch

_LABEL_Y_OFFSET = 0.04    # base gap above a bar's top
_LABEL_Y_STEP = 0.16      # extra vertical shift applied to staggered labels
_LABEL_X_GAP = 0.05       # log10-space x distance below which neighboring labels are considered to overlap

_TICK_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0]


class TonePlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(background="#1a1a2e")
        self._plot.setLabel("left", "Relative Volume")
        self._plot.setLabel("bottom", "Frequency (relative to base, log scale)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.enableAutoRange(False)

        # Axis upper bounds only ever grow during a session — reset_view()
        # brings them back to the defaults (called when starting a new
        # recording or loading a new file).
        self._x_max_hz = _DEFAULT_X_MAX_HZ
        self._y_max = _DEFAULT_Y_MAX

        self._plot.setXRange(np.log10(_X_MIN_FACTOR), np.log10(20.0), padding=0)
        self._plot.setYRange(0.0, self._y_max, padding=0)

        layout.addWidget(self._plot)

    # ------------------------------------------------------------------
    def reset_view(self) -> None:
        """Reset the axis bounds to their defaults (5 kHz / 2x volume).

        The bounds otherwise only ever grow to fit outliers, so the view
        doesn't jitter during live recording — call this when starting a
        fresh recording or loading a new file so stale expansion doesn't
        linger.
        """
        self._x_max_hz = _DEFAULT_X_MAX_HZ
        self._y_max = _DEFAULT_Y_MAX

    # ------------------------------------------------------------------
    def update_tone(self, tone: Tone | None) -> None:
        self._plot.clear()
        if tone is None:
            return

        # --- Base bar at log10(1) = 0 ---
        base_bar = pg.BarGraphItem(
            x=[0.0], height=[1.0], width=_BAR_WIDTH,
            brush=pg.mkBrush(*_BASE_COLOR),
            pen=pg.mkPen(None),
        )
        self._plot.addItem(base_bar)

        base_label = pg.TextItem(
            f"base\n{tone.base_frequency:.1f} Hz",
            color=_LABEL_COLOR,
            anchor=(0.5, 1.0),
        )
        base_label.setFont(QFont("monospace", 8))
        base_label.setPos(0.0, 1.05)
        self._plot.addItem(base_label)

        # --- Overtone bars ---
        max_overtone_hz = tone.base_frequency
        max_volume = 1.0
        if tone.overtones:
            xs = [np.log10(o.frequency_factor) for o in tone.overtones]
            ys = [o.volume_factor for o in tone.overtones]

            ot_bars = pg.BarGraphItem(
                x=xs, height=ys, width=_BAR_WIDTH,
                brush=pg.mkBrush(*_OVERTONE_COLOR),
                pen=pg.mkPen(None),
            )
            self._plot.addItem(ot_bars)

            # Stagger labels of neighboring bars vertically so their text
            # doesn't overlap when overtones sit close together on the log axis.
            order = sorted(range(len(tone.overtones)), key=lambda i: xs[i])
            prev_x = None
            stagger = 0
            for i in order:
                x, y, ot = xs[i], ys[i], tone.overtones[i]
                if prev_x is not None and abs(x - prev_x) < _LABEL_X_GAP:
                    stagger += 1
                else:
                    stagger = 0
                prev_x = x

                hz = tone.base_frequency * ot.frequency_factor
                lbl = pg.TextItem(
                    f"×{ot.frequency_factor:.2f}\n{hz:.0f} Hz",
                    color=_LABEL_COLOR,
                    anchor=(0.5, 1.0),
                )
                lbl.setFont(QFont("monospace", 7))
                lbl.setPos(x, y + _LABEL_Y_OFFSET + stagger * _LABEL_Y_STEP)
                self._plot.addItem(lbl)

            max_overtone_hz = max(tone.base_frequency * o.frequency_factor for o in tone.overtones)
            max_volume = max(ys)

        # --- Sticky axis bounds: only ever grow until reset_view() is called ---
        # The y-axis is capped at _Y_MAX_CAP — volumes beyond that are most
        # likely analysis glitches and shouldn't blow up the display.
        self._x_max_hz = max(self._x_max_hz, max_overtone_hz)
        self._y_max = min(max(self._y_max, max_volume), _Y_MAX_CAP)

        x_max_factor = self._x_max_hz / tone.base_frequency

        # --- Custom x-axis ticks (log10 of common relative-frequency factors) ---
        tick_factors = [f for f in _TICK_FACTORS if _X_MIN_FACTOR <= f <= x_max_factor]
        ticks = [(np.log10(f), f"×{f:g}") for f in tick_factors]
        self._plot.getAxis("bottom").setTicks([ticks])

        # --- Axis ranges ---
        self._plot.setXRange(np.log10(_X_MIN_FACTOR), np.log10(x_max_factor), padding=0)
        self._plot.setYRange(0.0, self._y_max, padding=0)
