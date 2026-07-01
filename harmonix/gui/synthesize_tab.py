from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QSlider, QPushButton, QSizePolicy, QStyle, QStyleOptionSlider,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor

from harmonix.model import Tone, ToneDatabase
from harmonix.synthesis import ToneSynthesizer
from harmonix.gui.multi_tone_plot import MultiTonePlotWidget, VoiceEntry, VOICE_COLORS


class LogPitchSlider(QSlider):
    """A horizontal slider mapping ×0.25–×4 (±2 octaves) onto a log scale,
    with ×1 in the middle. Internally counts in cents (1200 per octave) so
    `factor()` is a simple power-of-two conversion."""

    _CENTS_RANGE = 2400  # ±2 octaves

    # (frequency factor, is_octave) — marked tick positions: octaves, plus
    # perfect 4th/5th and major/minor 3rd (and their octave-shifted forms).
    _TICKS: list[tuple[float, bool]] = [
        (0.25, True), (0.3, False), (0.3125, False), (1 / 3, False), (0.375, False),
        (0.5, True), (0.6, False), (0.625, False), (2 / 3, False), (0.75, False),
        (1.0, True),
        (1.2, False), (1.25, False), (4 / 3, False), (1.5, False),
        (2.0, True), (2.4, False), (2.5, False), (8 / 3, False), (3.0, False),
        (4.0, True),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self.setMinimum(-self._CENTS_RANGE)
        self.setMaximum(self._CENTS_RANGE)
        self.setValue(0)
        self.setSingleStep(10)
        self.setPageStep(100)
        self.setMinimumHeight(28)

    def factor(self) -> float:
        return float(2.0 ** (self.value() / 1200.0))

    def setFactor(self, factor: float) -> None:
        cents = int(round(1200.0 * np.log2(factor)))
        cents = max(self.minimum(), min(self.maximum(), cents))
        self.setValue(cents)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)

        for factor, is_octave in self._TICKS:
            cents = int(round(1200.0 * np.log2(factor)))
            x = QStyle.sliderPositionFromValue(
                self.minimum(), self.maximum(), cents, groove.width()
            ) + groove.x()
            tick_height = 6 if is_octave else 3
            color = QColor(190, 190, 190) if is_octave else QColor(110, 110, 110)
            painter.setPen(QPen(color))
            painter.drawLine(x, groove.bottom() + 1, x, groove.bottom() + 1 + tick_height)
        painter.end()


class _VoiceRow(QWidget):
    """One playback voice: active checkbox, tone selector, volume + pitch sliders."""

    changed = pyqtSignal()

    def __init__(self, color: tuple[int, int, int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)

        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 3px;"
        )
        layout.addWidget(swatch)

        self.combo = QComboBox()
        self.combo.setMinimumWidth(140)
        layout.addWidget(self.combo)

        layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        layout.addWidget(self.volume_slider, stretch=1)
        self.volume_label = QLabel("100%")
        self.volume_label.setFixedWidth(40)
        layout.addWidget(self.volume_label)

        layout.addWidget(QLabel("Pitch:"))
        self.pitch_slider = LogPitchSlider()
        layout.addWidget(self.pitch_slider, stretch=1)
        self.pitch_label = QLabel("×1.00")
        self.pitch_label.setFixedWidth(50)
        layout.addWidget(self.pitch_label)

        self.checkbox.stateChanged.connect(self._emit_changed)
        self.combo.currentIndexChanged.connect(self._emit_changed)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.pitch_slider.valueChanged.connect(self._on_pitch_changed)

    # ------------------------------------------------------------------
    def _on_volume_changed(self, value: int) -> None:
        self.volume_label.setText(f"{value}%")
        self.changed.emit()

    def _on_pitch_changed(self, _value: int) -> None:
        self.pitch_label.setText(f"×{self.pitch_slider.factor():.2f}")
        self.changed.emit()

    def _emit_changed(self, *_args) -> None:
        self.changed.emit()

    # ------------------------------------------------------------------
    @property
    def volume_scale(self) -> float:
        return self.volume_slider.value() / 100.0

    @property
    def pitch_factor(self) -> float:
        return self.pitch_slider.factor()

    def is_active(self) -> bool:
        return self.checkbox.isChecked()

    def selected_tone_name(self) -> str | None:
        return self.combo.currentText() if self.combo.currentIndex() > 0 else None

    def populate(self, names: list[str]) -> None:
        prev = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("—")
        for name in names:
            self.combo.addItem(name)
        idx = self.combo.findText(prev)
        self.combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo.blockSignals(False)


class SynthesizeTab(QWidget):
    def __init__(self, db: ToneDatabase, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._synth = ToneSynthesizer()
        self._initialized = False
        self._build_ui()
        self.refresh(db)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._plot = MultiTonePlotWidget()
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._plot, stretch=1)

        self._rows: list[_VoiceRow] = []
        for i in range(4):
            row = _VoiceRow(VOICE_COLORS[i])
            row.changed.connect(self._update_plot)
            layout.addWidget(row)
            self._rows.append(row)

        play_row = QHBoxLayout()
        self._btn_play = QPushButton("▶  Play")
        self._btn_play_stop = QPushButton("■  Stop")
        play_row.addWidget(self._btn_play)
        play_row.addWidget(self._btn_play_stop)
        play_row.addStretch()
        layout.addLayout(play_row)

        self._btn_play.clicked.connect(self._on_play)
        self._btn_play_stop.clicked.connect(self._on_play_stop)

    # ------------------------------------------------------------------
    def refresh(self, db: ToneDatabase) -> None:
        self._db = db
        names = sorted(db.tones)
        for row in self._rows:
            row.populate(names)

        if not self._initialized and names:
            self._rows[0].combo.setCurrentIndex(1)  # first real tone
            self._rows[0].checkbox.setChecked(True)
            self._initialized = True

        self._update_plot()

    # ------------------------------------------------------------------
    def _active_rows(self) -> list[tuple[int, _VoiceRow, Tone]]:
        result = []
        for i, row in enumerate(self._rows):
            if not row.is_active():
                continue
            name = row.selected_tone_name()
            if name is None:
                continue
            tone = self._db.tones.get(name)
            if tone is None:
                continue
            result.append((i, row, tone))
        return result

    def _reference_frequency(self) -> float | None:
        """Tone 1's effective frequency (its recorded base × its pitch slider),
        used as the playback pitch reference for voices 2-4."""
        row0 = self._rows[0]
        name0 = row0.selected_tone_name()
        if name0 is None:
            return None
        tone0 = self._db.tones.get(name0)
        if tone0 is None:
            return None
        return tone0.base_frequency * row0.pitch_factor

    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        entries = [
            VoiceEntry(
                tone=tone,
                pitch_factor=1.0 if i == 0 else row.pitch_factor,
                volume_scale=row.volume_scale,
                color=VOICE_COLORS[i],
            )
            for i, row, tone in self._active_rows()
        ]
        self._plot.update_voices(entries)

    def _on_play(self) -> None:
        ref_freq = self._reference_frequency()
        parts: list[tuple[Tone, float, float]] = []
        for i, row, tone in self._active_rows():
            if i == 0:
                freq = tone.base_frequency * row.pitch_factor
            elif ref_freq is not None:
                freq = ref_freq * row.pitch_factor
            else:
                # Tone 1 has no selection — fall back to this tone's own pitch
                freq = tone.base_frequency * row.pitch_factor
            parts.append((tone, freq, tone.base_volume * row.volume_scale))

        if not parts:
            return
        self._synth.play_multi(parts, duration=2.0)

    def _on_play_stop(self) -> None:
        self._synth.stop()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._synth.stop()
        super().closeEvent(event)
