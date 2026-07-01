from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QSizePolicy, QHeaderView,
    QPushButton, QSlider,
)
from PyQt5.QtCore import Qt, QTimer

from harmonix.model import Overtone, Tone, ToneDatabase
from harmonix.gui.tone_plot import TonePlotWidget
from harmonix.synthesis import ToneSynthesizer


class EditTab(QWidget):
    def __init__(self, db: ToneDatabase, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._active_flags: list[bool] = []
        self._synth = ToneSynthesizer()
        self._build_ui()
        self._populate_combo()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Tone selector ---
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Tone:"))
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo.currentIndexChanged.connect(self._on_tone_selected)
        select_row.addWidget(self._combo)
        layout.addLayout(select_row)

        # --- Spectrum plot ---
        self._plot = TonePlotWidget()
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._plot, stretch=1)

        # --- Overtone table: Active | Factor | Volume | (action) ---
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Active", "Factor", "Volume", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self._table.verticalHeader().setVisible(False)
        self._table.setMaximumHeight(220)
        self._table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self._table)

        # --- Slider (0 = all off … n = all on) ---
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Overtones:"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider)
        layout.addLayout(slider_row)

        # --- Play / Stop ---
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
    def _populate_combo(self) -> None:
        self._combo.blockSignals(True)
        prev = self._combo.currentText()
        self._combo.clear()
        for name in sorted(self._db.tones):
            self._combo.addItem(name)
        idx = self._combo.findText(prev)
        self._combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._combo.blockSignals(False)
        self._on_tone_selected()

    def _on_tone_selected(self) -> None:
        name = self._combo.currentText()
        tone = self._db.tones.get(name)
        if tone is None:
            self._active_flags = []
            self._plot.update_tone(None)
            self._fill_table([])
            self._update_slider(0, 0)
            return
        tone.overtones.sort(key=lambda o: o.frequency_factor)
        n = len(tone.overtones)
        self._active_flags = [True] * n
        self._plot.reset_view()
        self._fill_table(tone.overtones)
        self._update_slider(n, n)
        self._refresh_plot(tone)

    # ------------------------------------------------------------------
    def _fill_table(self, overtones: list[Overtone]) -> None:
        self._table.blockSignals(True)
        n = len(overtones)
        self._table.setRowCount(n + 1)  # +1 for the add row

        for row, ot in enumerate(overtones):
            # Col 0: active checkbox (UI-only)
            chk = QTableWidgetItem()
            active = row < len(self._active_flags) and self._active_flags[row]
            chk.setCheckState(Qt.Checked if active else Qt.Unchecked)
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            self._table.setItem(row, 0, chk)

            # Col 1: factor — editable, updates model
            f_item = QTableWidgetItem(f"{ot.frequency_factor:.4f}")
            f_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, f_item)

            # Col 2: volume — editable, updates model
            v_item = QTableWidgetItem(f"{ot.volume_factor:.4f}")
            v_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 2, v_item)

            # Col 3: red delete button
            del_btn = QPushButton("✕")
            del_btn.setStyleSheet("color: red; border: none; font-weight: bold; font-size: 12px;")
            del_btn.setFixedWidth(28)
            del_btn.clicked.connect(self._on_delete_row)
            self._table.setCellWidget(row, 3, del_btn)

        # Add row — blank checkbox, editable factor/volume, "Add" button
        blank = QTableWidgetItem()
        blank.setFlags(Qt.NoItemFlags)
        self._table.setItem(n, 0, blank)

        self._table.setItem(n, 1, QTableWidgetItem(""))
        self._table.setItem(n, 2, QTableWidgetItem(""))

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_row)
        self._table.setCellWidget(n, 3, add_btn)

        self._table.blockSignals(False)

    # ------------------------------------------------------------------
    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        name = self._combo.currentText()
        tone = self._db.tones.get(name)
        if tone is None or row >= len(tone.overtones):
            return  # add row or empty database — nothing to update

        if col == 0:
            if row < len(self._active_flags):
                self._active_flags[row] = item.checkState() == Qt.Checked
            self._refresh_plot(tone)

        elif col == 1:
            try:
                tone.overtones[row].frequency_factor = round(float(item.text()), 4)
                # Defer re-sort so we don't rebuild the table from inside itemChanged
                QTimer.singleShot(0, lambda t=tone: self._refill_sorted(t))
            except ValueError:
                self._table.blockSignals(True)
                item.setText(f"{tone.overtones[row].frequency_factor:.4f}")
                self._table.blockSignals(False)
            self._refresh_plot(tone)

        elif col == 2:
            try:
                tone.overtones[row].volume_factor = round(float(item.text()), 4)
            except ValueError:
                self._table.blockSignals(True)
                item.setText(f"{tone.overtones[row].volume_factor:.4f}")
                self._table.blockSignals(False)
            self._refresh_plot(tone)

    def _refill_sorted(self, tone: Tone) -> None:
        self._resort_preserve_flags(tone)
        self._fill_table(tone.overtones)
        self._update_slider(len(tone.overtones), self._slider.value())

    # ------------------------------------------------------------------
    def _on_delete_row(self) -> None:
        btn = self.sender()
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, 3) is btn:
                self._delete_row(row)
                return

    def _delete_row(self, row: int) -> None:
        name = self._combo.currentText()
        tone = self._db.tones.get(name)
        if tone is None or row >= len(tone.overtones):
            return
        del tone.overtones[row]
        if row < len(self._active_flags):
            del self._active_flags[row]
        n = len(tone.overtones)
        self._fill_table(tone.overtones)
        self._update_slider(n, min(self._slider.value(), n))
        self._refresh_plot(tone)

    def _on_add_row(self) -> None:
        name = self._combo.currentText()
        tone = self._db.tones.get(name)
        if tone is None:
            return
        add_row = len(tone.overtones)
        f_item = self._table.item(add_row, 1)
        v_item = self._table.item(add_row, 2)
        try:
            factor = round(float(f_item.text() if f_item else ""), 4)
            vol = round(float(v_item.text() if v_item else ""), 4)
        except ValueError:
            return
        tone.overtones.append(Overtone(frequency_factor=factor, volume_factor=vol))
        self._active_flags.append(True)
        self._resort_preserve_flags(tone)
        n = len(tone.overtones)
        self._fill_table(tone.overtones)
        self._update_slider(n, n)
        self._refresh_plot(tone)

    # ------------------------------------------------------------------
    def _resort_preserve_flags(self, tone: Tone) -> None:
        """Sort overtones by factor in-place, reordering _active_flags to match."""
        paired = sorted(zip(tone.overtones, self._active_flags),
                        key=lambda x: x[0].frequency_factor)
        tone.overtones[:] = [p[0] for p in paired]
        self._active_flags = [p[1] for p in paired]

    def _update_slider(self, n: int, value: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setMaximum(n)
        self._slider.setValue(max(0, min(value, n)))
        self._slider.blockSignals(False)

    def _on_slider_changed(self, value: int) -> None:
        """Activate the first `value` overtones (by sorted frequency factor)."""
        for i in range(len(self._active_flags)):
            self._active_flags[i] = i < value
        # Mirror new state onto the checkboxes without triggering itemChanged
        self._table.blockSignals(True)
        for i in range(len(self._active_flags)):
            chk = self._table.item(i, 0)
            if chk is not None:
                chk.setCheckState(Qt.Checked if self._active_flags[i] else Qt.Unchecked)
        self._table.blockSignals(False)
        name = self._combo.currentText()
        tone = self._db.tones.get(name)
        if tone is not None:
            self._refresh_plot(tone)

    # ------------------------------------------------------------------
    def _refresh_plot(self, tone: Tone) -> None:
        active = [
            ot for i, ot in enumerate(tone.overtones)
            if i < len(self._active_flags) and self._active_flags[i]
        ]
        self._plot.update_tone(Tone(
            base_frequency=tone.base_frequency,
            base_volume=tone.base_volume,
            overtones=active,
        ))

    def _on_play(self) -> None:
        name = self._combo.currentText()
        tone = self._db.tones.get(name)
        if tone is None:
            return
        active = [
            ot for i, ot in enumerate(tone.overtones)
            if i < len(self._active_flags) and self._active_flags[i]
        ]
        self._synth.play(Tone(
            base_frequency=tone.base_frequency,
            base_volume=tone.base_volume,
            overtones=active,
        ), duration=2.0)

    def _on_play_stop(self) -> None:
        self._synth.stop()

    # ------------------------------------------------------------------
    def refresh(self, db: ToneDatabase) -> None:
        self._db = db
        self._populate_combo()

    def closeEvent(self, event) -> None:
        self._synth.stop()
        super().closeEvent(event)
