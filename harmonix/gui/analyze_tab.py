from __future__ import annotations
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QLabel, QFileDialog, QSizePolicy, QComboBox,
)
from PyQt5.QtCore import QTimer, Qt

from harmonix.model import Tone
from harmonix.analysis import MicCapture, analyze_file
from harmonix.synthesis import ToneSynthesizer
from harmonix.gui.tone_plot import TonePlotWidget


class AnalyzeTab(QWidget):
    def __init__(self, on_save_tone, parent: QWidget | None = None) -> None:
        """
        on_save_tone: callable(name: str, tone: Tone)
        """
        super().__init__(parent)
        self._on_save_tone = on_save_tone
        self._current_tone: Tone | None = None
        self._mic = MicCapture()
        self._synth = ToneSynthesizer()

        self._mic_timer = QTimer(self)
        self._mic_timer.setInterval(100)   # 10 Hz refresh
        self._mic_timer.timeout.connect(self._update_from_mic)

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Source buttons ---
        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("Load MP3 / Audio File")
        self._btn_start = QPushButton("Start Recording")
        self._btn_stop = QPushButton("Stop Recording")
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addSpacing(20)
        btn_row.addWidget(QLabel("Base detection:"))
        self._detection_combo = QComboBox()
        self._detection_combo.addItems(["YIN-80", "HPS"])
        btn_row.addWidget(self._detection_combo)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Status line
        self._status = QLabel("Load a file or start recording.")
        self._status.setAlignment(Qt.AlignLeft)
        layout.addWidget(self._status)

        # --- Spectrum plot (takes most of the vertical space) ---
        self._plot = TonePlotWidget()
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._plot, stretch=1)

        # --- Save row ---
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Enter a name for this tone…")
        save_row.addWidget(self._name_edit, stretch=1)
        self._btn_save = QPushButton("Save")
        self._btn_clear = QPushButton("Clear")
        save_row.addWidget(self._btn_save)
        save_row.addWidget(self._btn_clear)
        layout.addLayout(save_row)

        # --- Playback row ---
        play_row = QHBoxLayout()
        self._btn_play = QPushButton("▶  Play")
        self._btn_play_stop = QPushButton("■  Stop")
        play_row.addWidget(self._btn_play)
        play_row.addWidget(self._btn_play_stop)
        play_row.addStretch()
        layout.addLayout(play_row)

        # --- Connect ---
        self._btn_load.clicked.connect(self._on_load)
        self._btn_start.clicked.connect(self._on_start_recording)
        self._btn_stop.clicked.connect(self._on_stop_recording)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_play_stop.clicked.connect(self._on_play_stop)

    # ------------------------------------------------------------------
    def _use_hps(self) -> bool:
        return self._detection_combo.currentText() == "HPS"

    # ------------------------------------------------------------------
    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.aiff);;All Files (*)"
        )
        if not path:
            return
        self._plot.reset_view()
        self._status.setText(f"Analyzing {os.path.basename(path)}…")
        try:
            tone = analyze_file(path, use_hps=self._use_hps())
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
            return
        if tone is None:
            self._status.setText("Could not detect a clear pitch in this file.")
            return
        self._set_tone(tone)
        stem = os.path.splitext(os.path.basename(path))[0]
        if not self._name_edit.text():
            self._name_edit.setText(stem)
        self._status.setText(
            f"Loaded '{os.path.basename(path)}' — "
            f"F0 = {tone.base_frequency:.1f} Hz, {len(tone.overtones)} overtones"
        )

    def _on_start_recording(self) -> None:
        self._plot.reset_view()
        self._mic.start()
        self._mic_timer.start()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status.setText("Recording… (live analysis)")

    def _on_stop_recording(self) -> None:
        self._mic_timer.stop()
        self._mic.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText(
            "Recording stopped."
            + (f" Last: F0 = {self._current_tone.base_frequency:.1f} Hz" if self._current_tone else "")
        )

    def _update_from_mic(self) -> None:
        tone = self._mic.analyze(use_hps=self._use_hps())
        if tone is not None:
            self._set_tone(tone)
            self._status.setText(
                f"Live — F0 = {tone.base_frequency:.1f} Hz, {len(tone.overtones)} overtones"
            )

    def _set_tone(self, tone: Tone) -> None:
        self._current_tone = tone
        self._plot.update_tone(tone)

    def _on_save(self) -> None:
        if self._current_tone is None:
            self._status.setText("No tone to save.")
            return
        name = self._name_edit.text().strip()
        if not name:
            self._status.setText("Please enter a name before saving.")
            return
        self._on_save_tone(name, self._current_tone)
        self._status.setText(f"Saved '{name}' to database.")

    def _on_clear(self) -> None:
        self._current_tone = None
        self._name_edit.clear()
        self._plot.update_tone(None)
        self._status.setText("Cleared.")

    def _on_play(self) -> None:
        if self._current_tone is None:
            self._status.setText("No tone loaded.")
            return
        self._synth.play(self._current_tone, duration=2.0)

    def _on_play_stop(self) -> None:
        self._synth.stop()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._mic_timer.stop()
        self._mic.stop()
        self._synth.stop()
        super().closeEvent(event)
