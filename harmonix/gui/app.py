from __future__ import annotations
import os

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QAction, QFileDialog,
)

from harmonix.model import Tone, ToneDatabase
from harmonix.gui.analyze_tab import AnalyzeTab
from harmonix.gui.edit_tab import EditTab
from harmonix.gui.synthesize_tab import SynthesizeTab

_DEFAULT_DB_PATH = os.path.join("database", "tones.json")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HarmoniX")
        self.resize(960, 700)

        self._db_path = _DEFAULT_DB_PATH
        self._db = ToneDatabase.load(self._db_path)

        self._build_menu()
        self._build_tabs()

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_act = QAction("Open Database…", self)
        open_act.triggered.connect(self._open_db)
        file_menu.addAction(open_act)

        save_act = QAction("Save Database", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_db)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save Database As…", self)
        save_as_act.triggered.connect(self._save_db_as)
        file_menu.addAction(save_as_act)

    def _build_tabs(self) -> None:
        tabs = QTabWidget()
        self._analyze_tab = AnalyzeTab(on_save_tone=self._on_save_tone)
        self._edit_tab = EditTab(db=self._db)
        self._synth_tab = SynthesizeTab(db=self._db)
        tabs.addTab(self._analyze_tab, "Analyze")
        tabs.addTab(self._edit_tab, "Edit")
        tabs.addTab(self._synth_tab, "Synthesize")
        self.setCentralWidget(tabs)

    # ------------------------------------------------------------------
    def _on_save_tone(self, name: str, tone: Tone) -> None:
        self._db.add(name, tone)
        self._edit_tab.refresh(self._db)
        self._synth_tab.refresh(self._db)
        self._save_db()

    def _open_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Tone Database", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._db = ToneDatabase.load(path)
        self._db_path = path
        self._edit_tab.refresh(self._db)
        self._synth_tab.refresh(self._db)

    def _save_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._db.save(self._db_path)

    def _save_db_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Tone Database As", self._db_path, "JSON Files (*.json)"
        )
        if not path:
            return
        self._db_path = path
        self._save_db()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._save_db()
        super().closeEvent(event)
