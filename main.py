import sys
from PyQt5.QtWidgets import QApplication
from harmonix.gui.app import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("HarmoniX")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
