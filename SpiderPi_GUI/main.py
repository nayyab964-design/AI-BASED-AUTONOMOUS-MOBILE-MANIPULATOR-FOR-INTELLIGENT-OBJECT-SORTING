"""
main.py — entry point.

Run with:  python main.py
Requires:  pip install PySide6 opencv-python ultralytics numpy
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QTimer

from main_window import MainWindow


def make_splash_pixmap():
    """Draws a simple placeholder splash screen (no logo file needed)."""
    pix = QPixmap(520, 300)
    pix.fill(QColor("#14171b"))
    painter = QPainter(pix)
    painter.setPen(QColor("#58a6ff"))
    painter.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter,
                      "SpiderPi\nAutonomous Pick-and-Place\n\nLoading...")
    painter.end()
    return pix


def main():
    app = QApplication(sys.argv)

    qss_path = os.path.join(os.path.dirname(__file__), "styles.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())

    splash = QSplashScreen(make_splash_pixmap())
    splash.show()
    app.processEvents()

    win = MainWindow()

    def show_main():
        win.show()
        splash.finish(win)

    QTimer.singleShot(1200, show_main)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
