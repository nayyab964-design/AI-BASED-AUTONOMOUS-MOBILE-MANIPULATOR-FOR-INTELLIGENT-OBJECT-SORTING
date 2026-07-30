"""logger_panel.py -- scrolling, timestamped, colour-coded system log."""

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QFileDialog
from utils import timestamp

_LEVEL_COLORS = {
    "info": "#c9d1d9",
    "warn": "#d29922",
    "error": "#f85149",
}


class LogPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Console / System Log", parent)

        self.console = QTextEdit()
        self.console.setObjectName("logConsole")
        self.console.setReadOnly(True)

        export_btn = QPushButton("Export Logs")
        clear_btn = QPushButton("Clear")
        export_btn.clicked.connect(self.export_logs)
        clear_btn.clicked.connect(self.console.clear)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(export_btn)
        btn_row.addWidget(clear_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.console)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def log(self, level, text):
        color = _LEVEL_COLORS.get(level, "#c9d1d9")
        self.console.append(f'<span style="color:#6e7681">[{timestamp()}]</span> '
                             f'<span style="color:{color}">{text}</span>')

    def export_logs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Logs", "spiderpi_log.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.console.toPlainText())
