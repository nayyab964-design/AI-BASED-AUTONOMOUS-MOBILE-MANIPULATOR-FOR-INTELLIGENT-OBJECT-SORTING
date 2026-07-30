"""
queue_panel.py

Object Sequence Panel.

HONESTY NOTE: the Pi's actual pick order comes from TARGET_SEQUENCE, a
plain Python list hardcoded in the Pi's own script, read ONCE at startup.
There is currently no channel for the PC to push a live reordered queue
to the Pi mid-mission. So this panel is a genuinely useful "mission plan"
editor (add/remove/reorder/save/load as JSON) that the operator can use to
PLAN and DOCUMENT the intended sequence, and to visually track progress
against it (see main_window's mark_current()/mark_done() calls) -- but
changing it here does NOT yet reach into the Pi's TARGET_SEQUENCE. Wiring
that up for real would need one small addition on the Pi side (e.g. the
Pi reading target_sequence.json at startup instead of a hardcoded list),
which was intentionally left untouched here.
"""

import json
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QFileDialog, QInputDialog,
    QListWidgetItem,
)
from PySide6.QtCore import Qt


class QueuePanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Object Sequence (Mission Plan)", parent)

        self.list_widget = QListWidget()

        add_btn = QPushButton("Add Object")
        remove_btn = QPushButton("Remove Selected")
        up_btn = QPushButton("Move Up")
        down_btn = QPushButton("Move Down")
        clear_btn = QPushButton("Clear Queue")
        save_btn = QPushButton("Save Sequence")
        load_btn = QPushButton("Load Sequence")

        add_btn.clicked.connect(self.add_object)
        remove_btn.clicked.connect(self.remove_selected)
        up_btn.clicked.connect(self.move_up)
        down_btn.clicked.connect(self.move_down)
        clear_btn.clicked.connect(self.list_widget.clear)
        save_btn.clicked.connect(self.save_sequence)
        load_btn.clicked.connect(self.load_sequence)

        row1 = QHBoxLayout()
        row1.addWidget(add_btn)
        row1.addWidget(remove_btn)
        row2 = QHBoxLayout()
        row2.addWidget(up_btn)
        row2.addWidget(down_btn)
        row3 = QHBoxLayout()
        row3.addWidget(save_btn)
        row3.addWidget(load_btn)
        row3.addWidget(clear_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        self.setLayout(layout)

    def add_object(self):
        name, ok = QInputDialog.getItem(
            self, "Add Object", "Object class:", ["BlueBox", "GreenBox"], 0, False
        )
        if ok and name:
            self.list_widget.addItem(QListWidgetItem(name))

    def remove_selected(self):
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def move_down(self):
        row = self.list_widget.currentRow()
        if 0 <= row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def sequence(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

    def save_sequence(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Sequence", "target_sequence.json", "JSON (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(self.sequence(), f, indent=2)

    def load_sequence(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Sequence", "", "JSON (*.json)")
        if path:
            with open(path) as f:
                items = json.load(f)
            self.list_widget.clear()
            for name in items:
                self.list_widget.addItem(QListWidgetItem(name))

    def mark_current(self, index):
        """Highlight the item currently being pursued (best-effort, by position)."""
        self.list_widget.setCurrentRow(index)

    def mark_done(self, index):
        item = self.list_widget.item(index)
        if item:
            item.setText(item.text() + "  ✓")
