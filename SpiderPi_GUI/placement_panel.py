"""
placement_panel.py

Placement Status Panel.

HONESTY NOTE: AprilTag detection and placement success/failure happen
entirely on the Pi (locate_place_zone() / navigate_and_place_in_boundary()
in the robot script) and are never reported back to the PC over the
existing protocol. So "Current AprilTag", "Target Box", and "Placement
Success/Failure" cannot be shown live here -- they're left as explicit
"Not reported by robot" placeholders rather than guessed.

What IS genuinely derivable on the PC side: how many STABLE hand-offs
have been sent to the Pi vs. how many objects are in the planned queue
(queue_panel.py). That's used to drive "Objects Completed / Remaining"
and the progress bar -- it reflects "objects handed off for picking",
NOT confirmed placements, and is labeled that way in the UI.
"""

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QFormLayout, QLabel, QProgressBar


class PlacementStatusPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Placement Status", parent)

        self.apriltag_label = QLabel("Not reported by robot")
        self.target_box_label = QLabel("Not reported by robot")
        self.completed_label = QLabel("0")
        self.remaining_label = QLabel("--")
        self.success_label = QLabel("Not reported by robot")
        self.failure_label = QLabel("Not reported by robot")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_note = QLabel("(objects handed off to Pi, not confirmed placements)")
        self.progress_note.setStyleSheet("color:#8b949e; font-size:10px; font-style:italic;")

        form = QFormLayout()
        form.addRow("Current AprilTag:", self.apriltag_label)
        form.addRow("Target Box:", self.target_box_label)
        form.addRow("Objects Handed Off:", self.completed_label)
        form.addRow("Remaining (planned):", self.remaining_label)
        form.addRow("Placement Success:", self.success_label)
        form.addRow("Placement Failure:", self.failure_label)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(QLabel("Mission Progress:"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_note)
        self.setLayout(layout)

        self._done = 0
        self._total = 0

    def set_total(self, total):
        self._total = total
        self.remaining_label.setText(str(max(total - self._done, 0)))
        self._refresh_bar()

    def mark_one_done(self):
        self._done += 1
        self.completed_label.setText(str(self._done))
        self.remaining_label.setText(str(max(self._total - self._done, 0)))
        self._refresh_bar()

    def _refresh_bar(self):
        pct = int((self._done / self._total) * 100) if self._total else 0
        self.progress_bar.setValue(pct)
