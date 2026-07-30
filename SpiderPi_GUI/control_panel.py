"""
control_panel.py

Control Panel -- the big mission buttons.

HONESTY NOTE: only Start / Pause / Resume / Stop / Reset are wired to a
REAL action (they control YoloWorker: pause/resume the detection loop,
reset its stability counters). "Home Robot", "Reconnect Robot", and
"Emergency Stop" are included because the spec asked for them, but there
is currently no command channel from the PC to the Pi for arbitrary
commands like "go home" or "stop moving" -- the Pi only ever receives
SEEN/STABLE/LOST detection events, nothing else. Those three buttons log
a clearly-labeled message instead of silently doing nothing, so the
operator always knows what did/didn't happen. Wiring them for real would
need the Pi's YoloServer to understand new message types, which was
intentionally left untouched here.
"""

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QGridLayout, QPushButton


class ControlPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Mission Control", parent)

        self.start_btn = QPushButton("Start Mission")
        self.start_btn.setObjectName("primaryButton")
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("controlButton")
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setObjectName("controlButton")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("controlButton")
        self.home_btn = QPushButton("Home Robot")
        self.home_btn.setObjectName("controlButton")
        self.reconnect_btn = QPushButton("Reconnect Robot")
        self.reconnect_btn.setObjectName("controlButton")
        self.clear_queue_btn = QPushButton("Clear Queue")
        self.clear_queue_btn.setObjectName("controlButton")
        self.emergency_btn = QPushButton("EMERGENCY STOP")
        self.emergency_btn.setObjectName("emergencyButton")

        # NEW: let the operator request which object the robot should go
        # after FIRST. See pi_link.YoloWorker.request_priority() for what
        # this actually does on the wire, and the honesty note below.
        self.priority_blue_btn = QPushButton("Prioritize BlueBox")
        self.priority_blue_btn.setObjectName("controlButton")
        self.priority_green_btn = QPushButton("Prioritize GreenBox")
        self.priority_green_btn.setObjectName("controlButton")

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        grid = QGridLayout()
        grid.addWidget(self.start_btn, 0, 0)
        grid.addWidget(self.pause_btn, 0, 1)
        grid.addWidget(self.resume_btn, 1, 0)
        grid.addWidget(self.stop_btn, 1, 1)
        grid.addWidget(self.reset_btn, 2, 0)
        grid.addWidget(self.home_btn, 2, 1)
        grid.addWidget(self.reconnect_btn, 3, 0)
        grid.addWidget(self.clear_queue_btn, 3, 1)
        grid.addWidget(self.emergency_btn, 4, 0, 1, 2)
        grid.addWidget(self.priority_blue_btn, 5, 0)
        grid.addWidget(self.priority_green_btn, 5, 1)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        self.setLayout(layout)

    def set_running_state(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
