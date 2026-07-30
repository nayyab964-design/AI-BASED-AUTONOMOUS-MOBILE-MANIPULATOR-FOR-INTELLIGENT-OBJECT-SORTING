"""
robot_panel.py

Robot Status Panel.

HONESTY NOTE (read this before wiring anything else to it):
The Pi<->PC protocol only ever sends PC -> Pi detection events
(SEEN / STABLE / LOST). The Pi never reports its own state back. So:

  - "Detected Object", "Confidence", "Coordinates" -> REAL, come straight
    from YoloWorker's status_changed signal.
  - "Connection Status" (camera link / control link / model loaded) -> REAL,
    from YoloWorker's connection_changed / model_loaded signals.
  - "Current Mode" (Idle/Searching/Detecting/Picking/Returning/Placing/
    Finished) -> PARTIALLY inferred. This panel can only show
    Idle / Searching / Detecting / Stable-hand-off, because those are the
    only states visible from the PC side. Picking / Returning / Placing /
    Finished are shown greyed-out with "(needs Pi status link)" because
    nothing today tells the PC when the Pi starts picking, returns, or
    places -- that would require adding one small status message on the
    Pi side, which this task intentionally did not touch.
  - "Battery" is a placeholder (SpiderPi does not currently report battery
    level over this link either).
"""

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QFormLayout, QLabel


PC_VISIBLE_MODES = {"IDLE", "SEARCHING", "DETECTING", "STABLE"}
PI_ONLY_MODES = {"PICKING", "RETURNING", "PLACING", "FINISHED"}  # not observable from PC today


class RobotStatusPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Robot Status", parent)

        self.mode_label = QLabel("IDLE")
        self.mode_label.setStyleSheet("font-weight:700; color:#58a6ff; font-size:15px;")

        self.coords_label = QLabel("--")
        self.object_label = QLabel("--")
        self.confidence_label = QLabel("--")
        self.sequence_label = QLabel("--")

        self.cam_link_label = QLabel("● Pi Camera Link: Disconnected")
        self.ctl_link_label = QLabel("● Pi Control Link: Disconnected")
        self.model_label = QLabel("● YOLO Model: Not loaded")
        for lbl in (self.cam_link_label, self.ctl_link_label, self.model_label):
            lbl.setStyleSheet("color:#f85149; font-weight:600;")

        self.battery_label = QLabel("N/A (not reported by robot)")
        self.messages_label = QLabel("Waiting for events...")
        self.messages_label.setWordWrap(True)

        note = QLabel(
            "Picking / Returning / Placing / Finished stages require an "
            "additional Pi-side status message (not implemented yet) -- "
            "only Idle / Searching / Detecting / Stable are live today."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b949e; font-size:11px; font-style:italic;")

        form = QFormLayout()
        form.addRow("Current Mode:", self.mode_label)
        form.addRow("Coordinates:", self.coords_label)
        form.addRow("Detected Object:", self.object_label)
        form.addRow("Confidence:", self.confidence_label)
        form.addRow("Sequence #:", self.sequence_label)
        form.addRow("Battery:", self.battery_label)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.cam_link_label)
        layout.addWidget(self.ctl_link_label)
        layout.addWidget(self.model_label)
        layout.addWidget(QLabel("Robot Messages:"))
        layout.addWidget(self.messages_label)
        layout.addWidget(note)
        self.setLayout(layout)

        self._seq_num = 0

    # ---- slots ----
    def on_status_changed(self, event_type, detail):
        mode = event_type.upper()
        if mode in PC_VISIBLE_MODES:
            self.mode_label.setText(mode)
            self.mode_label.setStyleSheet("font-weight:700; color:#58a6ff; font-size:15px;")
        self.messages_label.setText(detail)

        if "@" in detail:
            try:
                obj_part, coord_part = detail.split("@")
                self.object_label.setText(obj_part.strip())
                self.coords_label.setText(coord_part.strip())
            except ValueError:
                pass

        if mode == "STABLE":
            self._seq_num += 1
            self.sequence_label.setText(str(self._seq_num))

    def on_connection_changed(self, cam_ok, ctl_ok):
        self.cam_link_label.setText(f"● Pi Camera Link: {'Connected' if cam_ok else 'Disconnected'}")
        self.cam_link_label.setStyleSheet(
            f"color:{'#3fb950' if cam_ok else '#f85149'}; font-weight:600;")
        self.ctl_link_label.setText(f"● Pi Control Link: {'Connected' if ctl_ok else 'Disconnected'}")
        self.ctl_link_label.setStyleSheet(
            f"color:{'#3fb950' if ctl_ok else '#f85149'}; font-weight:600;")

    def on_model_loaded(self, loaded):
        self.model_label.setText(f"● YOLO Model: {'Loaded' if loaded else 'Not loaded'}")
        self.model_label.setStyleSheet(
            f"color:{'#3fb950' if loaded else '#f85149'}; font-weight:600;")
