"""
camera_widget.py

One reusable panel for a single video feed. Used twice in main_window.py:
one instance for "Robot Camera (Raspberry Pi)" and one for "YOLO Detection
(PC)". Each keeps its own FPS counter, resolution label, connection status
dot, and timestamp -- and each is fed purely by Qt signals from
pi_link.YoloWorker, so a disconnect on one feed never blocks the other
(they're just two independent QLabel targets for two independent signals
coming off the same background QThread).
"""

import os
import time
import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QSizePolicy, QFileDialog
)

from utils import cv_frame_to_pixmap, timestamp


class CameraPanel(QGroupBox):
    def __init__(self, title, waiting_text, show_connect_buttons=True, parent=None):
        super().__init__(title, parent)
        self._last_frame = None
        self._frame_count = 0
        self._fps = 0.0
        self._fps_t0 = time.time()

        self.video_label = QLabel(waiting_text)
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(440, 330)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.status_dot = QLabel("● Disconnected")
        self.status_dot.setProperty("class", "statusRed")
        self.status_dot.setStyleSheet("color:#f85149; font-weight:700;")

        self.info_label = QLabel("FPS: -- | Res: -- | " + timestamp())
        self.info_label.setStyleSheet("color:#8b949e; font-size:11px;")

        top_row = QHBoxLayout()
        top_row.addWidget(self.status_dot)
        top_row.addStretch()
        top_row.addWidget(self.info_label)

        btn_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.refresh_btn = QPushButton("Refresh")
        self.snapshot_btn = QPushButton("Snapshot")
        for b in (self.connect_btn, self.disconnect_btn, self.refresh_btn, self.snapshot_btn):
            btn_row.addWidget(b)
        if not show_connect_buttons:
            self.connect_btn.hide()
            self.disconnect_btn.hide()

        self.snapshot_btn.clicked.connect(self._save_snapshot)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.video_label)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    # ---- called from main_window when a new frame signal arrives ----
    def update_frame(self, frame):
        self._last_frame = frame
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_t0 = now

        pix = cv_frame_to_pixmap(frame)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )
        h, w = frame.shape[:2]
        self.info_label.setText(f"FPS: {self._fps:.1f} | Res: {w}x{h} | {timestamp()}")

    def set_connected(self, connected: bool):
        if connected:
            self.status_dot.setText("● Connected")
            self.status_dot.setStyleSheet("color:#3fb950; font-weight:700;")
        else:
            self.status_dot.setText("● Disconnected")
            self.status_dot.setStyleSheet("color:#f85149; font-weight:700;")

    def _save_snapshot(self):
        if self._last_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "snapshot.png", "PNG Image (*.png)")
        if path:
            cv2.imwrite(path, self._last_frame)
