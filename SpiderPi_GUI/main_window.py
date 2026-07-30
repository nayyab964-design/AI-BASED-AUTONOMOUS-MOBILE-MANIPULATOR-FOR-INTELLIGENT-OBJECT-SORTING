"""
main_window.py

Assembles every panel into the full dashboard:

  LEFT  : two CameraPanel widgets side by side (Pi raw feed, YOLO feed)
  RIGHT : RobotStatusPanel, QueuePanel, ControlPanel, DetectionSettingsPanel,
          PlacementStatusPanel (stacked, inside a scroll area since that's
          a lot of vertical content)
  BOTTOM: LogPanel (full width)

  Menu bar / toolbar / status bar wrap the whole thing.

All the actual robot/YOLO work happens in pi_link.YoloWorker (a QThread).
This file only wires its signals to the right widgets and its slots to
button clicks -- it contains no networking or YOLO code itself.
"""

import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QToolBar,
    QStatusBar, QMessageBox, QLabel,
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QStyle

from pi_link import YoloWorker
from camera_widget import CameraPanel
from robot_panel import RobotStatusPanel
from queue_panel import QueuePanel
from control_panel import ControlPanel
from detection_panel import DetectionSettingsPanel
from placement_panel import PlacementStatusPanel
from logger_panel import LogPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpiderPi Autonomous Pick-and-Place — Control Dashboard")
        self.resize(1500, 900)

        # ---------------- panels ----------------
        self.pi_camera_panel = CameraPanel(
            "Robot Camera (Raspberry Pi)", "Waiting for Raspberry Pi Camera...",
            show_connect_buttons=True,
        )
        self.yolo_camera_panel = CameraPanel(
            "YOLO Detection (PC)", "Waiting for YOLO Detection...",
            show_connect_buttons=True,
        )
        self.robot_panel = RobotStatusPanel()
        self.queue_panel = QueuePanel()
        self.control_panel = ControlPanel()
        self.detection_panel = DetectionSettingsPanel()
        self.placement_panel = PlacementStatusPanel()
        self.log_panel = LogPanel()

        # ---------------- layout ----------------
        left_col = QHBoxLayout()
        left_col.addWidget(self.pi_camera_panel)
        left_col.addWidget(self.yolo_camera_panel)
        left_widget = QWidget()
        left_widget.setLayout(left_col)

        right_col = QVBoxLayout()
        right_col.addWidget(self.robot_panel)
        right_col.addWidget(self.queue_panel)
        right_col.addWidget(self.control_panel)
        right_col.addWidget(self.detection_panel)
        right_col.addWidget(self.placement_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_col)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        right_scroll.setFixedWidth(380)

        top_split = QHBoxLayout()
        top_split.addWidget(left_widget, stretch=1)
        top_split.addWidget(right_scroll)

        central_layout = QVBoxLayout()
        central_layout.addLayout(top_split, stretch=3)
        central_layout.addWidget(self.log_panel, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self._build_menu_bar()
        self._build_toolbar()
        self._build_status_bar()

        # ---------------- worker thread ----------------
        self.worker = YoloWorker()
        # NEW: pi_camera_panel now shows the Pi's own annotated debug view
        # (line-follow box, AprilTag, step count -- from DEBUG_STREAM_PORT/6002)
        # instead of the raw undistorted frame from CAMERA_STREAM_PORT/6000.
        # raw_frame_ready (6000) is still what feeds the PC-side YOLO model
        # below -- unchanged.
        self.worker.debug_frame_ready.connect(self.pi_camera_panel.update_frame)
        self.worker.annotated_frame_ready.connect(self.yolo_camera_panel.update_frame)
        self.worker.status_changed.connect(self.robot_panel.on_status_changed)
        self.worker.status_changed.connect(self._on_status_for_placement)
        self.worker.connection_changed.connect(self.robot_panel.on_connection_changed)
        self.worker.connection_changed.connect(self._on_connection_for_camera_dots)
        self.worker.connection_changed.connect(self._on_connection_for_status_bar)
        self.worker.model_loaded.connect(self.robot_panel.on_model_loaded)
        self.worker.model_loaded.connect(self.detection_panel.set_model_loaded)
        self.worker.model_loaded.connect(self._on_model_loaded_for_status_bar)
        self.worker.log_message.connect(self.log_panel.log)
        self.worker.start()

        # ---------------- button wiring ----------------
        self.control_panel.start_btn.clicked.connect(self.start_mission)
        self.control_panel.pause_btn.clicked.connect(self.pause_mission)
        self.control_panel.resume_btn.clicked.connect(self.resume_mission)
        self.control_panel.stop_btn.clicked.connect(self.stop_mission)
        self.control_panel.reset_btn.clicked.connect(self.reset_mission)
        self.control_panel.home_btn.clicked.connect(self._not_wired("Home Robot"))
        self.control_panel.reconnect_btn.clicked.connect(self._not_wired("Reconnect Robot"))
        self.control_panel.clear_queue_btn.clicked.connect(self.queue_panel.list_widget.clear)
        self.control_panel.emergency_btn.clicked.connect(self.emergency_stop)
        # NEW: these were previously unwired (button existed in control_panel.py,
        # but nothing called it and pi_link.py had no request_priority() method).
        self.control_panel.priority_blue_btn.clicked.connect(lambda: self.worker.request_priority("BlueBox"))
        self.control_panel.priority_green_btn.clicked.connect(lambda: self.worker.request_priority("GreenBox"))

        self.detection_panel.confidence_changed.connect(self.worker.set_confidence)
        self.detection_panel.start_detect_btn.clicked.connect(self.start_mission)
        self.detection_panel.stop_detect_btn.clicked.connect(self.stop_mission)
        self.detection_panel.reload_model_requested.connect(self._not_wired("Reload Model (restart app for now)"))

        # NEW: the camera stream itself still auto-connects/reconnects on its
        # own (unchanged) -- what these buttons now do is tell the Pi's own
        # main() to begin/halt its physical pick-and-place mission, via the
        # new START/STOP messages on the existing control channel. This is
        # separate from "Start/Stop Mission" in the Control Panel, which only
        # pauses/resumes THIS PC's YOLO detection loop.
        self.pi_camera_panel.connect_btn.clicked.connect(self.connect_robot)
        self.pi_camera_panel.disconnect_btn.clicked.connect(self.disconnect_robot)
        self.yolo_camera_panel.connect_btn.clicked.connect(self.start_mission)
        self.yolo_camera_panel.disconnect_btn.clicked.connect(self.stop_mission)

        # ---------------- clock in the status bar ----------------
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(1000)

        self.log_panel.log("info", "Dashboard started. Press Start Mission to begin detection.")

    # =========================================================
    # Menu bar / toolbar / status bar
    # =========================================================
    def _build_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Save Sequence", self.queue_panel.save_sequence)
        file_menu.addAction("Load Sequence", self.queue_panel.load_sequence)
        file_menu.addAction("Export Logs", self.log_panel.export_logs)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        robot_menu = menubar.addMenu("Robot")
        robot_menu.addAction("Connect", self.start_mission)
        robot_menu.addAction("Disconnect", self.stop_mission)
        robot_menu.addAction("Reconnect", self._not_wired("Reconnect (needs Pi-side command channel)"))

        camera_menu = menubar.addMenu("Camera")
        camera_menu.addAction("Connect Pi Camera", self.connect_robot)
        camera_menu.addAction("Connect YOLO Stream", self.start_mission)
        camera_menu.addAction("Refresh Streams", self._not_wired("Refresh (cosmetic only today)"))

        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction("Confidence", lambda: self.detection_panel.setFocus())
        settings_menu.addAction("Robot Parameters", self._not_wired(
            "Robot Parameters (would need Pi-side parameter channel)"))

        about_menu = menubar.addMenu("About")
        about_menu.addAction("About", self._show_about)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        style = self.style()
        start_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Start", self)
        stop_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop", self)
        reset_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Reset", self)
        snapshot_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Snapshot", self)

        start_action.triggered.connect(self.start_mission)
        stop_action.triggered.connect(self.stop_mission)
        reset_action.triggered.connect(self.reset_mission)
        snapshot_action.triggered.connect(self.yolo_camera_panel._save_snapshot)

        toolbar.addAction(start_action)
        toolbar.addAction(stop_action)
        toolbar.addAction(reset_action)
        toolbar.addAction(snapshot_action)

    def _build_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.sb_robot = QLabel("Robot: --")
        self.sb_camera = QLabel("Camera: --")
        self.sb_yolo = QLabel("YOLO: Not loaded")
        self.sb_mission = QLabel("Mission: Idle")
        self.sb_clock = QLabel(time.strftime("%H:%M:%S"))

        for lbl in (self.sb_robot, self.sb_camera, self.sb_yolo, self.sb_mission, self.sb_clock):
            self.status_bar.addPermanentWidget(lbl)
            self.status_bar.addPermanentWidget(QLabel(" | "))

    # =========================================================
    # Slots
    # =========================================================
    def start_mission(self):
        self.worker.set_paused(False)
        self.control_panel.set_running_state(True)
        self.sb_mission.setText("Mission: Running")
        self.placement_panel.set_total(self.queue_panel.list_widget.count())
        self.log_panel.log("info", "Mission started by operator.")

    def pause_mission(self):
        self.worker.set_paused(True)
        self.control_panel.set_running_state(False)
        self.sb_mission.setText("Mission: Paused")
        self.log_panel.log("info", "Mission paused by operator.")

    def resume_mission(self):
        self.start_mission()

    def stop_mission(self):
        self.worker.set_paused(True)
        self.control_panel.set_running_state(False)
        self.sb_mission.setText("Mission: Stopped")
        self.log_panel.log("warn", "Mission stopped by operator.")

    def reset_mission(self):
        self.worker.reset_stability()
        self.log_panel.log("info", "Stability counters reset by operator.")

    def connect_robot(self):
        """Pi Camera panel's Connect button -- tells the Pi's own main()
        loop to begin its pick-and-place mission (see robot_script's new
        outer wait-for-START loop). Does NOT affect this PC's YOLO detection
        loop -- use the Control Panel's Start Mission for that."""
        self.worker.request_remote_start()
        self.log_panel.log("info", "Connect pressed -- robot mission requested.")

    def disconnect_robot(self):
        """Pi Camera panel's Disconnect button -- tells the Pi to halt its
        mission at the next safe checkpoint (between line-follow frames or
        between objects -- not mid-pick/mid-place, same limitation as the
        Pi script's own 'q' keyboard-quit)."""
        self.worker.request_remote_stop()
        self.log_panel.log("warn", "Disconnect pressed -- robot will stop where it is at the next "
                                    "safe checkpoint.")

    def emergency_stop(self):
        self.worker.set_paused(True)
        self.control_panel.set_running_state(False)
        self.sb_mission.setText("Mission: EMERGENCY STOP")
        self.log_panel.log("error", "EMERGENCY STOP pressed — detection loop paused immediately.")
        QMessageBox.warning(
            self, "Emergency Stop",
            "Detection has been paused immediately.\n\n"
            "Note: this stops the PC from sending new pick commands to the "
            "Pi, but it cannot halt a movement already in progress on the "
            "robot itself — that would require a physical/onboard e-stop "
            "on the Pi side, which this GUI does not control."
        )

    def _on_status_for_placement(self, event_type, detail):
        if event_type.upper() == "STABLE":
            self.placement_panel.mark_one_done()

    def _on_connection_for_camera_dots(self, cam_ok, ctl_ok):
        self.pi_camera_panel.set_connected(cam_ok)
        self.yolo_camera_panel.set_connected(cam_ok)  # same underlying frame source

    def _on_connection_for_status_bar(self, cam_ok, ctl_ok):
        self.sb_robot.setText(f"Robot: {'Connected' if ctl_ok else 'Disconnected'}")
        self.sb_camera.setText(f"Camera: {'Connected' if cam_ok else 'Disconnected'}")

    def _on_model_loaded_for_status_bar(self, loaded):
        self.sb_yolo.setText(f"YOLO: {'Loaded' if loaded else 'Not loaded'}")

    def _tick_clock(self):
        self.sb_clock.setText(time.strftime("%H:%M:%S"))

    def _not_wired(self, feature_name):
        """Returns a click-handler that logs an honest 'not wired up' message
        instead of a button silently doing nothing."""
        def _handler(*_):
            self.log_panel.log("warn", f"{feature_name} — not connected to the robot yet.")
        return _handler

    def _show_about(self):
        QMessageBox.information(
            self, "About",
            "SpiderPi Autonomous Pick-and-Place — Control Dashboard\n\n"
            "Final Year Project GUI.\n"
            "Robot logic, YOLO detection, and TCP protocol are unchanged "
            "from the original project scripts."
        )

    def closeEvent(self, event):
        self.worker.request_stop()
        if not self.worker.wait(3000):
            # Last resort -- the socket retry loops inside PiCameraReceiver/
            # PiControlSender can occasionally be mid-sleep for a moment
            # longer than the wait() timeout; terminate() ensures the app
            # still closes promptly instead of hanging.
            self.worker.terminate()
            self.worker.wait(1000)
        event.accept()
