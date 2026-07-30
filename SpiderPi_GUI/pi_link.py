"""
pi_link.py

Everything that actually talks to the Pi / runs YOLO lives here, kept as
close as possible to the original pc_yolo_sender.py:

  - PiCameraReceiver  -- UNCHANGED. Receives JPEG frames from the Pi's
                         camera_stream_server() on CAMERA_STREAM_PORT.
  - PiControlSender   -- UNCHANGED. Sends "TYPE,cls_name,cx,cy\n" to the
                         Pi's YoloServer on CONTROL_PORT.
  - YoloWorker        -- the original main() while-loop, moved into a
                         QThread so the GUI can stay responsive. The
                         detection/stability logic inside (STABILITY_
                         FRAMES_REQUIRED, MISS_TOLERANCE_FRAMES) is IDENTICAL
                         to pc_yolo_sender.py -- only the output changed
                         (Qt signals instead of cv2.imshow), and the
                         confidence threshold is now live-adjustable from
                         the GUI (was a fixed constant before).

IMPORTANT / HONESTY NOTE:
This file only knows what pc_yolo_sender.py always knew: the incoming
camera frames and its own YOLO detections. It has NO visibility into the
Pi's actual robot state (whether it's line-following, picking, returning,
searching an AprilTag, or placing) -- the existing Pi<->PC protocol never
sends that information back. Anywhere the GUI shows a "robot mode" or
"mission stage", it is inferred/placeholder from YOLO events only, and is
labeled as such in robot_panel.py. Getting the *real* stage would require
adding a small new status message on the Pi side, which was intentionally
NOT done here per "do not modify existing robot algorithms".
"""

import cv2
import socket
import struct
import threading
import time
import numpy as np

from ultralytics import YOLO
from PySide6.QtCore import QThread, Signal


# =========================================================
# CONFIG -- identical defaults to pc_yolo_sender.py
# =========================================================
PI_IP = '192.168.149.1'
CAMERA_STREAM_PORT = 6000
DEBUG_STREAM_PORT = 6002   # NEW: matches the Pi's debug_stream_server() -- serves the
                           # annotated display_frame (line-follow box, AprilTag, step
                           # count) that camera_stream_server()/CAMERA_STREAM_PORT does
                           # NOT carry (that port only ever sends the raw undistorted
                           # frame used for YOLO). Nothing on the GUI side previously
                           # connected to this port -- that was the actual bug.
CONTROL_PORT = 5001

YOLO_MODEL_PATH = r"C:\Users\PMLS\FYP_PROJECT\yolov8_results-20260720T050149Z-1-001\yolov8_results\custom_train\weights\best.pt"
YOLO_CONF_DEFAULT = 0.2
CLASSES_OF_INTEREST = ['BlueBox', 'GreenBox']

STABILITY_FRAMES_REQUIRED = 5
MISS_TOLERANCE_FRAMES = 3


# =========================================================
# CAMERA STREAM RECEIVER (from SpiderPi) -- UNCHANGED
# =========================================================
class PiCameraReceiver:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.frame = None
        self.running = True
        self.connected = False          # NEW: exposed for the GUI's connection dot
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def _receive_loop(self):
        payload_size = struct.calcsize(">L")
        while self.running:
            try:
                cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                cam_sock.connect((self.ip, self.port))
                self.connected = True
                print("Connected to SpiderPi camera stream.")

                data = b""
                while self.running:
                    while len(data) < payload_size:
                        packet = cam_sock.recv(4096)
                        if not packet:
                            raise ConnectionError("Camera server closed connection")
                        data += packet

                    packed_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack(">L", packed_size)[0]

                    while len(data) < msg_size:
                        packet = cam_sock.recv(4096)
                        if not packet:
                            raise ConnectionError("Camera server closed connection")
                        data += packet

                    frame_data = data[:msg_size]
                    data = data[msg_size:]

                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.frame = frame

            except Exception as e:
                self.connected = False
                print(f"Camera stream error: {e} -- retrying in 2s")
                time.sleep(2)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


# =========================================================
# CONTROL SENDER (PC -> Pi) -- UNCHANGED
# =========================================================
class PiControlSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False           # NEW: exposed for the GUI's connection dot
        self._send_lock = threading.Lock()  # NEW: guards sendall() -- send_event() is now
                                             # called from both the worker thread (SEEN/
                                             # STABLE/LOST) and the GUI thread (START/STOP
                                             # from the Pi Camera panel's Connect/Disconnect)
        self._connect()

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, self.port))
            self.connected = True
            print("Connected to Pi control channel.")
        except Exception as e:
            self.connected = False
            print(f"Could not connect control channel: {e}")
            self.sock = None

    def send_event(self, event_type, cls_name, cx, cy):
        message = f"{event_type},{cls_name},{cx},{cy}\n"
        with self._send_lock:
            if self.sock is None:
                self._connect()
            try:
                self.sock.sendall(message.encode())
                print(f"Sent to Pi: {message.strip()}")
            except Exception as e:
                self.connected = False
                print(f"Failed to send, reconnecting: {e}")
                self._connect()


# =========================================================
# YoloWorker -- original main() loop, moved into a QThread
# =========================================================
class YoloWorker(QThread):
    raw_frame_ready = Signal(np.ndarray)
    annotated_frame_ready = Signal(np.ndarray)
    debug_frame_ready = Signal(np.ndarray)  # NEW: the Pi's own annotated display_frame
                                             # (line-follow/AprilTag/step-count overlays),
                                             # from DEBUG_STREAM_PORT -- distinct from
                                             # raw_frame_ready (undistorted, no overlays)
                                             # and annotated_frame_ready (PC-side YOLO boxes).
    status_changed = Signal(str, str)       # (event_type, detail) e.g. ("STABLE", "BlueBox @ (320,240)")
    log_message = Signal(str, str)          # (level, text) -- for the log panel: level in {info,warn,error}
    connection_changed = Signal(bool, bool)  # (camera_connected, control_connected)
    model_loaded = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._paused = True          # starts paused; GUI's Start button un-pauses it
        self._reset_flag = False
        self._conf = YOLO_CONF_DEFAULT
        self._conf_lock = threading.Lock()
        self.camera = None
        self.debug_camera = None  # NEW: second PiCameraReceiver, on DEBUG_STREAM_PORT
        self.control = None
        self.model = None

    # ---- thread-safe setters the GUI calls ----
    def request_stop(self):
        self._running = False

    def set_paused(self, paused):
        self._paused = paused

    def reset_stability(self):
        self._reset_flag = True

    def set_confidence(self, conf: float):
        with self._conf_lock:
            self._conf = conf

    def get_confidence(self) -> float:
        with self._conf_lock:
            return self._conf

    def snapshot_paths(self):
        return getattr(self, "_last_raw", None), getattr(self, "_last_annotated", None)

    # ---- NEW: remote mission control, called from the GUI thread by the
    # Pi Camera panel's Connect/Disconnect buttons. Reuses the existing
    # "TYPE,cls_name,cx,cy\n" control-channel format (send_event) -- no new
    # wire protocol, just two new TYPE values ("START"/"STOP") that the Pi's
    # updated YoloServer/main() now understand. This is separate from this
    # GUI's own Start/Stop Mission buttons, which only pause/resume THIS
    # PC's YOLO detection loop -- these two instead tell the Pi itself to
    # begin/halt its physical pick-and-place mission. ----
    def request_remote_start(self):
        if self.control is not None:
            self.control.send_event("START", "", 0, 0)
            self.log_message.emit("info", "Sent START to Pi -- robot mission should begin.")
        else:
            self.log_message.emit("warn", "Cannot send START yet -- Pi control channel not connected.")

    def request_remote_stop(self):
        if self.control is not None:
            self.control.send_event("STOP", "", 0, 0)
            self.log_message.emit("warn", "Sent STOP to Pi -- robot will halt at its next safe checkpoint "
                                           "(not mid-pick/mid-place -- see robot_panel's honesty note).")
        else:
            self.log_message.emit("warn", "Cannot send STOP yet -- Pi control channel not connected.")

    # ---- NEW: was referenced by control_panel.py's Prioritize buttons but never
    # actually implemented -- the Pi's get_priority_override() (main() loop) has
    # always been ready to consume a "PRIORITY,cls_name,0,0" message and move that
    # class to the front of remaining_targets, this method was just missing. Same
    # send_event()/control-channel reuse as request_remote_start/stop above -- no
    # new wire protocol. ----
    def request_priority(self, cls_name: str):
        if self.control is not None:
            self.control.send_event("PRIORITY", cls_name, 0, 0)
            self.log_message.emit("info", f"Sent PRIORITY({cls_name}) to Pi -- it will search for "
                                           f"{cls_name} next.")
        else:
            self.log_message.emit("warn", f"Cannot send PRIORITY({cls_name}) yet -- Pi control "
                                           f"channel not connected.")

    def run(self):
        self.camera = PiCameraReceiver(PI_IP, CAMERA_STREAM_PORT)
        self.debug_camera = PiCameraReceiver(PI_IP, DEBUG_STREAM_PORT)  # NEW: connects to
        # the Pi's debug_stream_server() so the "Robot Camera (Raspberry Pi)" panel can
        # show the actual annotated display_frame (line-follow box, AprilTag, step count)
        # instead of the raw undistorted frame from CAMERA_STREAM_PORT.
        self.control = PiControlSender(PI_IP, CONTROL_PORT)
        self.log_message.emit("info", "Loading YOLO model...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded.emit(True)
            self.log_message.emit("info", f"YOLO model loaded. Classes: {self.model.names}")
        except Exception as e:
            self.model_loaded.emit(False)
            self.log_message.emit("error", f"Failed to load YOLO model: {e}")
            return

        stable_count = 0
        last_cls = None
        reported = False
        seen_reported_cls = None
        miss_count = 0

        last_conn_state = (None, None)

        while self._running:
            cam_ok = bool(self.camera and self.camera.connected)
            ctl_ok = bool(self.control and self.control.connected)
            if (cam_ok, ctl_ok) != last_conn_state:
                self.connection_changed.emit(cam_ok, ctl_ok)
                last_conn_state = (cam_ok, ctl_ok)

            # NEW: pull/emit the Pi's own annotated debug frame regardless of the
            # pause state below -- "paused" only pauses this PC's YOLO detection
            # loop, it does not stop the Pi's own mission, so its debug view
            # (line-follow box, AprilTag, step count) should keep updating.
            if self.debug_camera is not None:
                debug_frame = self.debug_camera.get_frame()
                if debug_frame is not None:
                    self.debug_frame_ready.emit(debug_frame)

            if self._paused:
                time.sleep(0.05)
                continue

            if self._reset_flag:
                stable_count = 0
                last_cls = None
                reported = False
                seen_reported_cls = None
                miss_count = 0
                self._reset_flag = False
                self.status_changed.emit("IDLE", "Detection state reset")
                self.log_message.emit("info", "Detection state reset by operator.")

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            self._last_raw = frame
            self.raw_frame_ready.emit(frame)

            conf = self.get_confidence()
            results = self.model.predict(frame, conf=conf, verbose=False)
            annotated = frame.copy()

            best_box = None
            best_conf = 0
            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = self.model.names[cls_id]
                bconf = float(box.conf)
                if cls_name in CLASSES_OF_INTEREST and bconf > best_conf:
                    best_conf = bconf
                    best_box = (cls_name, box.xyxy[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{cls_name} {bconf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self._last_annotated = annotated
            self.annotated_frame_ready.emit(annotated)

            if best_box is not None:
                cls_name, (x1, y1, x2, y2) = best_box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if cls_name == last_cls:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_cls = cls_name
                    reported = False
                    seen_reported_cls = None

                miss_count = 0

                if seen_reported_cls != cls_name:
                    self.control.send_event("SEEN", cls_name, cx, cy)
                    seen_reported_cls = cls_name
                    self.status_changed.emit("DETECTING", f"{cls_name} ({best_conf:.2f}) @ ({cx},{cy})")
                    self.log_message.emit("info", f"{cls_name} detected ({best_conf:.2f})")

                if stable_count >= STABILITY_FRAMES_REQUIRED and not reported:
                    self.control.send_event("STABLE", cls_name, cx, cy)
                    reported = True
                    self.status_changed.emit("STABLE", f"{cls_name} @ ({cx},{cy})")
                    self.log_message.emit("info", f"{cls_name} confirmed stable -> sent to Pi for picking.")

            else:
                miss_count += 1
                if miss_count > MISS_TOLERANCE_FRAMES:
                    if seen_reported_cls is not None:
                        self.control.send_event("LOST", seen_reported_cls, 0, 0)
                        self.status_changed.emit("SEARCHING", f"{seen_reported_cls} lost")
                        self.log_message.emit("warn", f"{seen_reported_cls} lost.")
                    stable_count = 0
                    last_cls = None
                    reported = False
                    seen_reported_cls = None
                    miss_count = 0
                else:
                    self.status_changed.emit("SEARCHING", f"miss {miss_count}/{MISS_TOLERANCE_FRAMES}")

        if self.camera:
            self.camera.stop()
        if self.debug_camera:
            self.debug_camera.stop()
