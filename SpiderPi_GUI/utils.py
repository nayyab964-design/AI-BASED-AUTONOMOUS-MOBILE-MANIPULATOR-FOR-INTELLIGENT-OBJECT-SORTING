"""
utils.py

Small shared helpers used across the panels. Nothing here talks to the
Pi or YOLO -- purely presentation-layer utilities (frame -> Qt pixmap
conversion, timestamp formatting for the log panel / camera panels).
"""

import time
import cv2
from PySide6.QtGui import QImage, QPixmap


def cv_frame_to_pixmap(frame) -> QPixmap:
    """Convert a BGR OpenCV frame (numpy array) into a QPixmap that can be
    set directly on a QLabel. Used by camera_widget.CameraPanel.update_frame().
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    # .copy() so the QImage doesn't keep referencing a numpy buffer that
    # may be overwritten/garbage-collected right after this call returns.
    return QPixmap.fromImage(qimg.copy())


def timestamp() -> str:
    """HH:MM:SS timestamp string, used by the log panel and camera info labels."""
    return time.strftime("%H:%M:%S")
