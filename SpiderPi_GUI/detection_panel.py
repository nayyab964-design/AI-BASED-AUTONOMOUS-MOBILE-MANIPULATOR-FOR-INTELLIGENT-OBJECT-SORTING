"""detection_panel.py -- confidence slider + detection controls + model status."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton


class DetectionSettingsPanel(QGroupBox):
    confidence_changed = Signal(float)
    reload_model_requested = Signal()

    def __init__(self, initial_conf=0.2, parent=None):
        super().__init__("Detection Settings", parent)

        self.conf_label = QLabel(f"Confidence: {initial_conf:.2f}")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(initial_conf * 100))
        self.conf_slider.valueChanged.connect(self._on_slider_changed)

        self.start_detect_btn = QPushButton("Start Detection")
        self.stop_detect_btn = QPushButton("Stop Detection")
        self.reload_btn = QPushButton("Reload Model")

        self.model_status = QLabel("● Model Loaded: No")
        self.model_status.setStyleSheet("color:#f85149; font-weight:600;")

        self.reload_btn.clicked.connect(self.reload_model_requested.emit)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_detect_btn)
        btn_row.addWidget(self.stop_detect_btn)
        btn_row.addWidget(self.reload_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_slider)
        layout.addLayout(btn_row)
        layout.addWidget(self.model_status)
        self.setLayout(layout)

    def _on_slider_changed(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"Confidence: {conf:.2f}")
        self.confidence_changed.emit(conf)

    def set_model_loaded(self, loaded: bool):
        self.model_status.setText(f"● Model Loaded: {'Yes' if loaded else 'No'}")
        self.model_status.setStyleSheet(
            f"color:{'#3fb950' if loaded else '#f85149'}; font-weight:600;")
