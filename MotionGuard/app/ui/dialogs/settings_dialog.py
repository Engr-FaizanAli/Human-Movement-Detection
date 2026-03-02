"""
settings_dialog.py — Global and per-source detection settings.

All parameters are presented with plain-English labels and one-line descriptions.
Per-source overrides are stored in detection_params table.
Global settings are stored in global_settings table.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QWidget, QLabel, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QPushButton,
    QGroupBox, QFileDialog, QMessageBox,
)

from storage import repositories as repo
from utils.logging_setup import update_log_level

log = logging.getLogger(__name__)


def _slider(minimum: float, maximum: float, step: float = 1.0) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setMinimum(int(minimum / step))
    s.setMaximum(int(maximum / step))
    return s


def _desc_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #666666; font-size: 10px; font-style: italic;")
    lbl.setWordWrap(True)
    return lbl


class SettingsDialog(QDialog):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(620, 540)
        self._settings = repo.get_all_settings()
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_detection_tab(), "Detection")
        self._tabs.addTab(self._build_preprocess_tab(), "Pre-Processing")
        self._tabs.addTab(self._build_noise_tab(), "Noise Filtering")
        self._tabs.addTab(self._build_tracking_tab(), "Object Tracking")
        self._tabs.addTab(self._build_alerts_tab(), "Alerts & Snapshots")
        self._tabs.addTab(self._build_performance_tab(), "Performance")
        self._tabs.addTab(self._build_app_tab(), "Application")
        layout.addWidget(self._tabs)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._ok)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_detection_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        # Detection Mode
        self._method_combo = QComboBox()
        self._method_combo.addItem("MOG2 (Adaptive Background)", "mog2")
        self._method_combo.addItem("Frame Difference", "diff")
        self._method_combo.currentIndexChanged.connect(self._update_detection_visibility)
        form.addRow("Detection Mode:", self._method_combo)
        form.addRow("", _desc_label("MOG2 builds a background model; Frame Difference compares consecutive frames."))

        # Motion Sensitivity
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(1.0, 80.0)
        self._threshold_spin.setSingleStep(0.5)
        form.addRow("Motion Sensitivity:", self._threshold_spin)
        form.addRow("", _desc_label("Lower value detects more subtle movement; higher reduces false alarms."))

        # Background Memory (MOG2)
        self._history_spin = QSpinBox()
        self._history_spin.setRange(10, 500)
        form.addRow("Background Memory (MOG2):", self._history_spin)
        self._history_desc = _desc_label("Number of past frames used to build the background model.")
        form.addRow("", self._history_desc)

        # Frame Gap (Diff)
        self._diff_frames_spin = QSpinBox()
        self._diff_frames_spin.setRange(1, 20)
        form.addRow("Frame Comparison Gap (Diff):", self._diff_frames_spin)
        self._diff_desc = _desc_label("How many frames apart to compare for motion detection.")
        form.addRow("", self._diff_desc)

        # Show Unconfirmed Tracks
        self._show_candidates_cb = QCheckBox("Show unconfirmed tracks in preview")
        form.addRow("Candidate Tracks:", self._show_candidates_cb)
        form.addRow("", _desc_label("Display objects still being evaluated before they are confirmed as real motion."))

        # PTZ Motion Compensation
        self._ptz_aware_cb = QCheckBox("Enable PTZ motion compensation")
        form.addRow("PTZ Stabilizer:", self._ptz_aware_cb)
        form.addRow("", _desc_label("Pauses background learning during pan/tilt/zoom to prevent false detections."))

        self._ptz_thresh_spin = QDoubleSpinBox()
        self._ptz_thresh_spin.setRange(0.5, 50.0)
        self._ptz_thresh_spin.setSingleStep(0.5)
        form.addRow("PTZ Motion Threshold:", self._ptz_thresh_spin)
        form.addRow("", _desc_label("Frame-to-frame pixel shift that triggers PTZ compensation mode."))

        self._ptz_settle_spin = QSpinBox()
        self._ptz_settle_spin.setRange(10, 500)
        form.addRow("PTZ Settle Frames:", self._ptz_settle_spin)
        form.addRow("", _desc_label("Frames to wait after camera movement stops before resuming detection."))

        self._ptz_aware_cb.toggled.connect(self._update_ptz_visibility)

        return tab

    def _build_preprocess_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        self._clahe_cb = QCheckBox("Enabled")
        form.addRow("Enhance Low-Light Contrast:", self._clahe_cb)
        form.addRow("", _desc_label("Improves detection in dim or unevenly lit scenes."))

        self._clahe_clip_spin = QDoubleSpinBox()
        self._clahe_clip_spin.setRange(0.5, 8.0)
        self._clahe_clip_spin.setSingleStep(0.1)
        form.addRow("Contrast Boost Strength:", self._clahe_clip_spin)
        form.addRow("", _desc_label("How aggressively local contrast is enhanced. Higher = stronger."))

        self._clahe_tile_spin = QSpinBox()
        self._clahe_tile_spin.setRange(4, 64)
        form.addRow("Contrast Region Size:", self._clahe_tile_spin)
        form.addRow("", _desc_label("Size in pixels of the local area used for contrast enhancement."))

        self._temporal_cb = QCheckBox("Enabled")
        form.addRow("Temporal Frame Smoothing:", self._temporal_cb)
        form.addRow("", _desc_label("Reduces noise caused by flickering or camera grain between frames."))

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.05, 0.95)
        self._alpha_spin.setSingleStep(0.05)
        form.addRow("Background Update Speed:", self._alpha_spin)
        form.addRow("", _desc_label("How quickly the background model adapts to slow scene changes."))

        return tab

    def _build_noise_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        self._morph_close_spin = QSpinBox()
        self._morph_close_spin.setRange(0, 15)
        form.addRow("Fill Detection Gaps:", self._morph_close_spin)
        form.addRow("", _desc_label("Closes small holes inside detected moving regions."))

        self._morph_open_spin = QSpinBox()
        self._morph_open_spin.setRange(0, 15)
        form.addRow("Remove Motion Noise:", self._morph_open_spin)
        form.addRow("", _desc_label("Removes tiny isolated false-positive specks from detection."))

        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(1, 5000)
        form.addRow("Minimum Object Size (px²):", self._min_area_spin)
        form.addRow("", _desc_label("Ignore moving objects smaller than this pixel area."))

        self._max_area_spin = QSpinBox()
        self._max_area_spin.setRange(100, 50000)
        form.addRow("Maximum Object Size (px²):", self._max_area_spin)
        form.addRow("", _desc_label("Ignore detections larger than this (avoids background noise)."))

        self._solidity_spin = QDoubleSpinBox()
        self._solidity_spin.setRange(0.0, 1.0)
        self._solidity_spin.setSingleStep(0.05)
        form.addRow("Shape Regularity:", self._solidity_spin)
        form.addRow("", _desc_label("Rejects irregular blobs. 0 = accept any shape, 1 = only compact shapes."))

        self._density_spin = QDoubleSpinBox()
        self._density_spin.setRange(0.0, 1.0)
        self._density_spin.setSingleStep(0.05)
        form.addRow("Fill Density Filter:", self._density_spin)
        form.addRow("", _desc_label("How solidly filled the detected region must be inside its bounding box."))

        return tab

    def _build_tracking_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        self._persistence_spin = QSpinBox()
        self._persistence_spin.setRange(1, 30)
        form.addRow("Confirmation Frames:", self._persistence_spin)
        form.addRow("", _desc_label("How many consecutive frames a detection must appear before being shown."))

        self._min_disp_spin = QDoubleSpinBox()
        self._min_disp_spin.setRange(0.0, 200.0)
        self._min_disp_spin.setSingleStep(1.0)
        form.addRow("Minimum Movement Distance (px):", self._min_disp_spin)
        form.addRow("", _desc_label("Object must move at least this many pixels to be confirmed."))

        self._spatial_tol_spin = QDoubleSpinBox()
        self._spatial_tol_spin.setRange(1.0, 200.0)
        self._spatial_tol_spin.setSingleStep(1.0)
        form.addRow("Tracking Jump Tolerance (px):", self._spatial_tol_spin)
        form.addRow("", _desc_label("Maximum pixel distance a tracked object can move between frames."))

        self._max_absent_spin = QSpinBox()
        self._max_absent_spin.setRange(1, 30)
        form.addRow("Track Hold Duration (frames):", self._max_absent_spin)
        form.addRow("", _desc_label("How many frames to keep tracking a lost object before dropping it."))

        self._bbox_alpha_spin = QDoubleSpinBox()
        self._bbox_alpha_spin.setRange(0.05, 0.95)
        self._bbox_alpha_spin.setSingleStep(0.05)
        form.addRow("Box Movement Smoothness:", self._bbox_alpha_spin)
        form.addRow("", _desc_label("Smooths jitter in the displayed detection bounding box."))

        self._vel_alpha_spin = QDoubleSpinBox()
        self._vel_alpha_spin.setRange(0.05, 0.95)
        self._vel_alpha_spin.setSingleStep(0.05)
        form.addRow("Velocity Estimation Smoothness:", self._vel_alpha_spin)
        form.addRow("", _desc_label("Smooths the predicted movement direction and speed."))

        return tab

    def _build_alerts_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        self._alarm_cb = QCheckBox("Play sound alarm on motion")
        form.addRow("Sound Alarm:", self._alarm_cb)

        self._volume_spin = QDoubleSpinBox()
        self._volume_spin.setRange(0.0, 1.0)
        self._volume_spin.setSingleStep(0.1)
        form.addRow("Alarm Volume:", self._volume_spin)

        alarm_btn_row = QHBoxLayout()
        test_alarm_btn = QPushButton("Test Alarm")
        test_alarm_btn.clicked.connect(self._test_alarm)
        alarm_btn_row.addWidget(test_alarm_btn)

        self._alarm_file_lbl = QLabel("Default alarm")
        alarm_btn_row.addWidget(self._alarm_file_lbl)

        browse_alarm_btn = QPushButton("Browse…")
        browse_alarm_btn.clicked.connect(self._browse_alarm)
        alarm_btn_row.addWidget(browse_alarm_btn)
        alarm_btn_row.addStretch()
        form.addRow("", alarm_btn_row)

        self._snapshot_cb = QCheckBox("Save snapshot on motion start")
        form.addRow("Snapshots:", self._snapshot_cb)

        self._retention_spin = QSpinBox()
        self._retention_spin.setRange(1, 365)
        self._retention_spin.setSuffix(" days")
        form.addRow("Snapshot Retention:", self._retention_spin)
        form.addRow("", _desc_label("Snapshots and events older than this are deleted on startup."))

        return tab

    def _build_performance_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        self._det_fps_spin = QSpinBox()
        self._det_fps_spin.setRange(1, 30)
        self._det_fps_spin.setSuffix(" fps")
        form.addRow("Detection FPS:", self._det_fps_spin)
        form.addRow("", _desc_label("How many frames per second are processed for motion detection."))

        self._prev_fps_spin = QSpinBox()
        self._prev_fps_spin.setRange(1, 30)
        self._prev_fps_spin.setSuffix(" fps")
        form.addRow("Preview FPS:", self._prev_fps_spin)
        form.addRow("", _desc_label("How often the video preview updates in the UI."))

        self._stream_combo = QComboBox()
        self._stream_combo.addItem("Substream (recommended)", "sub")
        self._stream_combo.addItem("Main Stream", "main")
        form.addRow("Default Stream:", self._stream_combo)
        form.addRow("", _desc_label("Substream uses less bandwidth and CPU; main stream is higher resolution."))

        return tab

    def _build_app_tab(self) -> QWidget:
        tab, form = self._tab_with_form()

        self._auto_start_cb = QCheckBox("Start detection automatically when app opens")
        form.addRow("Auto-Start:", self._auto_start_cb)

        self._start_min_cb = QCheckBox("Start minimized to system tray")
        form.addRow("Start Minimized:", self._start_min_cb)

        self._log_combo = QComboBox()
        for lvl in ["DEBUG", "INFO", "WARNING"]:
            self._log_combo.addItem(lvl)
        form.addRow("Log Level:", self._log_combo)

        export_btn = QPushButton("Export Logs…")
        export_btn.clicked.connect(self._export_logs)
        form.addRow("Log Export:", export_btn)
        form.addRow("", _desc_label("Exports logs to a ZIP file (credentials are stripped)."))

        return tab

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tab_with_form(self) -> tuple[QWidget, QFormLayout]:
        tab = QWidget()
        scroll_layout = QVBoxLayout(tab)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        scroll_layout.addLayout(form)
        scroll_layout.addStretch()
        return tab, form

    def _update_detection_visibility(self) -> None:
        is_mog2 = self._method_combo.currentData() == "mog2"
        self._history_spin.setEnabled(is_mog2)
        self._diff_frames_spin.setEnabled(not is_mog2)

    def _update_ptz_visibility(self) -> None:
        enabled = self._ptz_aware_cb.isChecked()
        self._ptz_thresh_spin.setEnabled(enabled)
        self._ptz_settle_spin.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load_values(self) -> None:
        s = self._settings
        for i in range(self._method_combo.count()):
            if self._method_combo.itemData(i) == s.get("detection_mode", "mog2"):
                self._method_combo.setCurrentIndex(i)
                break
        self._threshold_spin.setValue(float(s.get("threshold", 15.0)))
        self._history_spin.setValue(int(s.get("history", 200)))
        self._diff_frames_spin.setValue(int(s.get("diff_frames", 5)))
        self._show_candidates_cb.setChecked(s.get("show_candidates", "0") == "1")
        self._ptz_aware_cb.setChecked(s.get("ptz_aware", "1") == "1")
        self._ptz_thresh_spin.setValue(float(s.get("ptz_motion_thresh", 5.0)))
        self._ptz_settle_spin.setValue(int(s.get("ptz_settle_frames", 125)))

        self._clahe_cb.setChecked(s.get("use_clahe", "0") == "1")
        self._clahe_clip_spin.setValue(float(s.get("clahe_clip", 2.5)))
        self._clahe_tile_spin.setValue(int(s.get("clahe_tile", 12)))
        self._temporal_cb.setChecked(s.get("use_temporal", "1") == "1")
        self._alpha_spin.setValue(float(s.get("alpha", 0.4)))

        self._morph_close_spin.setValue(int(s.get("morph_close", 5)))
        self._morph_open_spin.setValue(int(s.get("morph_open", 3)))
        self._min_area_spin.setValue(int(s.get("min_area", 10)))
        self._max_area_spin.setValue(int(s.get("max_area", 5000)))
        self._solidity_spin.setValue(float(s.get("min_solidity", 0.3)))
        self._density_spin.setValue(float(s.get("min_density", 0.0)))

        self._persistence_spin.setValue(int(s.get("persistence", 4)))
        self._min_disp_spin.setValue(float(s.get("min_displacement", 15.0)))
        self._spatial_tol_spin.setValue(float(s.get("spatial_tol", 20.0)))
        self._max_absent_spin.setValue(int(s.get("max_absent", 5)))
        self._bbox_alpha_spin.setValue(float(s.get("bbox_smooth_alpha", 0.35)))
        self._vel_alpha_spin.setValue(float(s.get("vel_smooth_alpha", 0.5)))

        self._alarm_cb.setChecked(s.get("alarm_enabled", "1") == "1")
        self._volume_spin.setValue(float(s.get("alarm_volume", 0.8)))
        self._snapshot_cb.setChecked(s.get("snapshot_enabled", "1") == "1")
        self._retention_spin.setValue(int(s.get("snapshot_retention_days", 30)))

        self._det_fps_spin.setValue(int(s.get("detection_fps", 10)))
        self._prev_fps_spin.setValue(int(s.get("preview_fps", 15)))
        for i in range(self._stream_combo.count()):
            if self._stream_combo.itemData(i) == s.get("stream_pref", "sub"):
                self._stream_combo.setCurrentIndex(i)

        self._auto_start_cb.setChecked(s.get("auto_start", "0") == "1")
        self._start_min_cb.setChecked(s.get("start_minimized", "0") == "1")
        for i in range(self._log_combo.count()):
            if self._log_combo.itemText(i) == s.get("log_level", "INFO"):
                self._log_combo.setCurrentIndex(i)

        self._update_detection_visibility()
        self._update_ptz_visibility()

    def _apply(self) -> None:
        s = {
            "detection_mode": self._method_combo.currentData(),
            "threshold": str(self._threshold_spin.value()),
            "history": str(self._history_spin.value()),
            "diff_frames": str(self._diff_frames_spin.value()),
            "show_candidates": "1" if self._show_candidates_cb.isChecked() else "0",
            "ptz_aware": "1" if self._ptz_aware_cb.isChecked() else "0",
            "ptz_motion_thresh": str(self._ptz_thresh_spin.value()),
            "ptz_settle_frames": str(self._ptz_settle_spin.value()),
            "use_clahe": "1" if self._clahe_cb.isChecked() else "0",
            "clahe_clip": str(self._clahe_clip_spin.value()),
            "clahe_tile": str(self._clahe_tile_spin.value()),
            "use_temporal": "1" if self._temporal_cb.isChecked() else "0",
            "alpha": str(self._alpha_spin.value()),
            "morph_close": str(self._morph_close_spin.value()),
            "morph_open": str(self._morph_open_spin.value()),
            "min_area": str(self._min_area_spin.value()),
            "max_area": str(self._max_area_spin.value()),
            "min_solidity": str(self._solidity_spin.value()),
            "min_density": str(self._density_spin.value()),
            "persistence": str(self._persistence_spin.value()),
            "min_displacement": str(self._min_disp_spin.value()),
            "spatial_tol": str(self._spatial_tol_spin.value()),
            "max_absent": str(self._max_absent_spin.value()),
            "bbox_smooth_alpha": str(self._bbox_alpha_spin.value()),
            "vel_smooth_alpha": str(self._vel_alpha_spin.value()),
            "alarm_enabled": "1" if self._alarm_cb.isChecked() else "0",
            "alarm_volume": str(self._volume_spin.value()),
            "snapshot_enabled": "1" if self._snapshot_cb.isChecked() else "0",
            "snapshot_retention_days": str(self._retention_spin.value()),
            "detection_fps": str(self._det_fps_spin.value()),
            "preview_fps": str(self._prev_fps_spin.value()),
            "stream_pref": self._stream_combo.currentData(),
            "auto_start": "1" if self._auto_start_cb.isChecked() else "0",
            "start_minimized": "1" if self._start_min_cb.isChecked() else "0",
            "log_level": self._log_combo.currentText(),
        }
        for key, value in s.items():
            repo.set_setting(key, value)
        update_log_level(s["log_level"])
        log.info("Settings saved")

    def _ok(self) -> None:
        self._apply()
        self.accept()

    # ------------------------------------------------------------------

    def _test_alarm(self) -> None:
        # Access alert manager via parent's reference if available
        parent = self.parent()
        if hasattr(parent, "alert_manager"):
            parent.alert_manager.test_alarm()

    def _browse_alarm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Alarm Sound", "", "WAV Files (*.wav)"
        )
        if path:
            repo.set_setting("alarm_file", path)
            self._alarm_file_lbl.setText(path)

    def _export_logs(self) -> None:
        import zipfile
        import re
        from pathlib import Path
        from utils.paths import logs_dir, app_data_dir

        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "motionguard_logs.zip", "ZIP Files (*.zip)"
        )
        if not dest:
            return
        try:
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                for log_file in logs_dir().glob("*.log*"):
                    # Strip credentials from log content
                    content = log_file.read_text(errors="replace")
                    content = re.sub(
                        r"rtsp://[^:]+:[^@]+@",
                        "rtsp://***:***@",
                        content,
                    )
                    zf.writestr(log_file.name, content)
            QMessageBox.information(
                self, "Export Complete", f"Logs exported to:\n{dest}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
