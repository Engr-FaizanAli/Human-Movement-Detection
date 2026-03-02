"""
add_recorder_dialog.py — Dialog to add a Dahua NVR, Hikvision DVR/NVR, or Generic recorder.

On acceptance it:
  1. Saves the recorder record to DB
  2. Provisions channel entries (CH1..CHN)
  3. Returns the recorder ID via .recorder_id
"""

import logging
import threading

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QPushButton,
    QGroupBox, QCheckBox, QMessageBox, QProgressBar, QTextEdit,
)

from storage import repositories as repo
from utils.rtsp_templates import TEMPLATES, BRAND_DEFAULT_TEMPLATE, get_template_choices
from utils.rtsp_test import test_rtsp
from utils.rtsp_templates import build_channel_url

log = logging.getLogger(__name__)

_BRANDS = [("dahua", "Dahua NVR"), ("hikvision", "Hikvision DVR/NVR"), ("generic", "Generic RTSP Recorder")]


class _TestThread(QThread):
    result_ready = Signal(dict)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self._url = url

    def run(self):
        result = test_rtsp(self._url, timeout_sec=8)
        self.result_ready.emit(result)


class AddRecorderDialog(QDialog):

    def __init__(self, parent=None, recorder_id: str | None = None) -> None:
        super().__init__(parent)
        self.recorder_id: str | None = None
        self._edit_id = recorder_id
        self._test_thread: _TestThread | None = None

        self.setWindowTitle("Add Recorder" if not recorder_id else "Edit Recorder")
        self.setMinimumWidth(520)
        self._build_ui()

        if recorder_id:
            self._load_existing(recorder_id)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Basic info
        basic = QGroupBox("Recorder Details")
        form = QFormLayout(basic)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._brand_combo = QComboBox()
        for key, label in _BRANDS:
            self._brand_combo.addItem(label, key)
        self._brand_combo.currentIndexChanged.connect(self._on_brand_changed)
        form.addRow("Brand:", self._brand_combo)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Office NVR")
        form.addRow("Friendly Name:", self._name_edit)

        self._ip_edit = QLineEdit()
        self._ip_edit.setPlaceholderText("192.168.1.100")
        form.addRow("IP / Hostname:", self._ip_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(554)
        form.addRow("RTSP Port:", self._port_spin)

        self._user_edit = QLineEdit()
        self._user_edit.setPlaceholderText("admin")
        form.addRow("Username:", self._user_edit)

        self._pass_edit = QLineEdit()
        self._pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password:", self._pass_edit)

        self._channels_spin = QSpinBox()
        self._channels_spin.setRange(1, 64)
        self._channels_spin.setValue(16)
        form.addRow("Channel Count:", self._channels_spin)

        self._stream_combo = QComboBox()
        self._stream_combo.addItem("Substream (recommended)", "sub")
        self._stream_combo.addItem("Main Stream", "main")
        form.addRow("Stream Preference:", self._stream_combo)

        layout.addWidget(basic)

        # RTSP Template
        tmpl_grp = QGroupBox("RTSP URL Template")
        tmpl_form = QFormLayout(tmpl_grp)

        self._tmpl_combo = QComboBox()
        for key, label in get_template_choices():
            self._tmpl_combo.addItem(label, key)
        self._tmpl_combo.currentIndexChanged.connect(self._on_template_changed)
        tmpl_form.addRow("Template:", self._tmpl_combo)

        self._custom_tmpl_edit = QLineEdit()
        self._custom_tmpl_edit.setPlaceholderText(
            "rtsp://{user}:{password}@{ip}:{port}/your/stream/path/{ch}"
        )
        self._custom_tmpl_edit.setEnabled(False)
        tmpl_form.addRow("Custom Template:", self._custom_tmpl_edit)

        layout.addWidget(tmpl_grp)

        # Test channel
        test_grp = QGroupBox("Test Channel")
        test_layout = QVBoxLayout(test_grp)

        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel:"))
        self._test_ch_spin = QSpinBox()
        self._test_ch_spin.setRange(1, 64)
        self._test_ch_spin.setValue(1)
        ch_row.addWidget(self._test_ch_spin)

        self._test_btn = QPushButton("Test Channel")
        self._test_btn.clicked.connect(self._test_channel)
        ch_row.addWidget(self._test_btn)
        ch_row.addStretch()
        test_layout.addLayout(ch_row)

        self._test_result = QLabel("")
        self._test_result.setWordWrap(True)
        test_layout.addWidget(self._test_result)

        layout.addWidget(test_grp)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._save_btn = QPushButton("Save Recorder")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)

        layout.addLayout(btn_row)

        # Set initial template for default brand
        self._on_brand_changed(0)

    # ------------------------------------------------------------------
    def _on_brand_changed(self, _idx: int) -> None:
        brand = self._brand_combo.currentData()
        default_key = BRAND_DEFAULT_TEMPLATE.get(brand, "generic")
        for i in range(self._tmpl_combo.count()):
            if self._tmpl_combo.itemData(i) == default_key:
                self._tmpl_combo.setCurrentIndex(i)
                break

    def _on_template_changed(self, _idx: int) -> None:
        key = self._tmpl_combo.currentData()
        self._custom_tmpl_edit.setEnabled(key == "custom")

    # ------------------------------------------------------------------
    def _test_channel(self) -> None:
        url = self._build_test_url()
        if not url:
            return
        self._test_btn.setEnabled(False)
        self._test_result.setText("Testing…")

        self._test_thread = _TestThread(url)
        self._test_thread.result_ready.connect(self._on_test_result)
        self._test_thread.start()

    def _build_test_url(self) -> str | None:
        ip = self._ip_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "Missing Field", "Enter an IP address first.")
            return None
        try:
            return build_channel_url(
                brand=self._brand_combo.currentData(),
                ip=ip,
                port=self._port_spin.value(),
                username=self._user_edit.text(),
                password=self._pass_edit.text(),
                channel_num=self._test_ch_spin.value(),
                stream_pref=self._stream_combo.currentData(),
                template_key=self._tmpl_combo.currentData(),
                custom_template=self._custom_tmpl_edit.text().strip() or None,
            )
        except Exception as exc:
            QMessageBox.warning(self, "URL Error", str(exc))
            return None

    def _on_test_result(self, result: dict) -> None:
        self._test_btn.setEnabled(True)
        if result["ok"]:
            self._test_result.setText(
                f"Connected  {result['width']}x{result['height']} @ {result['fps']:.1f} fps  "
                f"({result['latency_ms']:.0f} ms)"
            )
            self._test_result.setStyleSheet("color: #006400;")
        else:
            self._test_result.setText(f"Failed: {result['error']}")
            self._test_result.setStyleSheet("color: #8b0000;")

    # ------------------------------------------------------------------
    def _save(self) -> None:
        name = self._name_edit.text().strip()
        ip = self._ip_edit.text().strip()
        if not name or not ip:
            QMessageBox.warning(self, "Missing Fields", "Friendly Name and IP are required.")
            return

        tmpl_key = self._tmpl_combo.currentData()
        custom = self._custom_tmpl_edit.text().strip() if tmpl_key == "custom" else None

        try:
            if self._edit_id:
                repo.update_recorder(
                    self._edit_id,
                    brand=self._brand_combo.currentData(),
                    name=name,
                    ip=ip,
                    rtsp_port=self._port_spin.value(),
                    username=self._user_edit.text(),
                    password=self._pass_edit.text(),
                    channel_count=self._channels_spin.value(),
                    stream_pref=self._stream_combo.currentData(),
                    template_key=tmpl_key,
                    custom_template=custom,
                )
                self.recorder_id = self._edit_id
                repo.provision_channels(self._edit_id, self._channels_spin.value())
            else:
                rid = repo.add_recorder(
                    brand=self._brand_combo.currentData(),
                    name=name,
                    ip=ip,
                    rtsp_port=self._port_spin.value(),
                    username=self._user_edit.text(),
                    password=self._pass_edit.text(),
                    channel_count=self._channels_spin.value(),
                    stream_pref=self._stream_combo.currentData(),
                    template_key=tmpl_key,
                    custom_template=custom,
                )
                self.recorder_id = rid
                repo.provision_channels(rid, self._channels_spin.value())
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        log.info("Recorder saved: %s (%s)", name, self.recorder_id)
        self.accept()

    # ------------------------------------------------------------------
    def _load_existing(self, recorder_id: str) -> None:
        rec = repo.get_recorder(recorder_id)
        if not rec:
            return
        for i in range(self._brand_combo.count()):
            if self._brand_combo.itemData(i) == rec["brand"]:
                self._brand_combo.setCurrentIndex(i)
                break
        self._name_edit.setText(rec["name"])
        self._ip_edit.setText(rec["ip"])
        self._port_spin.setValue(rec["rtsp_port"])
        self._user_edit.setText(rec["username"] or "")
        # Do NOT pre-fill password for security
        self._channels_spin.setValue(rec["channel_count"])
        for i in range(self._stream_combo.count()):
            if self._stream_combo.itemData(i) == rec["stream_pref"]:
                self._stream_combo.setCurrentIndex(i)
                break
        if rec["template_key"]:
            for i in range(self._tmpl_combo.count()):
                if self._tmpl_combo.itemData(i) == rec["template_key"]:
                    self._tmpl_combo.setCurrentIndex(i)
                    break
        if rec["custom_template"]:
            self._custom_tmpl_edit.setText(rec["custom_template"])
