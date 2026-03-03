"""
add_camera_dialog.py â€” Add a direct IP camera via ONVIF discovery or manual RTSP URL.

Tabs:
  1. ONVIF Scan: broadcasts WS-Discovery, lists discovered devices, resolves RTSP URI
  2. Manual RTSP: enter URL directly with optional credentials
"""

import logging

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QWidget, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QMessageBox, QHeaderView,
)

from storage import repositories as repo
from utils.net_discovery import discover_onvif_devices, resolve_rtsp_uri
from utils.rtsp_test import test_rtsp
from utils.rtsp_templates import build_camera_url

log = logging.getLogger(__name__)


class _DiscoveryThread(QThread):
    result_ready = Signal(list, str)    # (devices, error_message)

    def run(self):
        devices, err = discover_onvif_devices(timeout=5.0)
        self.result_ready.emit(devices, err)


class _TestThread(QThread):
    result_ready = Signal(dict)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self._url = url

    def run(self):
        self.result_ready.emit(test_rtsp(self._url, timeout_sec=8))


class _ResolveThread(QThread):
    result_ready = Signal(str, str)   # (rtsp_url_or_empty, error)

    def __init__(self, xaddr, username, password, parent=None):
        super().__init__(parent)
        self._xaddr = xaddr
        self._user = username
        self._pass = password

    def run(self):
        url, err = resolve_rtsp_uri(self._xaddr, self._user, self._pass)
        self.result_ready.emit(url or "", err)


class AddCameraDialog(QDialog):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.camera_id: str | None = None
        self.setWindowTitle("Add Camera")
        self.setMinimumSize(580, 480)
        self._discovered: list[dict] = []
        self._resolve_thread: _ResolveThread | None = None
        self._test_thread: _TestThread | None = None
        self._disc_thread: _DiscoveryThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_onvif_tab(), "ONVIF Discovery")
        self._tabs.addTab(self._build_manual_tab(), "Manual RTSP")
        layout.addWidget(self._tabs)

        # Common name field
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Camera Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Front Door")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        self._save_btn = QPushButton("Save Camera")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _build_onvif_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Scan controls
        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan Network (ONVIF)")
        self._scan_btn.clicked.connect(self._start_scan)
        scan_row.addWidget(self._scan_btn)
        self._scan_status = QLabel("Click Scan to discover devices.")
        self._scan_status.setStyleSheet("color: #555555; font-size: 10px;")
        scan_row.addWidget(self._scan_status)
        scan_row.addStretch()
        layout.addLayout(scan_row)

        # Results table
        self._disc_table = QTableWidget(0, 3)
        self._disc_table.setHorizontalHeaderLabels(["IP", "Name / EPR", "Type"])
        self._disc_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._disc_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._disc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._disc_table.itemSelectionChanged.connect(self._on_device_selected)
        layout.addWidget(self._disc_table)

        # ONVIF credentials
        cred_grp = QGroupBox("Credentials for Selected Device")
        cred_form = QFormLayout(cred_grp)
        self._onvif_user = QLineEdit()
        self._onvif_user.setPlaceholderText("admin")
        cred_form.addRow("Username:", self._onvif_user)
        self._onvif_pass = QLineEdit()
        self._onvif_pass.setEchoMode(QLineEdit.EchoMode.Password)
        cred_form.addRow("Password:", self._onvif_pass)
        layout.addWidget(cred_grp)

        resolve_row = QHBoxLayout()
        self._resolve_btn = QPushButton("Resolve RTSP Stream")
        self._resolve_btn.setEnabled(False)
        self._resolve_btn.clicked.connect(self._resolve_onvif)
        resolve_row.addWidget(self._resolve_btn)
        self._resolve_status = QLabel("")
        resolve_row.addWidget(self._resolve_status)
        resolve_row.addStretch()
        layout.addLayout(resolve_row)

        return tab

    def _build_manual_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        self._rtsp_url_edit = QLineEdit()
        self._rtsp_url_edit.setPlaceholderText("rtsp://192.168.1.100:554/stream1")
        form.addRow("RTSP URL:", self._rtsp_url_edit)

        self._manual_user = QLineEdit()
        self._manual_user.setPlaceholderText("(optional if already in URL)")
        form.addRow("Username:", self._manual_user)

        self._manual_pass = QLineEdit()
        self._manual_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self._manual_pass.setPlaceholderText("(optional if already in URL)")
        form.addRow("Password:", self._manual_pass)

        layout.addLayout(form)

        test_row = QHBoxLayout()
        self._manual_test_btn = QPushButton("Test Connection")
        self._manual_test_btn.clicked.connect(self._test_manual)
        test_row.addWidget(self._manual_test_btn)
        self._manual_test_result = QLabel("")
        self._manual_test_result.setWordWrap(True)
        test_row.addWidget(self._manual_test_result)
        test_row.addStretch()
        layout.addLayout(test_row)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    def done(self, result: int) -> None:
        """Stop any running background threads before the dialog closes."""
        for thread in (self._disc_thread, self._resolve_thread, self._test_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(600):   # 600 ms grace period
                    thread.terminate()
                    thread.wait()
        super().done(result)

    def _start_scan(self) -> None:
        # Guard: don't start a second scan while one is already running
        if self._disc_thread is not None and self._disc_thread.isRunning():
            return

        self._scan_btn.setEnabled(False)
        self._scan_status.setText("Scanningâ€¦ (5 second timeout)")
        self._disc_table.setRowCount(0)

        self._disc_thread = _DiscoveryThread()
        self._disc_thread.result_ready.connect(self._on_scan_done)
        self._disc_thread.start()

    def _on_scan_done(self, devices: list, error: str) -> None:
        self._scan_btn.setEnabled(True)
        self._discovered = devices

        if error and not devices:
            self._scan_status.setText(f"Error: {error}")
            return

        self._scan_status.setText(f"Found {len(devices)} device(s)")
        self._disc_table.setRowCount(len(devices))
        for row, dev in enumerate(devices):
            self._disc_table.setItem(row, 0, QTableWidgetItem(dev.get("ip", "")))
            self._disc_table.setItem(row, 1, QTableWidgetItem(dev.get("name", "")))
            self._disc_table.setItem(row, 2, QTableWidgetItem(dev.get("types", "")))

    def _on_device_selected(self) -> None:
        rows = self._disc_table.selectedItems()
        self._resolve_btn.setEnabled(bool(rows))

    def _resolve_onvif(self) -> None:
        row = self._disc_table.currentRow()
        if row < 0 or row >= len(self._discovered):
            return
        dev = self._discovered[row]
        self._resolve_btn.setEnabled(False)
        self._resolve_status.setText("Resolvingâ€¦")
        self._resolve_thread = _ResolveThread(
            dev["xaddr"],
            self._onvif_user.text(),
            self._onvif_pass.text(),
        )
        self._resolve_thread.result_ready.connect(self._on_resolved)
        self._resolve_thread.start()

    def _on_resolved(self, rtsp_url: str, error: str) -> None:
        self._resolve_btn.setEnabled(True)
        if rtsp_url:
            self._rtsp_url_edit.setText(rtsp_url)
            self._manual_user.setText(self._onvif_user.text())
            self._manual_pass.setText(self._onvif_pass.text())
            self._resolve_status.setText("Resolved â€” check Manual RTSP tab")
            self._resolve_status.setStyleSheet("color: #006400;")
            self._tabs.setCurrentIndex(1)  # switch to Manual tab
        else:
            self._resolve_status.setText(f"Failed: {error}")
            self._resolve_status.setStyleSheet("color: #8b0000;")

    def _test_manual(self) -> None:
        raw_url = self._rtsp_url_edit.text().strip()
        if not raw_url:
            self._manual_test_result.setText("Enter a URL first.")
            return

        url = build_camera_url(
            raw_url,
            username=self._manual_user.text().strip(),
            password=self._manual_pass.text(),
        )

        self._manual_test_btn.setEnabled(False)
        self._manual_test_result.setText("Testing...")
        self._test_thread = _TestThread(url)
        self._test_thread.result_ready.connect(self._on_test_done)
        self._test_thread.start()

    def _on_test_done(self, result: dict) -> None:
        self._manual_test_btn.setEnabled(True)
        if result["ok"]:
            self._manual_test_result.setText(
                f"Connected  {result['width']}x{result['height']} @ {result['fps']:.1f} fps"
            )
            self._manual_test_result.setStyleSheet("color: #006400;")
        else:
            self._manual_test_result.setText(f"Failed: {result['error']}")
            self._manual_test_result.setStyleSheet("color: #8b0000;")

    # ------------------------------------------------------------------
    def _save(self) -> None:
        name = self._name_edit.text().strip()
        rtsp_url = self._rtsp_url_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Missing Field", "Enter a camera name.")
            return
        if not rtsp_url:
            QMessageBox.warning(self, "Missing Field", "Enter or resolve an RTSP URL.")
            return

        try:
            cid = repo.add_camera(
                name=name,
                rtsp_url=rtsp_url,
                username=self._manual_user.text().strip(),
                password=self._manual_pass.text(),
                source_type="rtsp",
            )
            self.camera_id = cid
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        log.info("Camera saved: %s (%s)", name, cid)
        self.accept()

