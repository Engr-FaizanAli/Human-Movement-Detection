"""
diagnostics_dialog.py — System diagnostics and connectivity testing.

Shows:
  - Local network info
  - Active sources status + last error
  - RTSP URL tester
  - Log export button
"""

import logging

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QTabWidget, QWidget, QMessageBox,
)

from utils.net_discovery import get_network_info
from utils.rtsp_test import test_rtsp

log = logging.getLogger(__name__)


class _RtspTestThread(QThread):
    result_ready = Signal(dict)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self._url = url

    def run(self):
        self.result_ready.emit(test_rtsp(self._url, timeout_sec=8))


class DiagnosticsDialog(QDialog):

    def __init__(self, source_manager=None, error_log: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        self.setMinimumSize(640, 500)
        self._source_manager = source_manager
        self._error_log = error_log or {}
        self._test_thread: _RtspTestThread | None = None
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_network_tab(), "Network")
        tabs.addTab(self._build_sources_tab(), "Active Sources")
        tabs.addTab(self._build_rtsp_test_tab(), "RTSP Tester")
        layout.addWidget(tabs)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)

    def _build_network_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self._hostname_lbl = QLabel("—")
        form.addRow("Hostname:", self._hostname_lbl)
        self._ip_lbl = QLabel("—")
        form.addRow("Local IP:", self._ip_lbl)
        layout.addLayout(form)
        layout.addStretch()
        return tab

    def _build_sources_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self._sources_table = QTableWidget(0, 3)
        self._sources_table.setHorizontalHeaderLabels(["Source", "Status", "Last Error"])
        self._sources_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._sources_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._sources_table)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._populate_sources)
        layout.addWidget(refresh_btn)
        return tab

    def _build_rtsp_test_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        self._rtsp_test_edit = QLineEdit()
        self._rtsp_test_edit.setPlaceholderText("rtsp://192.168.1.100:554/stream")
        form.addRow("RTSP URL:", self._rtsp_test_edit)
        layout.addLayout(form)

        test_row = QHBoxLayout()
        self._rtsp_test_btn = QPushButton("Test Connection")
        self._rtsp_test_btn.clicked.connect(self._run_rtsp_test)
        test_row.addWidget(self._rtsp_test_btn)
        test_row.addStretch()
        layout.addLayout(test_row)

        self._rtsp_result = QTextEdit()
        self._rtsp_result.setReadOnly(True)
        self._rtsp_result.setMaximumHeight(120)
        self._rtsp_result.setPlaceholderText("Test result will appear here…")
        layout.addWidget(self._rtsp_result)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------

    def _populate(self) -> None:
        info = get_network_info()
        self._hostname_lbl.setText(info.get("hostname", "—"))
        self._ip_lbl.setText(info.get("local_ip", "—"))
        self._populate_sources()

    def _populate_sources(self) -> None:
        if self._source_manager is None:
            return
        sources = self._source_manager.get_all_sources()
        active_ids = self._source_manager.get_active_ids()

        self._sources_table.setRowCount(len(sources))
        for row, src in enumerate(sources):
            sid = src["source_id"]
            name = src.get("source_name", sid)
            status = "Active" if sid in active_ids else "Inactive"
            error = self._error_log.get(sid, "")

            self._sources_table.setItem(row, 0, QTableWidgetItem(name))
            status_item = QTableWidgetItem(status)
            if status == "Active":
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            self._sources_table.setItem(row, 1, status_item)
            self._sources_table.setItem(row, 2, QTableWidgetItem(error))

    def _run_rtsp_test(self) -> None:
        url = self._rtsp_test_edit.text().strip()
        if not url:
            self._rtsp_result.setText("Enter a URL to test.")
            return
        self._rtsp_test_btn.setEnabled(False)
        self._rtsp_result.setText("Testing…")

        self._test_thread = _RtspTestThread(url)
        self._test_thread.result_ready.connect(self._on_rtsp_test_done)
        self._test_thread.start()

    def _on_rtsp_test_done(self, result: dict) -> None:
        self._rtsp_test_btn.setEnabled(True)
        if result["ok"]:
            self._rtsp_result.setText(
                f"SUCCESS\n"
                f"Resolution : {result['width']}x{result['height']}\n"
                f"FPS        : {result['fps']:.2f}\n"
                f"Latency    : {result['latency_ms']:.0f} ms"
            )
            self._rtsp_result.setStyleSheet("color: #006400;")
        else:
            self._rtsp_result.setText(f"FAILED\n{result['error']}")
            self._rtsp_result.setStyleSheet("color: #8b0000;")

    # ------------------------------------------------------------------
    # Update error log from main window
    # ------------------------------------------------------------------

    def log_error(self, source_id: str, message: str) -> None:
        self._error_log[source_id] = message
