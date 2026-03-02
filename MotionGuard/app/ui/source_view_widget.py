"""
source_view_widget.py — Single source cell widget.

Displays the video frame for one source with:
  - Source name label
  - Connection status badge
  - Motion indicator
  - Stats HUD (FPS, confirmed, candidates)
  - Overlay toggle button
  - Edit Zones button
"""

import logging

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QFrame,
)

log = logging.getLogger(__name__)

_STATUS_COLORS = {
    "connected":    "#1a7a1a",
    "reconnecting": "#b8860b",
    "offline":      "#8b0000",
    "finished":     "#444444",
}


class VideoLabel(QLabel):
    """QLabel that scales its pixmap to fill available space, maintaining aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None:
            return
        painter = QPainter(self)
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


class SourceViewWidget(QWidget):
    """
    Displays one video source with controls.

    Signals
    -------
    edit_zones_requested(source_id)   : user clicked Edit Zones
    start_requested(source_id)        : user clicked Start
    stop_requested(source_id)         : user clicked Stop
    """

    edit_zones_requested = Signal(str)
    start_requested      = Signal(str)
    stop_requested       = Signal(str)

    def __init__(self, source_id: str, source_name: str, parent=None) -> None:
        super().__init__(parent)
        self._source_id = source_id
        self._source_name = source_name
        self._overlay_enabled = True
        self._current_frame: QImage | None = None

        self._build_ui()
        self.update_status("offline")

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        # --- Top bar (container widget so we can hide it as one unit) ---
        self._top_bar = QWidget()
        self._top_bar.setMaximumHeight(28)
        top = QHBoxLayout(self._top_bar)
        top.setContentsMargins(4, 2, 4, 2)
        top.setSpacing(4)

        self._name_lbl = QLabel(self._source_name)
        self._name_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        self._name_lbl.setMaximumWidth(200)
        top.addWidget(self._name_lbl)

        top.addStretch()

        self._motion_lbl = QLabel("NO MOTION")
        self._motion_lbl.setStyleSheet(
            "color: #555555; font-size: 10px; font-weight: bold;"
        )
        top.addWidget(self._motion_lbl)

        top.addSpacing(8)

        self._status_lbl = QLabel("OFFLINE")
        self._status_lbl.setStyleSheet(
            f"color: {_STATUS_COLORS['offline']}; font-size: 10px; font-weight: bold;"
        )
        top.addWidget(self._status_lbl)

        root.addWidget(self._top_bar)

        # --- Video area ---
        self._video = VideoLabel()
        self._video.setText("No Signal")
        self._video.setStyleSheet(
            "background-color: #1a1a1a; color: #555555; font-size: 14px;"
        )
        root.addWidget(self._video, 1)

        # --- Stats bar ---
        self._stats_lbl = QLabel("Confirmed: 0  Candidates: 0")
        self._stats_lbl.setStyleSheet("font-size: 9px; color: #555555;")
        self._stats_lbl.setContentsMargins(4, 0, 4, 0)
        self._stats_lbl.setMaximumHeight(16)
        root.addWidget(self._stats_lbl)

        # --- Bottom controls (container widget so we can hide it as one unit) ---
        self._controls_bar = QWidget()
        self._controls_bar.setMaximumHeight(30)
        bottom = QHBoxLayout(self._controls_bar)
        bottom.setContentsMargins(4, 2, 4, 2)
        bottom.setSpacing(4)

        self._overlay_btn = QPushButton("Hide Overlay")
        self._overlay_btn.setFixedHeight(22)
        self._overlay_btn.setCheckable(True)
        self._overlay_btn.setChecked(False)
        self._overlay_btn.clicked.connect(self._toggle_overlay)
        bottom.addWidget(self._overlay_btn)

        self._zones_btn = QPushButton("Edit Zones")
        self._zones_btn.setFixedHeight(22)
        self._zones_btn.clicked.connect(
            lambda: self.edit_zones_requested.emit(self._source_id)
        )
        bottom.addWidget(self._zones_btn)

        bottom.addStretch()

        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedHeight(22)
        self._start_btn.clicked.connect(
            lambda: self.start_requested.emit(self._source_id)
        )
        bottom.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(22)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(
            lambda: self.stop_requested.emit(self._source_id)
        )
        bottom.addWidget(self._stop_btn)

        root.addWidget(self._controls_bar)

        # Border frame
        self.setStyleSheet(
            "SourceViewWidget { border: 1px solid #cccccc; background: #ffffff; }"
        )

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------
    # Public slots (connected to SourceWorker signals via MainWindow)
    # ------------------------------------------------------------------

    def on_frame(self, source_id: str, qimage: QImage) -> None:
        if source_id != self._source_id:
            return
        self._current_frame = qimage
        pixmap = QPixmap.fromImage(qimage)
        self._video.set_pixmap(pixmap)

    def on_status_changed(self, source_id: str, status: str) -> None:
        if source_id != self._source_id:
            return
        self.update_status(status)

    def on_motion_detected(self, source_id: str, _frame) -> None:
        if source_id != self._source_id:
            return
        self._motion_lbl.setText("MOTION")
        self._motion_lbl.setStyleSheet(
            "color: #cc0000; font-size: 10px; font-weight: bold;"
        )

    def on_motion_ended(self, source_id: str, _duration) -> None:
        if source_id != self._source_id:
            return
        self._motion_lbl.setText("NO MOTION")
        self._motion_lbl.setStyleSheet(
            "color: #555555; font-size: 10px; font-weight: bold;"
        )

    def on_stats_updated(self, source_id: str, stats: dict) -> None:
        if source_id != self._source_id:
            return
        confirmed = stats.get("confirmed", 0)
        candidates = stats.get("candidates", 0)
        warmup = stats.get("warmup_remaining", 0)
        det_ms = stats.get("det_ms", 0)
        warmup_str = f"  [Warming up: {warmup}]" if warmup > 0 else ""
        self._stats_lbl.setText(
            f"Confirmed: {confirmed}  Candidates: {candidates}  Det: {det_ms:.0f}ms{warmup_str}"
        )

    # ------------------------------------------------------------------

    def set_minimal_mode(self, minimal: bool) -> None:
        """Hide/show chrome (header, stats, controls) to maximise video area."""
        self._top_bar.setVisible(not minimal)
        self._stats_lbl.setVisible(not minimal)
        self._controls_bar.setVisible(not minimal)
        # Remove border in minimal mode so no outline distracts
        if minimal:
            self.setStyleSheet(
                "SourceViewWidget { border: none; background: #000000; }"
            )
        else:
            self.setStyleSheet(
                "SourceViewWidget { border: 1px solid #cccccc; background: #ffffff; }"
            )

    def update_status(self, status: str) -> None:
        color = _STATUS_COLORS.get(status, "#555555")
        self._status_lbl.setText(status.upper())
        self._status_lbl.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold;"
        )
        is_running = status == "connected"
        self._start_btn.setEnabled(not is_running)
        self._stop_btn.setEnabled(is_running)
        if status == "connected":
            self._video.setStyleSheet("background-color: #1a1a1a;")
            self._video.setText("")
        else:
            self._video.setText("No Signal")

    def update_source_name(self, name: str) -> None:
        self._source_name = name
        self._name_lbl.setText(name)

    def _toggle_overlay(self, checked: bool) -> None:
        self._overlay_enabled = not checked
        self._overlay_btn.setText("Show Overlay" if checked else "Hide Overlay")

    @property
    def source_id(self) -> str:
        return self._source_id

    def get_current_frame(self) -> QImage | None:
        return self._current_frame
