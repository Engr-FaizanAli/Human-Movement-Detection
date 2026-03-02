"""
polygon_editor.py — Interactive polygon exclusion zone editor.

Opens as a full-screen overlay dialog on top of a reference video frame.
User draws polygons by clicking; right-click closes the current polygon.
Saves normalized (0..1) vertices to the DB via a signal.

Controls:
  Left-click      Add vertex to current polygon
  Right-click     Close current polygon (needs 3+ vertices)
  Delete key      Delete selected zone or last vertex of active polygon
  Ctrl+Z          Undo last vertex of active polygon
  Enter/S         Save all zones
  Escape          Cancel without saving
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QKeySequence,
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QWidget,
)

from core.mask_engine import pixel_to_normalized

log = logging.getLogger(__name__)

_ZONE_COLORS = [
    QColor(220, 50, 50, 120),    # red
    QColor(50, 50, 220, 120),    # blue
    QColor(50, 180, 50, 120),    # green
    QColor(180, 100, 50, 120),   # orange
    QColor(140, 50, 180, 120),   # purple
]
_BORDER_COLOR = QColor(200, 30, 30, 220)
_ACTIVE_COLOR = QColor(255, 200, 0, 180)
_VERTEX_COLOR = QColor(255, 255, 255, 230)
_SEL_VERTEX_COLOR = QColor(255, 80, 80, 230)
_VERTEX_RADIUS = 5


class _CanvasWidget(QWidget):
    """The actual drawing canvas inside the dialog."""

    def __init__(self, background: QImage, parent=None):
        super().__init__(parent)
        self._bg_pixmap = QPixmap.fromImage(background)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)

        # State
        self.completed_zones: list[dict] = []   # {"name": str, "px_vertices": [[x,y],...]}
        self._active_vertices: list[list[int]] = []   # current in-progress polygon
        self._selected_zone: int | None = None
        self._hover_pos: QPoint | None = None
        self._frame_offset: tuple = (0, 0, 0, 0)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return self._bg_pixmap.size() if not self._bg_pixmap.isNull() else QSize(960, 540)

    # --- painting ---
    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background frame
            scaled_bg = self._bg_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (self.width() - scaled_bg.width()) // 2
            oy = (self.height() - scaled_bg.height()) // 2
            painter.drawPixmap(ox, oy, scaled_bg)
            self._frame_offset = (ox, oy, scaled_bg.width(), scaled_bg.height())

            # Completed zones
            for i, zone in enumerate(self.completed_zones):
                pts = self._to_widget(zone["px_vertices"])
                color = _ZONE_COLORS[i % len(_ZONE_COLORS)]
                border_color = _SEL_VERTEX_COLOR if i == self._selected_zone else _BORDER_COLOR

                # Fill polygon
                from PySide6.QtGui import QPolygon
                poly = QPolygon([QPoint(x, y) for x, y in pts])
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(border_color, 2))
                painter.drawPolygon(poly)

                # Zone label
                if pts:
                    cx = sum(p[0] for p in pts) // len(pts)
                    cy = sum(p[1] for p in pts) // len(pts)
                    painter.setPen(QPen(QColor(255, 255, 255, 220)))
                    painter.drawText(QPoint(cx - 20, cy), zone["name"])

                # Vertices
                painter.setBrush(QBrush(_VERTEX_COLOR))
                painter.setPen(QPen(QColor(100, 100, 100, 200), 1))
                for px, py in pts:
                    painter.drawEllipse(QPoint(px, py), _VERTEX_RADIUS, _VERTEX_RADIUS)

            # Active polygon being drawn
            if self._active_vertices:
                pts = self._to_widget(self._active_vertices)
                painter.setPen(QPen(_ACTIVE_COLOR, 2, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(255, 200, 0, 40)))
                for i in range(len(pts) - 1):
                    painter.drawLine(QPoint(*pts[i]), QPoint(*pts[i + 1]))
                # Line to cursor
                if self._hover_pos and pts:
                    painter.drawLine(QPoint(*pts[-1]), self._hover_pos)
                # Vertices
                painter.setBrush(QBrush(_ACTIVE_COLOR))
                painter.setPen(QPen(QColor(150, 100, 0), 1))
                for px, py in pts:
                    painter.drawEllipse(QPoint(px, py), _VERTEX_RADIUS, _VERTEX_RADIUS)

        except Exception:
            log.exception("Error in polygon editor paintEvent")
        finally:
            painter.end()

    def mouseMoveEvent(self, event):
        self._hover_pos = event.position().toPoint()
        self.update()

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            px = self._to_frame(pos)
            if px:
                self._active_vertices.append(px)
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._close_polygon()

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Z and mods & Qt.KeyboardModifier.ControlModifier:
            if self._active_vertices:
                self._active_vertices.pop()
                self.update()
        elif key == Qt.Key.Key_Delete:
            if self._selected_zone is not None and self.completed_zones:
                self.completed_zones.pop(self._selected_zone)
                self._selected_zone = None
                self.update()
            elif self._active_vertices:
                self._active_vertices.pop()
                self.update()

    def close_current_polygon(self):
        self._close_polygon()

    def _close_polygon(self):
        if len(self._active_vertices) < 3:
            return
        name = f"Zone {len(self.completed_zones) + 1}"
        self.completed_zones.append(
            {"name": name, "px_vertices": list(self._active_vertices)}
        )
        self._active_vertices = []
        self.update()

    # --- coordinate mapping ---
    def _to_frame(self, widget_pos: QPoint) -> list[int] | None:
        """Convert widget pixel → original frame pixel (pre-scale)."""
        if not hasattr(self, "_frame_offset"):
            return None
        ox, oy, sw, sh = self._frame_offset
        x = widget_pos.x() - ox
        y = widget_pos.y() - oy
        if x < 0 or y < 0 or x > sw or y > sh:
            return None
        orig_w = self._bg_pixmap.width()
        orig_h = self._bg_pixmap.height()
        if sw == 0 or sh == 0:
            return None
        fx = int(x * orig_w / sw)
        fy = int(y * orig_h / sh)
        return [fx, fy]

    def _to_widget(self, px_vertices: list[list[int]]) -> list[tuple[int, int]]:
        """Convert original frame pixels → widget pixels (scaled)."""
        if not hasattr(self, "_frame_offset") or not px_vertices:
            return []
        ox, oy, sw, sh = self._frame_offset
        orig_w = self._bg_pixmap.width()
        orig_h = self._bg_pixmap.height()
        if orig_w == 0 or orig_h == 0:
            return []
        result = []
        for fx, fy in px_vertices:
            wx = ox + int(fx * sw / orig_w)
            wy = oy + int(fy * sh / orig_h)
            result.append((wx, wy))
        return result


class PolygonEditor(QDialog):
    """
    Full-screen polygon editor dialog.

    Signals
    -------
    zones_saved(list)  — list of {"name": str, "vertices": [[x_norm, y_norm]]}
    """

    zones_saved = Signal(list)

    def __init__(
        self,
        source_id: str,
        source_type: str,
        background_frame: QImage,
        existing_zones: list[dict] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source_id = source_id
        self._source_type = source_type
        self._frame_w = background_frame.width()
        self._frame_h = background_frame.height()

        self.setWindowTitle(f"Edit Exclusion Zones — {source_id[:16]}")
        self.setMinimumSize(800, 500)
        self.showMaximized()   # open at full screen size for best editing experience

        self._build_ui(background_frame)

        # Load existing zones
        if existing_zones:
            for zone in existing_zones:
                norm_verts = zone.get("vertices", [])
                px_verts = [
                    [int(x * self._frame_w), int(y * self._frame_h)]
                    for x, y in norm_verts
                ]
                self._canvas.completed_zones.append(
                    {"name": zone.get("name", "Zone"), "px_vertices": px_verts}
                )
            self._canvas.update()

    def _build_ui(self, background_frame: QImage) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Instructions
        info = QLabel(
            "Left-click: add vertex   |   Right-click: close polygon   |   "
            "Ctrl+Z: undo vertex   |   Del: remove zone/vertex"
        )
        info.setStyleSheet("font-size: 11px; color: #555555; padding: 4px;")
        layout.addWidget(info)

        # Canvas
        self._canvas = _CanvasWidget(background_frame)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self._canvas, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        close_btn = QPushButton("Close Polygon (Right-click)")
        close_btn.setFixedHeight(28)
        close_btn.clicked.connect(self._canvas.close_current_polygon)
        btn_row.addWidget(close_btn)

        clear_btn = QPushButton("Clear All Zones")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(28)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save Zones")
        save_btn.setFixedHeight(28)
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def _clear_all(self) -> None:
        self._canvas.completed_zones.clear()
        self._canvas._active_vertices.clear()
        self._canvas.update()

    def _save(self) -> None:
        # Auto-close any in-progress polygon
        if len(self._canvas._active_vertices) >= 3:
            self._canvas.close_current_polygon()

        zones = []
        for zone in self._canvas.completed_zones:
            px_verts = zone["px_vertices"]
            norm_verts = pixel_to_normalized(px_verts, self._frame_w, self._frame_h)
            zones.append({"name": zone["name"], "vertices": norm_verts})

        log.info(
            "Saving %d zone(s) for source %s", len(zones), self._source_id
        )
        self.zones_saved.emit(zones)
        self.accept()

    def keyPressEvent(self, event):
        # Forward keyboard events to canvas when it doesn't have focus
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save()
        elif key == Qt.Key.Key_Escape:
            self.reject()
        else:
            self._canvas.keyPressEvent(event)
