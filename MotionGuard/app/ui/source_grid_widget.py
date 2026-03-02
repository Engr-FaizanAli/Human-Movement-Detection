"""
source_grid_widget.py — Adaptive grid layout for 1-3 source view widgets.

Layout rules:
  0 active: placeholder text
  1 active: full-size single cell
  2 active: side-by-side (landscape container) or stacked (portrait container)
  3 active: 2x2 grid with one empty placeholder cell (bottom-right)
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QSizePolicy,
)

from ui.source_view_widget import SourceViewWidget

log = logging.getLogger(__name__)


class _EmptyCell(QWidget):
    """Gray placeholder for the empty 4th cell in 3-source 2x2 layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #e0e0e0; border: 1px solid #cccccc;")
        lbl = QLabel("—", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #aaaaaa; font-size: 20px;")
        from PySide6.QtWidgets import QVBoxLayout
        lay = QVBoxLayout(self)
        lay.addWidget(lbl)


class SourceGridWidget(QWidget):
    """
    Central grid that holds SourceViewWidget cells.

    Call relayout(active_ids, view_map) after the active source list changes.
    view_map: dict[source_id → SourceViewWidget]
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)
        self._current_ids: list[str] = []
        self._empty_cell = _EmptyCell()
        self._empty_cell.hide()

        # Initial placeholder
        self._placeholder = QLabel("No sources active.\nSelect a source from the left panel and click Start.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888888; font-size: 14px;")
        self._grid.addWidget(self._placeholder, 0, 0)

    # ------------------------------------------------------------------

    def relayout(
        self,
        active_ids: list[str],
        view_map: dict[str, SourceViewWidget],
    ) -> None:
        """
        Rebuild the grid to show exactly the active sources.

        Parameters
        ----------
        active_ids : ordered list of up to 3 source IDs
        view_map   : source_id → SourceViewWidget (must include all active IDs)
        """
        self._clear_grid()
        n = len(active_ids)
        self._current_ids = list(active_ids[:3])

        if n == 0:
            self._grid.addWidget(self._placeholder, 0, 0)
            self._placeholder.show()
            return

        self._placeholder.hide()
        views = [view_map[sid] for sid in self._current_ids if sid in view_map]

        if n == 1:
            self._grid.addWidget(views[0], 0, 0)
            views[0].show()
            # Only column 0 / row 0 are used — set stretch explicitly to avoid
            # competing with phantom columns left over from previous layouts
            self._grid.setColumnStretch(0, 1)
            self._grid.setRowStretch(0, 1)

        elif n == 2:
            # Landscape: side-by-side; portrait: stacked
            w = self.width()
            h = self.height()
            if w >= h:
                self._grid.addWidget(views[0], 0, 0)
                self._grid.addWidget(views[1], 0, 1)
                self._grid.setColumnStretch(0, 1)
                self._grid.setColumnStretch(1, 1)
                self._grid.setRowStretch(0, 1)
            else:
                self._grid.addWidget(views[0], 0, 0)
                self._grid.addWidget(views[1], 1, 0)
                self._grid.setColumnStretch(0, 1)
                self._grid.setRowStretch(0, 1)
                self._grid.setRowStretch(1, 1)
            for v in views:
                v.show()

        elif n == 3:
            # 2×2 grid, bottom-right empty
            self._grid.addWidget(views[0], 0, 0)
            self._grid.addWidget(views[1], 0, 1)
            self._grid.addWidget(views[2], 1, 0)
            self._empty_cell.show()
            self._grid.addWidget(self._empty_cell, 1, 1)
            for v in views:
                v.show()
            self._grid.setColumnStretch(0, 1)
            self._grid.setColumnStretch(1, 1)
            self._grid.setRowStretch(0, 1)
            self._grid.setRowStretch(1, 1)

    # ------------------------------------------------------------------

    def _clear_grid(self) -> None:
        """Remove all widgets from the grid without deleting them."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item and item.widget():
                item.widget().hide()

        # Reset stretches
        for i in range(4):
            self._grid.setColumnStretch(i, 0)
            self._grid.setRowStretch(i, 0)
