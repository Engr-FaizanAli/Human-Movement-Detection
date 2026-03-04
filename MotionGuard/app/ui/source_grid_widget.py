"""
source_grid_widget.py - Adaptive grid layout for any number of active sources.

Layout rules:
  - 0 active: placeholder text
  - 1+ active: dynamic rows x columns grid, sized close to square
"""

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QGridLayout,
    QSizePolicy,
    QWidget,
)

from ui.source_view_widget import SourceViewWidget


class SourceGridWidget(QWidget):
    """
    Central grid that holds SourceViewWidget cells.

    Call relayout(active_ids, view_map) after the active source list changes.
    view_map: dict[source_id -> SourceViewWidget]
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(4)
        self._current_ids: list[str] = []

        self._placeholder = QLabel("No sources active.\nSelect a source from the left panel and click Start.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888888; font-size: 14px;")
        self._grid.addWidget(self._placeholder, 0, 0)

    def relayout(
        self,
        active_ids: list[str],
        view_map: dict[str, SourceViewWidget],
    ) -> None:
        """
        Rebuild the grid to show all active sources.

        Parameters
        ----------
        active_ids : ordered list of active source IDs
        view_map   : source_id -> SourceViewWidget (must include all active IDs)
        """
        self._clear_grid()
        self._current_ids = list(active_ids)
        n = len(self._current_ids)

        if n == 0:
            self._grid.addWidget(self._placeholder, 0, 0)
            self._placeholder.show()
            self._grid.setColumnStretch(0, 1)
            self._grid.setRowStretch(0, 1)
            return

        self._placeholder.hide()
        views = [view_map[sid] for sid in self._current_ids if sid in view_map]
        if not views:
            return

        cols = max(1, math.ceil(math.sqrt(len(views))))
        rows = max(1, math.ceil(len(views) / cols))

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= len(views):
                    break
                v = views[idx]
                self._grid.addWidget(v, r, c)
                v.show()
                idx += 1

        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
        for r in range(rows):
            self._grid.setRowStretch(r, 1)

    def _clear_grid(self) -> None:
        """Remove all widgets from the grid without deleting them."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item and item.widget():
                item.widget().hide()

        # Reset stretch factors for a reasonable upper bound.
        for i in range(64):
            self._grid.setColumnStretch(i, 0)
            self._grid.setRowStretch(i, 0)
