"""
offline_player_controls.py — Playback control bar for offline video sources.

Shows Play/Pause toggle, seek slider, speed selector, and loop toggle.
Connects to a SourceWorker to control offline playback.
"""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox,
    QCheckBox, QSizePolicy,
)

log = logging.getLogger(__name__)

_SPEED_OPTIONS = [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0)]


class OfflinePlayerControls(QWidget):
    """
    Horizontal control bar for an offline video source.

    Signals
    -------
    play_pause_toggled(bool)  : True = play, False = pause
    seek_requested(int)       : frame index
    speed_changed(float)      : speed multiplier
    loop_toggled(bool)        : loop on/off
    """

    play_pause_toggled = Signal(bool)
    seek_requested     = Signal(int)
    speed_changed      = Signal(float)
    loop_toggled       = Signal(bool)

    def __init__(self, source_id: str, total_frames: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._source_id = source_id
        self._total_frames = total_frames
        self._is_playing = True
        self._seeking = False   # prevent feedback loop from slider.setValue

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Play/Pause
        self._play_btn = QPushButton("Pause")
        self._play_btn.setFixedWidth(60)
        self._play_btn.setFixedHeight(24)
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        # Seek slider
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(max(1, self._total_frames - 1))
        self._seek_slider.setValue(0)
        self._seek_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._seek_slider.sliderPressed.connect(self._on_seek_start)
        self._seek_slider.sliderReleased.connect(self._on_seek_end)
        layout.addWidget(self._seek_slider)

        # Frame counter
        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setFixedWidth(80)
        self._frame_lbl.setStyleSheet("font-size: 10px; color: #555555;")
        layout.addWidget(self._frame_lbl)

        # Speed
        layout.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.setFixedWidth(55)
        for label, _ in _SPEED_OPTIONS:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(1)  # default 1x
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_combo)

        # Loop
        self._loop_cb = QCheckBox("Loop")
        self._loop_cb.setChecked(False)
        self._loop_cb.toggled.connect(self.loop_toggled.emit)
        layout.addWidget(self._loop_cb)

    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        self._is_playing = not self._is_playing
        self._play_btn.setText("Pause" if self._is_playing else "Play")
        self.play_pause_toggled.emit(self._is_playing)

    def _on_seek_start(self) -> None:
        self._seeking = True

    def _on_seek_end(self) -> None:
        self._seeking = False
        self.seek_requested.emit(self._seek_slider.value())

    def _on_speed_changed(self, idx: int) -> None:
        _, mult = _SPEED_OPTIONS[idx]
        self.speed_changed.emit(mult)

    # ------------------------------------------------------------------
    # Called by the main window to update slider as video plays
    # ------------------------------------------------------------------

    def update_position(self, frame_idx: int) -> None:
        if self._seeking:
            return
        self._seeking = True
        self._seek_slider.setValue(frame_idx)
        self._frame_lbl.setText(f"{frame_idx} / {self._total_frames}")
        self._seeking = False

    def set_total_frames(self, total: int) -> None:
        self._total_frames = total
        self._seek_slider.setMaximum(max(1, total - 1))
        self._frame_lbl.setText(f"0 / {total}")
