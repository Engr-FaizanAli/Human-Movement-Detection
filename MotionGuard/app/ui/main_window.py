"""
main_window.py — QMainWindow: the primary application shell.

Layout:
  Left   : Device tree (recorders → channels, cameras, offline sources)
  Center : Source grid (1–3 active sources)
  Right  : Motion events list + source quick controls
  Top    : Toolbar (Add Recorder, Add Camera, Offline, Scan ONVIF, Settings, Logs, Start All, Stop All)
  Bottom : Status bar
  Tray   : System tray icon with menu
"""

import datetime
import logging
import os
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QThread, Signal, Slot, QTimer
from PySide6.QtGui import (
    QIcon, QAction, QImage, QColor, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QToolBar, QTreeWidget, QTreeWidgetItem, QLabel, QListWidget,
    QListWidgetItem, QStatusBar, QFileDialog, QMessageBox,
    QSystemTrayIcon, QMenu, QPushButton, QFrame,
)

from core.alerts import AlertManager
from core.source_manager import SourceManager
from storage import repositories as repo
from ui.source_grid_widget import SourceGridWidget
from ui.source_view_widget import SourceViewWidget
from ui.polygon_editor import PolygonEditor
from ui.dialogs.add_recorder_dialog import AddRecorderDialog
from ui.dialogs.add_camera_dialog import AddCameraDialog
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.diagnostics_dialog import DiagnosticsDialog
from utils.paths import logo_path

log = logging.getLogger(__name__)

_OFFLINE_EXTENSIONS = "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.m4v *.webm)"


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MotionGuard")
        self.setMinimumSize(1100, 650)
        self.resize(1400, 800)

        # Core objects
        self.alert_manager = AlertManager()
        self.source_manager = SourceManager()
        self._source_views: dict[str, SourceViewWidget] = {}
        self._error_log: dict[str, str] = {}

        # Load settings
        self._auto_start = repo.get_setting("auto_start", "0") == "1"

        self._setup_icon()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._setup_tray()
        self._setup_shortcuts()

        # Connect source manager signals
        self.source_manager.source_added.connect(self._on_source_changed)
        self.source_manager.source_removed.connect(self._on_source_changed)
        self.source_manager.active_changed.connect(self._on_active_changed)
        self.source_manager.at_capacity.connect(self._on_at_capacity)

        # Populate device tree
        self._refresh_device_tree()

        # Apply alert settings
        self.alert_manager.set_enabled(repo.get_setting("alarm_enabled", "1") == "1")
        self.alert_manager.set_volume(float(repo.get_setting("alarm_volume", "0.8")))
        self.alert_manager.set_snapshot_enabled(repo.get_setting("snapshot_enabled", "1") == "1")

        # Cleanup old snapshots on startup
        try:
            retention = int(repo.get_setting("snapshot_retention_days", "30"))
            self.alert_manager.cleanup_old_snapshots(retention)
        except Exception:
            pass

        if self._auto_start:
            self._start_all()

    # ------------------------------------------------------------------
    # Icon
    # ------------------------------------------------------------------

    def _setup_icon(self) -> None:
        icon_p = logo_path()
        if icon_p.exists():
            self.setWindowIcon(QIcon(str(icon_p)))

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setStyleSheet("QToolBar { spacing: 6px; padding: 4px; }")
        self.addToolBar(tb)
        self._toolbar = tb

        # Logo
        icon_p = logo_path()
        if icon_p.exists():
            logo_lbl = QLabel()
            from PySide6.QtGui import QPixmap
            pix = QPixmap(str(icon_p)).scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            logo_lbl.setContentsMargins(4, 0, 8, 0)
            tb.addWidget(logo_lbl)

        title_lbl = QLabel("MotionGuard")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 14px; padding-right: 12px;")
        tb.addWidget(title_lbl)

        tb.addSeparator()

        self._tb_action("Add Recorder",    self._add_recorder, tb)
        self._tb_action("Add Camera",      self._add_camera, tb)
        self._tb_action("Offline Mode",    self._open_offline, tb)
        self._tb_action("Scan ONVIF",      self._scan_onvif, tb)
        tb.addSeparator()
        self._tb_action("Settings",        self._open_settings, tb)
        self._tb_action("Diagnostics",     self._open_diagnostics, tb)
        tb.addSeparator()
        self._tb_action("Start All",       self._start_all, tb)
        self._tb_action("Stop All",        self.source_manager.stop_all, tb)
        tb.addSeparator()

        self._expand_btn = QPushButton("Expand View")
        self._expand_btn.setFixedHeight(26)
        self._expand_btn.setCheckable(True)
        self._expand_btn.clicked.connect(self._toggle_expand_view)
        tb.addWidget(self._expand_btn)

        self._fullscreen_btn = QPushButton("Full Screen")
        self._fullscreen_btn.setFixedHeight(26)
        self._fullscreen_btn.setCheckable(True)
        self._fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        tb.addWidget(self._fullscreen_btn)

    def _tb_action(self, label: str, callback, tb: QToolBar) -> QAction:
        btn = QPushButton(label)
        btn.setFixedHeight(26)
        btn.clicked.connect(callback)
        tb.addWidget(btn)
        return QAction(label, self)

    # ------------------------------------------------------------------
    # Central widget
    # ------------------------------------------------------------------

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = splitter

        # --- Left panel: Device tree ---
        left = QWidget()
        self._left_panel = left
        left.setMinimumWidth(200)
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        left_title = QLabel("Devices")
        left_title.setStyleSheet("font-weight: bold; padding: 4px; background: #f0f0f0; border-bottom: 1px solid #ccc;")
        left_layout.addWidget(left_title)

        self._device_tree = QTreeWidget()
        self._device_tree.setHeaderHidden(True)
        self._device_tree.setColumnCount(1)
        self._device_tree.itemDoubleClicked.connect(self._on_tree_double_click)
        self._device_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._device_tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        left_layout.addWidget(self._device_tree)

        splitter.addWidget(left)

        # --- Center: Source grid ---
        self._grid = SourceGridWidget()
        splitter.addWidget(self._grid)

        # --- Right panel: Events ---
        right = QWidget()
        self._right_panel = right
        right.setMinimumWidth(220)
        right.setMaximumWidth(300)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        right_title = QLabel("Motion Events")
        right_title.setStyleSheet("font-weight: bold; padding: 4px; background: #f0f0f0; border-bottom: 1px solid #ccc;")
        right_layout.addWidget(right_title)

        self._event_list = QListWidget()
        self._event_list.setStyleSheet("font-size: 10px;")
        right_layout.addWidget(self._event_list)

        clear_btn = QPushButton("Clear List")
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._event_list.clear)
        right_layout.addWidget(clear_btn)

        splitter.addWidget(right)

        splitter.setSizes([240, 900, 260])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_lbl = QLabel("Ready")
        sb.addWidget(self._status_lbl)
        self._active_count_lbl = QLabel("Active: 0 / 3")
        sb.addPermanentWidget(self._active_count_lbl)

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_tray(self) -> None:
        icon_p = logo_path()
        icon = QIcon(str(icon_p)) if icon_p.exists() else self.style().standardIcon(
            self.style().StandardPixmap.SP_ComputerIcon
        )
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("MotionGuard")
        self._tray.activated.connect(self._on_tray_activated)

        menu = QMenu()
        menu.addAction("Open MotionGuard", self._restore_window)
        menu.addSeparator()
        menu.addAction("Start Detection",  self._start_all)
        menu.addAction("Stop Detection",   self.source_manager.stop_all)
        menu.addSeparator()
        menu.addAction("Quit", self._quit)

        self._tray.setContextMenu(menu)
        self._tray.show()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._restore_window()

    def _restore_window(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def closeEvent(self, event) -> None:
        # Minimize to tray instead of closing
        event.ignore()
        self.hide()
        self._tray.showMessage(
            "MotionGuard",
            "Running in system tray. Right-click the tray icon to quit.",
            QSystemTrayIcon.MessageIcon.Information,
            2000,
        )

    def _quit(self) -> None:
        log.info("Quitting — stopping all sources")
        self.source_manager.stop_all()
        self._tray.hide()
        from PySide6.QtWidgets import QApplication
        QApplication.quit()

    # ------------------------------------------------------------------
    # Device tree
    # ------------------------------------------------------------------

    def _refresh_device_tree(self) -> None:
        self._device_tree.clear()

        # Recorders
        recorders = repo.get_recorders()
        if recorders:
            rec_root = QTreeWidgetItem(["Recorders"])
            rec_root.setExpanded(True)
            self._device_tree.addTopLevelItem(rec_root)
            for rec in recorders:
                rec_item = QTreeWidgetItem([rec["name"]])
                rec_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "recorder", "id": rec["id"]})
                for ch in repo.get_channels(rec["id"]):
                    ch_name = ch["friendly_name"] or f"CH{ch['channel_num']}"
                    ch_item = QTreeWidgetItem([ch_name])
                    ch_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "channel", "id": ch["id"]})
                    rec_item.addChild(ch_item)
                rec_root.addChild(rec_item)

        # Cameras
        cameras = repo.get_cameras()
        if cameras:
            cam_root = QTreeWidgetItem(["Cameras"])
            cam_root.setExpanded(True)
            self._device_tree.addTopLevelItem(cam_root)
            for cam in cameras:
                cam_item = QTreeWidgetItem([cam["name"]])
                cam_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "camera", "id": cam["id"]})
                cam_root.addChild(cam_item)

        # Offline
        offline = repo.get_offline_sources()
        if offline:
            off_root = QTreeWidgetItem(["Offline Files"])
            off_root.setExpanded(True)
            self._device_tree.addTopLevelItem(off_root)
            for src in offline:
                src_item = QTreeWidgetItem([src["name"]])
                src_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "offline", "id": src["id"]})
                off_root.addChild(src_item)

        self._device_tree.expandAll()

    def _on_tree_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get("type") in ("channel", "camera", "offline"):
            self._toggle_source(data["id"])

    def _show_tree_context_menu(self, pos) -> None:
        item = self._device_tree.itemAt(pos)
        if not item:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        menu = QMenu(self)
        dtype = data.get("type")
        sid = data.get("id")

        if dtype in ("channel", "camera", "offline"):
            is_active = self.source_manager.is_active(sid)
            if is_active:
                menu.addAction("Stop", lambda: self._stop_source(sid))
            else:
                menu.addAction("Start", lambda: self._start_source(sid))
            menu.addSeparator()
            menu.addAction("Edit Zones", lambda: self._open_zone_editor(sid, dtype))

        if dtype == "recorder":
            menu.addAction("Edit Recorder", lambda: self._edit_recorder(sid))
            menu.addAction("Delete Recorder", lambda: self._delete_recorder(sid))

        if dtype == "camera":
            menu.addAction("Delete Camera", lambda: self._delete_camera(sid))

        if dtype == "offline":
            menu.addAction("Delete", lambda: self._delete_offline(sid))

        menu.exec(self._device_tree.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Source start / stop
    # ------------------------------------------------------------------

    def _toggle_source(self, source_id: str) -> None:
        if self.source_manager.is_active(source_id):
            self._stop_source(source_id)
        else:
            self._start_source(source_id)

    def _start_source(self, source_id: str) -> None:
        # Ensure we have a SourceViewWidget for this source
        if source_id not in self._source_views:
            name = self._get_source_name(source_id)
            view = SourceViewWidget(source_id, name)
            view.start_requested.connect(self._start_source)
            view.stop_requested.connect(self._stop_source)
            view.edit_zones_requested.connect(self._open_zone_editor_from_view)
            self._source_views[source_id] = view

        ok = self.source_manager.start_source(source_id)
        if not ok:
            return

        # Connect worker signals to the view
        worker = self.source_manager.get_worker(source_id)
        if worker:
            view = self._source_views[source_id]
            worker.frame_ready.connect(view.on_frame)
            worker.status_changed.connect(view.on_status_changed)
            worker.motion_detected.connect(view.on_motion_detected)
            worker.motion_ended.connect(view.on_motion_ended)
            worker.stats_updated.connect(view.on_stats_updated)
            # Connect motion to alert manager
            worker.motion_detected.connect(self._on_worker_motion_detected)
            worker.motion_ended.connect(self._on_worker_motion_ended)
            worker.error_occurred.connect(self._on_worker_error)

        self._grid.relayout(self.source_manager.get_active_ids(), self._source_views)
        self._status_lbl.setText(f"Started: {self._get_source_name(source_id)}")

    def _stop_source(self, source_id: str) -> None:
        self.source_manager.stop_source(source_id)
        self._grid.relayout(self.source_manager.get_active_ids(), self._source_views)

    def _start_all(self) -> None:
        for src in self.source_manager.get_all_sources():
            if src["enabled"] and not src["is_active"]:
                if self.source_manager.active_count() >= 3:
                    break
                self._start_source(src["source_id"])

    # ------------------------------------------------------------------
    # Source manager signal handlers
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_source_changed(self, _data) -> None:
        self._refresh_device_tree()

    @Slot(list)
    def _on_active_changed(self, active_ids: list) -> None:
        self._active_count_lbl.setText(f"Active: {len(active_ids)} / 3")
        self._grid.relayout(active_ids, self._source_views)

    @Slot()
    def _on_at_capacity(self) -> None:
        QMessageBox.information(
            self,
            "Source Limit Reached",
            "Maximum of 3 sources can be active simultaneously.\n"
            "Stop an active source before starting another.",
        )

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    @Slot(str, object)
    def _on_worker_motion_detected(self, source_id: str, frame) -> None:
        name = self._get_source_name(source_id)
        self.alert_manager.on_motion_detected(
            source_id=source_id,
            source_type=self._get_source_type(source_id),
            source_name=name,
            frame=frame,
        )
        # Add to event list
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{ts}] MOTION — {name}")
        item.setForeground(QColor("#8b0000"))
        self._event_list.insertItem(0, item)
        # Keep list manageable
        while self._event_list.count() > 200:
            self._event_list.takeItem(self._event_list.count() - 1)

    @Slot(str, float)
    def _on_worker_motion_ended(self, source_id: str, duration: float) -> None:
        name = self._get_source_name(source_id)
        stype = self._get_source_type(source_id)
        self.alert_manager.on_motion_ended(source_id, stype, name, duration)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{ts}] ended ({duration:.1f}s) — {name}")
        item.setForeground(QColor("#444444"))
        self._event_list.insertItem(0, item)

    @Slot(str, str)
    def _on_worker_error(self, source_id: str, message: str) -> None:
        self._error_log[source_id] = message
        log.warning("Worker error [%s]: %s", source_id[:8], message)

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _add_recorder(self) -> None:
        dlg = AddRecorderDialog(self)
        if dlg.exec():
            self._refresh_device_tree()
            self.source_manager.source_added.emit({})

    def _add_camera(self) -> None:
        dlg = AddCameraDialog(self)
        if dlg.exec():
            self._refresh_device_tree()
            self.source_manager.source_added.emit({})

    def _open_offline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", _OFFLINE_EXTENSIONS
        )
        if not path:
            return
        name = Path(path).stem
        oid = repo.add_offline_source(name, path)
        self._refresh_device_tree()
        self.source_manager.source_added.emit({})

        # Ask whether to set exclusion zones before starting
        reply = QMessageBox.question(
            self,
            "Exclusion Zones",
            "Do you want to set exclusion zones before starting detection?\n\n"
            "Zones let you ignore static regions (e.g. trees, roads) to reduce false alarms.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._open_zone_editor_before_start(oid, "offline", path)
        else:
            self._start_source(oid)

    def _scan_onvif(self) -> None:
        dlg = AddCameraDialog(self)
        dlg._tabs.setCurrentIndex(0)
        dlg._start_scan()
        if dlg.exec():
            self._refresh_device_tree()

    def _open_zone_editor_before_start(
        self, source_id: str, source_type: str, file_path: str
    ) -> None:
        """Open polygon editor using a frame grabbed directly from the file, then start."""
        import cv2

        bg_image = QImage(640, 480, QImage.Format.Format_RGB888)
        bg_image.fill(QColor(80, 80, 80))
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    import numpy as np
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, c = frame_rgb.shape
                    bg_image = QImage(
                        frame_rgb.data, w, h, w * c, QImage.Format.Format_RGB888
                    ).copy()
            cap.release()
        except Exception as exc:
            log.warning("Could not read preview frame from %s: %s", file_path, exc)

        existing = repo.get_zones(source_id, source_type)
        dlg = PolygonEditor(
            source_id=source_id,
            source_type=source_type,
            background_frame=bg_image,
            existing_zones=existing,
            parent=self,
        )
        dlg.zones_saved.connect(
            lambda zones: self._on_zones_saved(source_id, source_type, zones)
        )
        dlg.exec()
        # Start the source whether zones were saved or the dialog was cancelled
        self._start_source(source_id)

    def _setup_shortcuts(self) -> None:
        """Install keyboard shortcuts for view modes."""
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self._exit_fullscreen)
        f11 = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        f11.activated.connect(self._toggle_fullscreen)

    # ------------------------------------------------------------------
    # View modes: expand and fullscreen
    # ------------------------------------------------------------------

    def _toggle_expand_view(self) -> None:
        """Hide/show side panels to give more space to the video grid."""
        expanding = self._expand_btn.isChecked()
        if expanding:
            self._left_panel.hide()
            self._right_panel.hide()
            self._expand_btn.setText("Restore View")
            # Defer setSizes so Qt processes the hide() before we redistribute space
            QTimer.singleShot(0, lambda: self._splitter.setSizes([0, 99999, 0]))
        else:
            self._left_panel.show()
            self._right_panel.show()
            self._expand_btn.setText("Expand View")
            QTimer.singleShot(0, lambda: self._splitter.setSizes([240, 900, 260]))

    def _toggle_fullscreen(self) -> None:
        """Toggle true OS fullscreen — hides toolbar, status bar, and side panels."""
        if self.isFullScreen():
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        self._left_panel.hide()
        self._right_panel.hide()
        self._toolbar.hide()
        self.statusBar().hide()
        for view in self._source_views.values():
            view.set_minimal_mode(True)
        self._fullscreen_btn.setChecked(True)
        self._fullscreen_btn.setText("Exit Full Screen")
        self.showFullScreen()
        # Defer setSizes until after showFullScreen() resizes the window
        QTimer.singleShot(0, lambda: self._splitter.setSizes([0, 99999, 0]))

    def _exit_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        self.showNormal()
        self._toolbar.show()
        self.statusBar().show()
        for view in self._source_views.values():
            view.set_minimal_mode(False)
        # Only restore side panels if expand mode is not active
        if not self._expand_btn.isChecked():
            self._left_panel.show()
            self._right_panel.show()
            QTimer.singleShot(0, lambda: self._splitter.setSizes([240, 900, 260]))
        else:
            # Expand mode still active — keep panels hidden, center fills all
            QTimer.singleShot(0, lambda: self._splitter.setSizes([0, 99999, 0]))
        self._fullscreen_btn.setChecked(False)
        self._fullscreen_btn.setText("Full Screen")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.exec()
        # Re-apply alert settings
        self.alert_manager.set_enabled(repo.get_setting("alarm_enabled", "1") == "1")
        self.alert_manager.set_volume(float(repo.get_setting("alarm_volume", "0.8")))
        self.alert_manager.set_snapshot_enabled(repo.get_setting("snapshot_enabled", "1") == "1")

    def _open_diagnostics(self) -> None:
        dlg = DiagnosticsDialog(
            source_manager=self.source_manager,
            error_log=self._error_log,
            parent=self,
        )
        dlg.exec()

    # ------------------------------------------------------------------
    # Polygon editor
    # ------------------------------------------------------------------

    def _open_zone_editor_from_view(self, source_id: str) -> None:
        self._open_zone_editor(source_id, self._get_source_type(source_id))

    def _open_zone_editor(self, source_id: str, source_type: str) -> None:
        # Get a reference frame from the current worker
        bg_image: QImage | None = None
        view = self._source_views.get(source_id)
        if view:
            bg_image = view.get_current_frame()

        if bg_image is None:
            # No live frame — create a gray placeholder
            bg_image = QImage(640, 480, QImage.Format.Format_RGB888)
            bg_image.fill(QColor(80, 80, 80))

        existing = repo.get_zones(source_id, source_type)

        dlg = PolygonEditor(
            source_id=source_id,
            source_type=source_type,
            background_frame=bg_image,
            existing_zones=existing,
            parent=self,
        )
        dlg.zones_saved.connect(lambda zones: self._on_zones_saved(source_id, source_type, zones))
        dlg.exec()

    def _on_zones_saved(
        self, source_id: str, source_type: str, zones: list[dict]
    ) -> None:
        repo.save_zones(source_id, source_type, zones)
        self.source_manager.update_zones(source_id, zones)
        log.info("Zones updated for source %s: %d zone(s)", source_id[:8], len(zones))

    # ------------------------------------------------------------------
    # Recorder/camera management
    # ------------------------------------------------------------------

    def _edit_recorder(self, recorder_id: str) -> None:
        dlg = AddRecorderDialog(self, recorder_id=recorder_id)
        if dlg.exec():
            self._refresh_device_tree()

    def _delete_recorder(self, recorder_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Recorder",
            "Delete this recorder and all its channels?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Stop active channels first
            for ch in repo.get_channels(recorder_id):
                self.source_manager.stop_source(ch["id"])
            repo.delete_recorder(recorder_id)
            self._refresh_device_tree()

    def _delete_camera(self, camera_id: str) -> None:
        self.source_manager.stop_source(camera_id)
        repo.delete_camera(camera_id)
        self._source_views.pop(camera_id, None)
        self._refresh_device_tree()

    def _delete_offline(self, source_id: str) -> None:
        self.source_manager.stop_source(source_id)
        repo.delete_offline_source(source_id)
        self._source_views.pop(source_id, None)
        self._refresh_device_tree()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_source_name(self, source_id: str) -> str:
        for src in self.source_manager.get_all_sources():
            if src["source_id"] == source_id:
                return src.get("source_name", source_id)
        return source_id

    def _get_source_type(self, source_id: str) -> str:
        for src in self.source_manager.get_all_sources():
            if src["source_id"] == source_id:
                return src.get("source_type", "camera")
        return "camera"
