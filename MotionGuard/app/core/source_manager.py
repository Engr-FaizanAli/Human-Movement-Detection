"""
source_manager.py — Manages all SourceWorker instances.

Rules:
  - Sources can be started dynamically (no hard active-source cap)
  - start_source() returns False only on unresolved/invalid source
  - Workers are stopped gracefully on remove/disable
  - Provides a unified source list (recorders+channels, cameras, offline)
"""

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from core.motion_engine import EngineConfig
from core.source_worker import SourceConfig, SourceWorker
from storage import repositories as repo
from utils.rtsp_templates import build_channel_url, build_camera_url

log = logging.getLogger(__name__)

class SourceManager(QObject):
    # Signals for UI
    source_added     = Signal(dict)        # source info dict
    source_removed   = Signal(str)         # source_id
    active_changed   = Signal(list)        # list of active source_ids
    at_capacity      = Signal()            # reserved for backward compatibility

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: dict[str, SourceWorker] = {}   # source_id → worker

    # ------------------------------------------------------------------
    # Source inventory (from DB)
    # ------------------------------------------------------------------

    def get_all_sources(self) -> list[dict]:
        """
        Return a flat list of all sources across all types.
        Each dict contains at minimum:
            source_id, source_type, source_name, enabled, is_active
        """
        sources = []

        # Recorder channels
        for rec in repo.get_recorders():
            for ch in repo.get_channels(rec["id"]):
                sources.append(
                    {
                        "source_id": ch["id"],
                        "source_type": "recorder_channel",
                        "source_name": f"{rec['name']} — {ch['friendly_name']}",
                        "recorder_id": rec["id"],
                        "recorder_name": rec["name"],
                        "channel_num": ch["channel_num"],
                        "enabled": bool(ch["enabled"]),
                        "is_active": ch["id"] in self._workers,
                        "brand": rec["brand"],
                        "ip": rec["ip"],
                        "rtsp_port": rec["rtsp_port"],
                        "stream_pref": rec["stream_pref"],
                        "template_key": rec["template_key"],
                        "custom_template": rec["custom_template"],
                    }
                )

        # Direct cameras
        for cam in repo.get_cameras():
            sources.append(
                {
                    "source_id": cam["id"],
                    "source_type": "camera",
                    "source_name": cam["name"],
                    "rtsp_url": cam["rtsp_url"],
                    "enabled": bool(cam["enabled"]),
                    "is_active": cam["id"] in self._workers,
                }
            )

        # Offline sources
        for off in repo.get_offline_sources():
            sources.append(
                {
                    "source_id": off["id"],
                    "source_type": "offline",
                    "source_name": off["name"],
                    "file_path": off["file_path"],
                    "loop_enabled": bool(off["loop_enabled"]),
                    "enabled": True,
                    "is_active": off["id"] in self._workers,
                }
            )

        return sources

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start_source(self, source_id: str) -> bool:
        """
        Start detection on the given source.
        """
        if source_id in self._workers:
            log.debug("start_source: %s already active", source_id)
            return True

        source_info = self._resolve_source(source_id)
        if source_info is None:
            log.error("start_source: cannot resolve source %s", source_id)
            return False

        sc, ec = source_info
        worker = SourceWorker(sc, ec)
        self._workers[source_id] = worker
        worker.start()

        log.info("Started source: %s [%s]", sc.source_name, sc.source_type)
        self.active_changed.emit(self.get_active_ids())
        return True

    def stop_source(self, source_id: str) -> None:
        worker = self._workers.pop(source_id, None)
        if worker is None:
            return
        worker.stop()
        worker.wait(5000)  # wait up to 5 s
        if worker.isRunning():
            worker.terminate()
        log.info("Stopped source: %s", source_id)
        self.active_changed.emit(self.get_active_ids())

    def stop_all(self) -> None:
        for sid in list(self._workers.keys()):
            self.stop_source(sid)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_active_ids(self) -> list[str]:
        return list(self._workers.keys())

    def get_worker(self, source_id: str) -> SourceWorker | None:
        return self._workers.get(source_id)

    def is_active(self, source_id: str) -> bool:
        return source_id in self._workers

    def active_count(self) -> int:
        return len(self._workers)

    # ------------------------------------------------------------------
    # Hot-update running workers
    # ------------------------------------------------------------------

    def update_engine_config(self, source_id: str, config: EngineConfig) -> None:
        worker = self._workers.get(source_id)
        if worker:
            worker.update_engine_config(config)

    def update_zones(self, source_id: str, zones: list[dict]) -> None:
        worker = self._workers.get(source_id)
        if worker:
            worker.update_zones(zones)

    # ------------------------------------------------------------------
    # Connect worker signals to external handlers
    # ------------------------------------------------------------------

    def connect_worker_signals(self, source_id: str, handler_obj: Any) -> None:
        """
        Connect all signals of the worker for source_id to methods on handler_obj.
        Expected methods:
            on_frame_ready(source_id, qimage)
            on_motion_detected(source_id, frame)
            on_motion_ended(source_id, duration)
            on_status_changed(source_id, status)
            on_error_occurred(source_id, message)
            on_stats_updated(source_id, stats_dict)
        """
        worker = self._workers.get(source_id)
        if worker is None:
            return
        if hasattr(handler_obj, "on_frame_ready"):
            worker.frame_ready.connect(handler_obj.on_frame_ready)
        if hasattr(handler_obj, "on_motion_detected"):
            worker.motion_detected.connect(handler_obj.on_motion_detected)
        if hasattr(handler_obj, "on_motion_ended"):
            worker.motion_ended.connect(handler_obj.on_motion_ended)
        if hasattr(handler_obj, "on_status_changed"):
            worker.status_changed.connect(handler_obj.on_status_changed)
        if hasattr(handler_obj, "on_error_occurred"):
            worker.error_occurred.connect(handler_obj.on_error_occurred)
        if hasattr(handler_obj, "on_stats_updated"):
            worker.stats_updated.connect(handler_obj.on_stats_updated)

    # ------------------------------------------------------------------
    # Internal: resolve source → (SourceConfig, EngineConfig)
    # ------------------------------------------------------------------

    def _resolve_source(
        self, source_id: str
    ) -> tuple[SourceConfig, EngineConfig] | None:
        # Recorder channel?
        for rec in repo.get_recorders():
            for ch in repo.get_channels(rec["id"]):
                if ch["id"] == source_id:
                    return self._resolve_channel(rec, ch)

        # Camera?
        cam = repo.get_camera(source_id)
        if cam:
            return self._resolve_camera(cam)

        # Offline?
        off = repo.get_offline_source(source_id)
        if off:
            return self._resolve_offline(off)

        return None

    def _resolve_channel(
        self, rec: dict, ch: dict
    ) -> tuple[SourceConfig, EngineConfig]:
        password = repo.get_recorder_password(rec["id"])
        url = build_channel_url(
            brand=rec["brand"],
            ip=rec["ip"],
            port=rec["rtsp_port"],
            username=rec["username"] or "",
            password=password,
            channel_num=ch["channel_num"],
            stream_pref=rec["stream_pref"] or "sub",
            template_key=rec["template_key"],
            custom_template=rec["custom_template"],
        )
        name = f"{rec['name']} — {ch['friendly_name'] or 'CH' + str(ch['channel_num'])}"
        sc = SourceConfig(
            source_id=ch["id"],
            source_type="recorder_channel",
            source_name=name,
            rtsp_url=url,
            is_offline=False,
        )
        ec = self._load_engine_config(ch["id"], "recorder_channel")
        return sc, ec

    def _resolve_camera(self, cam: dict) -> tuple[SourceConfig, EngineConfig]:
        password = repo.get_camera_password(cam["id"])
        url = build_camera_url(cam["rtsp_url"], cam["username"] or "", password)
        sc = SourceConfig(
            source_id=cam["id"],
            source_type="camera",
            source_name=cam["name"],
            rtsp_url=url,
            is_offline=False,
        )
        ec = self._load_engine_config(cam["id"], "camera")
        return sc, ec

    def _resolve_offline(self, off: dict) -> tuple[SourceConfig, EngineConfig]:
        sc = SourceConfig(
            source_id=off["id"],
            source_type="offline",
            source_name=off["name"],
            rtsp_url=off["file_path"],
            is_offline=True,
            loop_offline=bool(off["loop_enabled"]),
        )
        ec = self._load_engine_config(off["id"], "offline")
        return sc, ec

    def _load_engine_config(
        self, source_id: str, source_type: str
    ) -> EngineConfig:
        """
        Merge global settings with per-source overrides.
        Per-source params take precedence over global defaults.
        """
        global_settings = repo.get_all_settings()
        per_source = repo.get_params(source_id, source_type)
        merged = {**global_settings, **per_source}
        return EngineConfig.from_dict(merged)
