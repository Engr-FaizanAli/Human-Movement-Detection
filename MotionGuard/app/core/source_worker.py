"""
source_worker.py — Per-source QThread: capture + motion detection loop.

One SourceWorker per active source (RTSP channel, direct camera, offline file).
Communicates exclusively via Qt signals — never touches UI objects directly.

Reconnect strategy (RTSP sources):
    backoff = [2, 5, 15, 30, 60] seconds
    Each failed open/read cycle advances the backoff index (capped at last value).
    A successful connection resets the index to 0.

Offline mode:
    - No reconnect on EOF; loops if loop_enabled, otherwise emits "finished"
    - Supports pause/seek/speed controls via thread-safe flags set by the UI

Frame dropping:
    - Detection: processed at most detection_fps times per second
    - Preview:   frame_ready emitted at most preview_fps times per second
    - Uses time.monotonic() — not frame counters — for accurate rate control
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from core.motion_engine import EngineConfig, MotionEngine
from core.mask_engine import build_zone_manager

log = logging.getLogger(__name__)

_RECONNECT_BACKOFF = [2, 5, 15, 30, 60]


# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    source_id: str
    source_type: str         # "recorder_channel" | "camera" | "offline"
    source_name: str
    rtsp_url: str            # full URL for RTSP; file path for offline
    detection_fps: float = 10.0
    preview_fps: float = 15.0
    loop_offline: bool = False
    is_offline: bool = False


# ---------------------------------------------------------------------------
# SourceWorker
# ---------------------------------------------------------------------------

class SourceWorker(QThread):
    # --- Qt Signals ---
    frame_ready      = Signal(str, QImage)    # source_id, preview frame
    motion_detected  = Signal(str, object)    # source_id, np.ndarray snapshot frame
    motion_ended     = Signal(str, float)     # source_id, duration_seconds
    status_changed   = Signal(str, str)       # source_id, "connected"|"reconnecting"|"offline"|"finished"
    error_occurred   = Signal(str, str)       # source_id, message
    stats_updated    = Signal(str, dict)      # source_id, {fps, det_fps, confirmed, candidates, warmup}

    def __init__(
        self,
        source_config: SourceConfig,
        engine_config: EngineConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sc = source_config
        self._ec = engine_config

        # Thread-safe control flags
        self._stop_flag = threading.Event()
        self._paused = threading.Event()
        self._config_lock = threading.Lock()

        # Pending config/zone updates — applied at frame boundary
        self._pending_engine_config: EngineConfig | None = None
        self._pending_zones: list[dict] | None = None

        # Offline playback controls
        self._seek_target: int | None = None
        self._speed_multiplier: float = 1.0
        self._seek_lock = threading.Lock()

        # Motion state tracking
        self._motion_active = False
        self._motion_start_time: float = 0.0
        self._last_snapshot_frame: np.ndarray | None = None

        self.setObjectName(f"worker-{source_config.source_id[:8]}")

    # ------------------------------------------------------------------
    # Public thread-safe API (called from UI thread)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop_flag.set()
        self._paused.clear()  # unblock if paused

    def pause(self) -> None:
        """Pause offline playback."""
        self._paused.set()

    def resume(self) -> None:
        """Resume offline playback."""
        self._paused.clear()

    def seek(self, frame_idx: int) -> None:
        """Request a seek to frame_idx (offline mode only)."""
        with self._seek_lock:
            self._seek_target = frame_idx

    def set_speed(self, multiplier: float) -> None:
        """Set playback speed multiplier (0.5, 1.0, 2.0 etc.)."""
        self._speed_multiplier = max(0.1, min(8.0, multiplier))

    def update_engine_config(self, config: EngineConfig) -> None:
        """Hot-update detection parameters (applied on next frame)."""
        with self._config_lock:
            self._pending_engine_config = config

    def update_zones(self, zones: list[dict]) -> None:
        """Hot-update exclusion zones (applied on next frame)."""
        with self._config_lock:
            self._pending_zones = zones

    # ------------------------------------------------------------------
    # QThread.run()
    # ------------------------------------------------------------------

    def run(self) -> None:
        log.info("Worker started: %s [%s]", self._sc.source_name, self._sc.source_type)
        try:
            engine = MotionEngine(self._ec)
        except Exception as exc:
            log.error("MotionEngine init failed: %s", exc)
            self.error_occurred.emit(self._sc.source_id, str(exc))
            return

        if self._sc.is_offline:
            self._run_offline(engine)
        else:
            self._run_rtsp(engine)

        log.info("Worker stopped: %s", self._sc.source_name)

    # ------------------------------------------------------------------
    # RTSP loop with reconnect
    # ------------------------------------------------------------------

    def _run_rtsp(self, engine: MotionEngine) -> None:
        backoff_idx = 0

        while not self._stop_flag.is_set():
            cap = self._open_capture(self._sc.rtsp_url)

            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                self.status_changed.emit(self._sc.source_id, "reconnecting")
                delay = _RECONNECT_BACKOFF[min(backoff_idx, len(_RECONNECT_BACKOFF) - 1)]
                log.warning(
                    "Could not open %s — retry in %ds",
                    self._sc.source_name, delay,
                )
                if self._interruptible_sleep(delay):
                    break
                backoff_idx += 1
                continue

            backoff_idx = 0
            self.status_changed.emit(self._sc.source_id, "connected")
            log.info("Connected: %s", self._sc.source_name)
            engine.reset()

            self._capture_loop(cap, engine)
            cap.release()

            if self._stop_flag.is_set():
                break

            # Connection dropped — attempt reconnect
            self._end_motion_if_active(engine)
            self.status_changed.emit(self._sc.source_id, "reconnecting")
            delay = _RECONNECT_BACKOFF[min(backoff_idx, len(_RECONNECT_BACKOFF) - 1)]
            log.warning("Stream dropped for %s — retry in %ds", self._sc.source_name, delay)
            if self._interruptible_sleep(delay):
                break
            backoff_idx += 1

        self._end_motion_if_active(engine)
        self.status_changed.emit(self._sc.source_id, "offline")

    # ------------------------------------------------------------------
    # Offline file loop
    # ------------------------------------------------------------------

    def _run_offline(self, engine: MotionEngine) -> None:
        path = self._sc.rtsp_url
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            msg = f"Cannot open file: {path}"
            log.error(msg)
            self.error_occurred.emit(self._sc.source_id, msg)
            self.status_changed.emit(self._sc.source_id, "offline")
            return

        self.status_changed.emit(self._sc.source_id, "connected")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        engine.reset()

        while not self._stop_flag.is_set():
            # Handle pause
            while self._paused.is_set() and not self._stop_flag.is_set():
                time.sleep(0.05)

            # Handle seek
            with self._seek_lock:
                if self._seek_target is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_target)
                    self._seek_target = None
                    engine.reset()

            ok, frame = cap.read()
            if not ok:
                if self._sc.loop_offline:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    engine.reset()
                    continue
                else:
                    self._end_motion_if_active(engine)
                    self.status_changed.emit(self._sc.source_id, "finished")
                    break

            self._process_frame(frame, engine, is_offline=True)

            # Throttle to source FPS adjusted by speed multiplier
            inter_frame_delay = (1.0 / source_fps) / self._speed_multiplier
            time.sleep(max(0.0, inter_frame_delay - 0.005))

        cap.release()

    # ------------------------------------------------------------------
    # Core per-frame capture loop (RTSP)
    # ------------------------------------------------------------------

    def _capture_loop(self, cap: cv2.VideoCapture, engine: MotionEngine) -> None:
        last_detect_t = 0.0
        last_preview_t = 0.0
        det_interval = 1.0 / self._sc.detection_fps
        preview_interval = 1.0 / self._sc.preview_fps

        while not self._stop_flag.is_set():
            # Apply pending config changes
            self._apply_pending(engine, cap)

            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            do_detect = (now - last_detect_t >= det_interval)
            do_preview = (now - last_preview_t >= preview_interval)

            if not do_detect and not do_preview:
                continue  # drop frame

            if do_detect:
                last_detect_t = now
                last_preview_t = now  # detection emits its own annotated preview
                self._process_frame(frame, engine, is_offline=False)
            else:
                # Between detection cycles: emit the raw frame so the display
                # stays in sync with real-time without running the full pipeline
                last_preview_t = now
                qimage = self._bgr_to_qimage(frame)
                if qimage is not None:
                    self.frame_ready.emit(self._sc.source_id, qimage)

    # ------------------------------------------------------------------
    # Single-frame processing (shared by RTSP and offline)
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        engine: MotionEngine,
        is_offline: bool,
    ) -> None:
        # Apply any pending config or zone updates
        self._apply_pending(engine, None)

        t0 = time.monotonic()
        try:
            result = engine.process_frame(frame)
        except Exception as exc:
            log.error("Engine error on %s: %s", self._sc.source_name, exc)
            return

        det_ms = (time.monotonic() - t0) * 1000

        # --- Motion state machine ---
        if result.motion_detected and not self._motion_active:
            self._motion_active = True
            self._motion_start_time = time.monotonic()
            self._last_snapshot_frame = frame.copy()
            self.motion_detected.emit(self._sc.source_id, frame.copy())

        elif not result.motion_detected and self._motion_active:
            duration = time.monotonic() - self._motion_start_time
            self._motion_active = False
            self.motion_ended.emit(self._sc.source_id, duration)

        # --- Emit preview frame ---
        qimage = self._bgr_to_qimage(result.annotated_frame)
        if qimage is not None:
            self.frame_ready.emit(self._sc.source_id, qimage)

        # --- Stats ---
        self.stats_updated.emit(
            self._sc.source_id,
            {
                "confirmed": result.confirmed_count,
                "candidates": result.candidate_count,
                "warmup_remaining": engine.get_warmup_remaining(),
                "det_ms": round(det_ms, 1),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_pending(self, engine: MotionEngine, cap) -> None:
        """Apply any pending config/zone updates from the UI thread."""
        with self._config_lock:
            if self._pending_engine_config is not None:
                engine.update_config(self._pending_engine_config)
                self._sc.detection_fps = self._pending_engine_config.history  # no-op; kept for clarity
                self._pending_engine_config = None

            if self._pending_zones is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else 0
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 0
                if w > 0 and h > 0:
                    zone_mgr = build_zone_manager(self._pending_zones, w, h)
                    engine.set_zones(zone_mgr)
                self._pending_zones = None

    def _end_motion_if_active(self, engine: MotionEngine) -> None:
        if self._motion_active:
            duration = time.monotonic() - self._motion_start_time
            self._motion_active = False
            self.motion_ended.emit(self._sc.source_id, duration)

    def _open_capture(self, url: str) -> cv2.VideoCapture | None:
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            return cap
        except Exception as exc:
            log.error("VideoCapture open error: %s", exc)
            return None

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep in 0.2s chunks, checking stop_flag. Returns True if stopped."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stop_flag.is_set():
                return True
            time.sleep(0.2)
        return False

    @staticmethod
    def _bgr_to_qimage(bgr: np.ndarray) -> QImage | None:
        try:
            if bgr is None or bgr.size == 0:
                return None
            h, w, ch = bgr.shape
            if ch == 3:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Use rgb.data (memoryview) instead of rgb.tobytes() to avoid
                # an extra full-frame allocation; .copy() detaches from the buffer
                qimg = QImage(
                    rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888
                )
                return qimg.copy()
            return None
        except Exception:
            return None
