"""
motion_engine.py — Thin wrapper over ti_motion_detect_v4 algorithm classes.

Design principle: this file wraps the existing algorithm with zero modifications
to the v4 source.  Drop in a newer version of ti_motion_detect_v4.py and
update_config() will apply the new parameters on the next reset().

Supported detection modes: "mog2" | "diff"
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Prevent ti_motion_detect_v4 from clobbering Qt platform to "offscreen".
# The v4 script sets QT_QPA_PLATFORM=offscreen when DISPLAY is absent and the
# var is not already set.  We pre-populate it so v4's guard becomes a no-op.
# ---------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

# ---------------------------------------------------------------------------
# Resolve path to the v4 script (dev vs frozen exe)
# ---------------------------------------------------------------------------
def _v4_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "motion_scripts"
    # dev layout: MotionGuard/app/core/motion_engine.py
    #             → MotionGuard → Human-Movement-Detection → motion_scripts
    return Path(__file__).resolve().parent.parent.parent.parent / "motion_scripts"


_v4_path = _v4_dir()
if str(_v4_path) not in sys.path:
    sys.path.insert(0, str(_v4_path))

try:
    from ti_motion_detect_v4 import (  # type: ignore
        TIPreprocessor,
        MotionDetector,
        MaskPostprocessor,
        MotionVisualizer,
        ExclusionZoneManager,
        CameraMotionSensor,
    )
    _V4_AVAILABLE = True
except ImportError as _e:
    _V4_AVAILABLE = False
    _V4_IMPORT_ERROR = str(_e)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    method: str = "mog2"           # "mog2" | "diff"
    history: int = 200
    threshold: float = 15.0
    diff_frames: int = 5
    use_clahe: bool = False
    clahe_clip: float = 2.5
    clahe_tile: int = 12
    use_temporal: bool = True
    alpha: float = 0.4
    morph_close: int = 5
    morph_open: int = 3
    min_area: int = 10
    max_area: int = 5000
    min_solidity: float = 0.3
    min_density: float = 0.0
    persistence: int = 4
    min_displacement: float = 15.0
    spatial_tol: float = 20.0
    max_absent: int = 5
    bbox_smooth_alpha: float = 0.35
    vel_smooth_alpha: float = 0.5
    # PTZ motion compensation
    ptz_aware: bool = True
    ptz_motion_thresh: float = 5.0
    ptz_settle_frames: int = 125
    # Display options
    show_candidates: bool = False   # --no-candidates: hide unconfirmed tracks

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        """Create from a string→string dict (e.g. from DB settings)."""
        def _bool(key, default="1"):
            return d.get(key, default) not in ("0", "false", "False")
        return cls(
            method=d.get("detection_mode", "mog2"),
            history=int(d.get("history", 200)),
            threshold=float(d.get("threshold", 15.0)),
            diff_frames=int(d.get("diff_frames", 5)),
            use_clahe=_bool("use_clahe", "0"),
            clahe_clip=float(d.get("clahe_clip", 2.5)),
            clahe_tile=int(d.get("clahe_tile", 12)),
            use_temporal=_bool("use_temporal", "1"),
            alpha=float(d.get("alpha", 0.4)),
            morph_close=int(d.get("morph_close", 5)),
            morph_open=int(d.get("morph_open", 3)),
            min_area=int(d.get("min_area", 10)),
            max_area=int(d.get("max_area", 5000)),
            min_solidity=float(d.get("min_solidity", 0.3)),
            min_density=float(d.get("min_density", 0.0)),
            persistence=int(d.get("persistence", 4)),
            min_displacement=float(d.get("min_displacement", 15.0)),
            spatial_tol=float(d.get("spatial_tol", 20.0)),
            max_absent=int(d.get("max_absent", 5)),
            bbox_smooth_alpha=float(d.get("bbox_smooth_alpha", 0.35)),
            vel_smooth_alpha=float(d.get("vel_smooth_alpha", 0.5)),
            ptz_aware=_bool("ptz_aware", "1"),
            ptz_motion_thresh=float(d.get("ptz_motion_thresh", 5.0)),
            ptz_settle_frames=int(d.get("ptz_settle_frames", 125)),
            show_candidates=_bool("show_candidates", "0"),
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    annotated_frame: np.ndarray   # BGR, safe to display
    motion_detected: bool
    confirmed_count: int
    candidate_count: int
    raw_mask: np.ndarray          # binary motion mask (before zone exclusion)
    clean_mask: np.ndarray        # mask after zone exclusion applied
    frame_idx: int = 0


# ---------------------------------------------------------------------------
# MotionEngine
# ---------------------------------------------------------------------------

class MotionEngine:
    """
    Wraps the v4 pipeline: Preprocess → Detect → Morphology → Zones → Track → Visualize.

    Thread-safety: this object must be owned and used from a single thread
    (typically a SourceWorker QThread). update_config() and set_zones() are
    called from that same thread via the worker's internal queue.
    """

    def __init__(self, config: EngineConfig) -> None:
        if not _V4_AVAILABLE:
            raise RuntimeError(
                f"ti_motion_detect_v4 could not be imported: {_V4_IMPORT_ERROR}\n"
                f"Searched in: {_v4_path}"
            )
        self._config = config
        self._frame_idx = 0
        self._warmup = config.history // 2
        self._zone_mgr: ExclusionZoneManager | None = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        c = self._config
        self._preprocessor = TIPreprocessor(
            use_clahe=c.use_clahe,
            clahe_clip=c.clahe_clip,
            clahe_tile=c.clahe_tile,
            use_temporal=c.use_temporal,
            alpha=c.alpha,
        )
        self._detector = MotionDetector(
            method=c.method,
            history=c.history,
            threshold=c.threshold,
            diff_frames=c.diff_frames,
        )
        self._postprocessor = MaskPostprocessor(
            morph_close=c.morph_close,
            morph_open=c.morph_open,
            min_area=c.min_area,
            max_area=c.max_area,
            min_solidity=c.min_solidity,
            persistence=c.persistence,
            spatial_tol=c.spatial_tol,
            min_displacement=c.min_displacement,
            min_density=c.min_density,
            max_absent=c.max_absent,
            bbox_smooth_alpha=c.bbox_smooth_alpha,
            vel_smooth_alpha=c.vel_smooth_alpha,
        )
        self._visualizer = MotionVisualizer(persistence=c.persistence)
        self._warmup = c.history // 2
        # PTZ motion compensation sensor (gates BG learning during camera movement)
        if c.ptz_aware:
            self._ptz_sensor = CameraMotionSensor(
                motion_thresh=c.ptz_motion_thresh,
                settle_frames=c.ptz_settle_frames,
            )
        else:
            self._ptz_sensor = None
        log.debug(
            "MotionEngine built  method=%s  warmup=%d  ptz=%s",
            c.method, self._warmup, c.ptz_aware,
        )

    # ------------------------------------------------------------------
    def process_frame(
        self,
        bgr_frame: np.ndarray,
        learning_rate: float = -1.0,
        show_overlay: bool = True,
    ) -> EngineResult:
        """
        Run the full detection pipeline on one BGR frame.

        Parameters
        ----------
        bgr_frame     : H×W×3 uint8 array
        learning_rate : -1 = adaptive (default); 0 = frozen; 0..1 explicit
        show_overlay  : if False, return the raw frame without boxes/HUD
        """
        self._frame_idx += 1
        warming_up = self._frame_idx <= self._warmup

        # 1. Preprocess
        gray = self._preprocessor.process(bgr_frame)

        # 2. PTZ motion compensation — overrides learning_rate when camera moves
        ptz_state = ""
        ptz_motion = 0.0
        ptz_settle_pct = 0.0
        if self._ptz_sensor is not None:
            ptz_state = self._ptz_sensor.update(gray)
            learning_rate = self._ptz_sensor.learning_rate
            ptz_motion = self._ptz_sensor.last_motion
            ptz_settle_pct = self._ptz_sensor.settling_progress

        # 3. Background subtraction
        raw_mask = self._detector.apply(gray, learning_rate)

        # 4. Morphological cleanup — returns np.ndarray directly
        morph_mask = self._postprocessor.apply_morphology(raw_mask)

        # 5. Apply exclusion zones
        if self._zone_mgr is not None:
            clean_mask = self._zone_mgr.apply(morph_mask)
        else:
            clean_mask = morph_mask

        # 6. Extract blobs
        blobs = self._postprocessor.extract_blobs(clean_mask)

        # 7. Track blobs
        confirmed, candidates = self._postprocessor.update_tracker(blobs)

        # 8. Visualize
        display_candidates = candidates if self._config.show_candidates else []
        annotated = bgr_frame.copy()
        if show_overlay:
            annotated = self._visualizer.draw_boxes(annotated, confirmed, display_candidates)
            annotated = self._visualizer.draw_hud(
                annotated,
                frame_idx=self._frame_idx,
                fps=0,  # caller fills in actual FPS
                method=self._config.method.upper(),
                n_candidates=len(candidates),
                n_confirmed=len(confirmed),
                warming_up=warming_up,
                n_zones=len(self._zone_mgr.zones) if self._zone_mgr else 0,
                ptz_state=ptz_state,
                ptz_motion=ptz_motion,
                ptz_settle_pct=ptz_settle_pct,
            )
            # Draw zone overlay if zones present
            if self._zone_mgr is not None:
                annotated = self._zone_mgr.draw_overlay(annotated, alpha=0.3)

        return EngineResult(
            annotated_frame=annotated,
            motion_detected=len(confirmed) > 0,
            confirmed_count=len(confirmed),
            candidate_count=len(candidates),
            raw_mask=raw_mask,
            clean_mask=clean_mask,
            frame_idx=self._frame_idx,
        )

    # ------------------------------------------------------------------
    def set_zones(self, zone_mgr: "ExclusionZoneManager | None") -> None:
        """Replace the active exclusion zone manager (pass None to clear)."""
        self._zone_mgr = zone_mgr

    def reset(self) -> None:
        """Reset all stateful components (useful after a stream reconnect)."""
        self._frame_idx = 0
        self._preprocessor.reset()
        self._detector.reset()
        self._postprocessor.reset()
        if self._ptz_sensor is not None:
            self._ptz_sensor.reset()
        log.debug("MotionEngine reset")

    def update_config(self, config: EngineConfig) -> None:
        """Apply a new config — rebuilds all internal objects and resets state."""
        self._config = config
        self._build()
        self.reset()

    def get_warmup_remaining(self) -> int:
        """Frames left in the background-model warmup period (0 = ready)."""
        return max(0, self._warmup - self._frame_idx)

    @property
    def is_available(self) -> bool:
        return _V4_AVAILABLE

    @staticmethod
    def check_availability() -> tuple[bool, str]:
        """Return (available, error_message)."""
        if _V4_AVAILABLE:
            return True, ""
        return False, _V4_IMPORT_ERROR
