"""
mask_engine.py — Convert normalized zone vertices (0..1) to ExclusionZoneManager.

Vertices are stored normalized in the DB so they survive resolution changes.
This module rebuilds the pixel-coordinate mask on each worker startup.
"""

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _ensure_v4_import() -> None:
    """Ensure ti_motion_detect_v4 is importable (path already added by motion_engine)."""
    try:
        import ti_motion_detect_v4  # noqa: F401
    except ImportError:
        from utils.paths import motion_scripts_dir
        path = str(motion_scripts_dir())
        if path not in sys.path:
            sys.path.insert(0, path)


def normalized_to_pixel(
    vertices_norm: list[list[float]],
    width: int,
    height: int,
) -> list[list[int]]:
    """
    Convert normalized coordinates [x_norm, y_norm] → pixel coordinates [px, py].

    Parameters
    ----------
    vertices_norm : [[x_norm, y_norm], ...]  where x_norm, y_norm ∈ [0, 1]
    width, height : frame dimensions
    """
    return [
        [int(x * width), int(y * height)]
        for x, y in vertices_norm
    ]


def pixel_to_normalized(
    vertices_px: list[list[int]],
    width: int,
    height: int,
) -> list[list[float]]:
    """Convert pixel coordinates → normalized (for saving to DB)."""
    if width == 0 or height == 0:
        return vertices_px  # type: ignore
    return [
        [x / width, y / height]
        for x, y in vertices_px
    ]


def build_zone_manager(
    zones_db: list[dict],
    frame_width: int,
    frame_height: int,
) -> "ExclusionZoneManager | None":  # type: ignore
    """
    Build an ExclusionZoneManager from DB zone records.

    Parameters
    ----------
    zones_db : list of {"name": str, "vertices": [[x_norm, y_norm], ...]}
    frame_width, frame_height : current frame resolution

    Returns
    -------
    ExclusionZoneManager instance, or None if no zones or import fails
    """
    if not zones_db:
        return None

    _ensure_v4_import()

    try:
        from ti_motion_detect_v4 import ExclusionZoneManager  # type: ignore

        mgr = ExclusionZoneManager(frame_width, frame_height)
        mgr.zones = []
        for zone in zones_db:
            pixel_polygon = normalized_to_pixel(
                zone["vertices"], frame_width, frame_height
            )
            if len(pixel_polygon) >= 3:
                mgr.zones.append(
                    {"name": zone["name"], "polygon": pixel_polygon}
                )
        mgr._build_mask()
        log.debug(
            "Built zone manager: %d zone(s) at %dx%d",
            len(mgr.zones),
            frame_width,
            frame_height,
        )
        return mgr

    except Exception as exc:
        log.error("Failed to build zone manager: %s", exc)
        return None
