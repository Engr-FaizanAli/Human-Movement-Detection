#!/usr/bin/env python3
"""
TI Motion Detection v4
======================
Extends v3 with exclusion zones — the production-grade fix for wind, waving
vegetation, water surfaces, and any other known high-motion background region.

  What exclusion zones do
  -----------------------
  You draw polygons over known noisy areas (tree lines, bushes, water, flags).
  Any motion detected inside those polygons is zeroed out of the mask BEFORE
  blob extraction — so those areas never produce candidates or confirmed tracks.

  This is the same feature called "privacy zones" in FLIR Elara, "alarm
  exclusion masks" in Axis, and "intrusion exclusion" in Hikvision.

  How to use — two steps
  ----------------------
  STEP 1: Draw zones once per camera position
  -------------------------------------------
  python motion_scripts/ti_motion_detect_v4.py \\
      --input test_cases/site_A.mp4 --draw-zones

  An OpenCV window opens on the first frame of the video.
  Draw polygons over vegetation / water / noise areas.
  Zones are saved to:  test_cases/site_A_zones.json

  STEP 2: Run detection (zone file is auto-loaded every time)
  -----------------------------------------------------------
  python motion_scripts/ti_motion_detect_v4.py \\
      --input test_cases/site_A.mp4 --no-clahe --threshold 10 \\
      --min-area 20 --persistence 4 --min-displacement 10

  If site_A_zones.json exists next to site_A.mp4 it loads silently.
  Add --zone-overlay to see red zone outlines on every output frame.

  Zone editor keyboard controls
  ------------------------------
  LEFT CLICK   Add a vertex to the current polygon
  C            Close the polygon (needs 3+ points) and start the next one
  U            Undo / remove the last vertex
  D            Delete the last completed zone
  S            Save all zones and begin video processing
  Q            Exit editor without saving (continues without zones)

  Zone file format (JSON, hand-editable)
  ---------------------------------------
  {
    "resolution": [1920, 1080],
    "zones": [
      {"name": "zone_0", "polygon": [[120, 50], [800, 50], [800, 400], [120, 400]]},
      {"name": "zone_1", "polygon": [[1500, 200], [1900, 200], [1900, 600], [1500, 600]]}
    ]
  }
  Polygons use pixel coordinates at the ORIGINAL video resolution.

All v3 parameters are preserved (stabilisation, PTZ-aware gating, CLAHE,
IIR smoothing, bbox smoothing, velocity-predicted tracking).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Parse --draw-zones BEFORE importing cv2 so we can control QT_QPA_PLATFORM.
# If draw-zones is requested we must NOT set offscreen — we need a real window.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--draw-zones", action="store_true", default=False)
_pre_args, _ = _pre_parser.parse_known_args()

if not _pre_args.draw_zones:
    if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") is None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Data structures  (identical to v3)
# ---------------------------------------------------------------------------

@dataclass
class Blob:
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    solidity: float


@dataclass
class TrackedBlob:
    blob_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    initial_centroid: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    frames_seen: int = 1
    frames_absent: int = 0
    confirmed: bool = False
    smooth_bbox: Tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )
    velocity: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    @property
    def displacement(self) -> float:
        dx = self.centroid[0] - self.initial_centroid[0]
        dy = self.centroid[1] - self.initial_centroid[1]
        return (dx * dx + dy * dy) ** 0.5

    @property
    def speed(self) -> float:
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5

    def predicted_centroid(self, n_steps: int = 1) -> Tuple[float, float]:
        return (
            self.centroid[0] + self.velocity[0] * n_steps,
            self.centroid[1] + self.velocity[1] * n_steps,
        )

    def smooth_bbox_int(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.smooth_bbox[0])),
            int(round(self.smooth_bbox[1])),
            max(1, int(round(self.smooth_bbox[2]))),
            max(1, int(round(self.smooth_bbox[3]))),
        )


# ---------------------------------------------------------------------------
# NEW — Stage 0: Exclusion zone manager
# ---------------------------------------------------------------------------

class ExclusionZoneManager:
    """
    Manages named polygon exclusion zones.

    Workflow
    --------
    1. draw_interactive()  — operator draws zones on a video frame, zones saved
                             to JSON.
    2. load()              — future runs load the JSON silently.
    3. apply()             — called once per frame; zeros excluded regions out
                             of the cleaned motion mask before blob extraction.
    4. draw_overlay()      — optionally renders red zone outlines on the output
                             frame so the operator can see what is excluded.

    Mask convention: 255 = detect (active), 0 = excluded.
    """

    def __init__(self, width: int, height: int):
        self._w = width
        self._h = height
        self._zones: List[Dict] = []
        self._mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ I/O

    def load(self, path: str) -> int:
        """Load zones from a JSON file. Returns the number of zones loaded."""
        p = Path(path)
        if not p.is_file():
            return 0
        with open(p) as f:
            data = json.load(f)

        stored_res = data.get("resolution")
        if stored_res and stored_res != [self._w, self._h]:
            print(
                f"  [Zones] WARNING: zones were drawn at {stored_res[0]}×{stored_res[1]} "
                f"but video is {self._w}×{self._h}. Zones may be misaligned."
            )

        self._zones = data.get("zones", [])
        self._build_mask()
        return len(self._zones)

    def save(self, path: str) -> None:
        """Save zones to a JSON file."""
        with open(path, "w") as f:
            json.dump(
                {"resolution": [self._w, self._h], "zones": self._zones},
                f,
                indent=2,
            )

    # ------------------------------------------------------------------ mask

    def _build_mask(self) -> None:
        """Rebuild the binary mask from the current polygon list."""
        mask = np.ones((self._h, self._w), dtype=np.uint8) * 255
        for zone in self._zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)
        self._mask = mask

    def apply(self, motion_mask: np.ndarray) -> np.ndarray:
        """
        Zero out excluded regions from the motion mask.
        Returns the mask unchanged if no zones are defined.
        """
        if self._mask is None or not self._zones:
            return motion_mask
        return cv2.bitwise_and(motion_mask, self._mask)

    # ------------------------------------------------------------------ overlay

    def draw_overlay(self, frame: np.ndarray, alpha: float = 0.22) -> np.ndarray:
        """
        Draw exclusion zones as a semi-transparent red overlay on the frame.
        Adds a thin bright-red border around each zone polygon.
        """
        if not self._zones:
            return frame
        img = frame.copy()
        overlay = img.copy()
        for zone in self._zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 160))
        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        for zone in self._zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 1)
        return img

    # ------------------------------------------------------------------ properties

    @property
    def count(self) -> int:
        return len(self._zones)

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    # ------------------------------------------------------------------ interactive editor

    def draw_interactive(
        self,
        frame: np.ndarray,
        zone_file: str,
        window_name: str = "Zone Editor",
    ) -> bool:
        """
        Open the interactive polygon editor.

        Returns True  if the operator pressed S (zones saved to zone_file).
        Returns False if the operator pressed Q (no changes saved).
        """
        # State
        current_poly: List[List[int]] = []
        zones: List[Dict] = list(self._zones)     # pre-populate with any loaded zones
        zone_counter: int = len(zones)
        mouse_pos: List[int] = [0, 0]

        def mouse_cb(event, x, y, flags, param):
            mouse_pos[0], mouse_pos[1] = x, y
            if event == cv2.EVENT_LBUTTONDOWN:
                current_poly.append([x, y])

        h, w = frame.shape[:2]
        win_w = min(w, 1280)
        win_h = min(h, 800)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, win_w, win_h)
        cv2.setMouseCallback(window_name, mouse_cb)

        CONTROLS = [
            "LEFT CLICK : add vertex",
            "C          : close polygon (need 3+ pts)",
            "U          : undo last vertex",
            "D          : delete last zone",
            "S          : save & start processing",
            "Q          : exit without saving",
        ]
        CTRL_LINE_H = 22
        PANEL_H = CTRL_LINE_H * len(CONTROLS) + 10

        saved = False

        while True:
            display = frame.copy()

            # ---- draw completed zones ----
            if zones:
                overlay = display.copy()
                for z in zones:
                    pts = np.array(z["polygon"], dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], (0, 0, 180))
                display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
                for z in zones:
                    pts = np.array(z["polygon"], dtype=np.int32)
                    cv2.polylines(display, [pts], True, (0, 0, 255), 2)
                    # label at centroid
                    cx = int(np.mean([p[0] for p in z["polygon"]]))
                    cy = int(np.mean([p[1] for p in z["polygon"]]))
                    cv2.putText(
                        display, z["name"], (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1, cv2.LINE_AA,
                    )

            # ---- draw in-progress polygon ----
            if current_poly:
                for pt in current_poly:
                    cv2.circle(display, tuple(pt), 5, (0, 255, 255), -1)
                if len(current_poly) > 1:
                    pts_arr = np.array(current_poly, dtype=np.int32)
                    cv2.polylines(display, [pts_arr], False, (0, 255, 255), 2)
                # rubber-band line to cursor
                cv2.line(
                    display,
                    tuple(current_poly[-1]),
                    tuple(mouse_pos),
                    (0, 200, 200), 1,
                )

            # ---- instruction panel (semi-transparent — frame shows through) ----
            panel_overlay = display.copy()
            cv2.rectangle(panel_overlay, (0, h - PANEL_H), (w, h), (10, 10, 10), -1)
            display = cv2.addWeighted(panel_overlay, 0.45, display, 0.55, 0)
            for i, txt in enumerate(CONTROLS):
                cv2.putText(
                    display, txt,
                    (8, h - PANEL_H + CTRL_LINE_H + i * CTRL_LINE_H),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 255, 200), 1, cv2.LINE_AA,
                )

            # ---- status line ----
            status = (
                f"Zones: {len(zones)}   "
                f"Current polygon points: {len(current_poly)}"
            )
            cv2.putText(
                display, status, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 1, cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            # ---- key handling ----
            if key == ord("c"):
                if len(current_poly) >= 3:
                    name = f"zone_{zone_counter}"
                    zones.append({"name": name, "polygon": [list(p) for p in current_poly]})
                    zone_counter += 1
                    current_poly.clear()
                    print(f"  [Zones] Polygon closed → {name}  ({len(zones)} zone(s) total)")
                else:
                    print("  [Zones] Need at least 3 points to close a polygon")

            elif key == ord("u"):
                if current_poly:
                    current_poly.pop()

            elif key == ord("d"):
                if zones:
                    removed = zones.pop()
                    zone_counter = max(0, zone_counter - 1)
                    print(f"  [Zones] Deleted {removed['name']}")

            elif key == ord("s"):
                self._zones = zones
                self._build_mask()
                self.save(zone_file)
                print(f"  [Zones] Saved {len(zones)} zone(s) → {zone_file}")
                saved = True
                break

            elif key == ord("q"):
                print("  [Zones] Editor closed without saving")
                break

            # window closed via OS button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("  [Zones] Editor window closed without saving")
                break

        cv2.destroyWindow(window_name)
        return saved


# ---------------------------------------------------------------------------
# Stage 1: TI-specific preprocessing  (unchanged)
# ---------------------------------------------------------------------------

class TIPreprocessor:
    def __init__(
        self,
        use_clahe: bool = True,
        clahe_clip: float = 2.5,
        clahe_tile: int = 12,
        use_temporal: bool = True,
        alpha: float = 0.4,
    ):
        self._use_clahe = use_clahe
        self._use_temporal = use_temporal
        self._alpha = alpha
        self._smooth: Optional[np.ndarray] = None
        self._clahe_tile_base = clahe_tile
        self._clahe_clip = clahe_clip
        self._clahe: Optional[cv2.CLAHE] = None

    def _build_clahe(self, height: int, width: int) -> None:
        tile = min(self._clahe_tile_base, min(width, height) // 4)
        tile = max(tile, 1)
        self._clahe = cv2.createCLAHE(clipLimit=self._clahe_clip,
                                      tileGridSize=(tile, tile))

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        if self._use_clahe:
            if self._clahe is None:
                self._build_clahe(gray.shape[0], gray.shape[1])
            gray = self._clahe.apply(gray)
        if self._use_temporal:
            if self._smooth is None:
                self._smooth = gray.astype(np.float32)
            else:
                self._smooth = (self._alpha * gray.astype(np.float32)
                                + (1.0 - self._alpha) * self._smooth)
            gray = np.clip(self._smooth, 0, 255).astype(np.uint8)
        return gray

    def reset(self) -> None:
        self._smooth = None


# ---------------------------------------------------------------------------
# Stage 1b: Camera-shake stabilisation  (unchanged)
# ---------------------------------------------------------------------------

class CameraStabilizer:
    def __init__(
        self,
        smooth_radius: int = 30,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 30.0,
    ):
        self._alpha = 2.0 / (smooth_radius + 1)
        self._max_corners = max_corners
        self._quality = quality_level
        self._min_dist = min_distance
        self._prev_gray: Optional[np.ndarray] = None
        self._cum = np.zeros(3, dtype=np.float64)
        self._smooth = np.zeros(3, dtype=np.float64)

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        if self._prev_gray is None:
            self._prev_gray = gray
            return frame
        delta = self._estimate_transform(self._prev_gray, gray)
        self._prev_gray = gray
        self._cum    += delta
        self._smooth += self._alpha * (self._cum - self._smooth)
        corr = self._smooth - self._cum
        h, w = frame.shape[:2]
        M = _make_affine(corr[0], corr[1], corr[2], w / 2.0, h / 2.0)
        return cv2.warpAffine(frame, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def reset(self) -> None:
        self._prev_gray = None
        self._cum[:] = 0.0
        self._smooth[:] = 0.0

    def _estimate_transform(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self._max_corners,
                                      qualityLevel=self._quality,
                                      minDistance=self._min_dist, blockSize=7)
        if pts is None or len(pts) < 4:
            return np.zeros(3)
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
        if pts2 is None:
            return np.zeros(3)
        ok = status.ravel() == 1
        if ok.sum() < 4:
            return np.zeros(3)
        M, _ = cv2.estimateAffinePartial2D(pts[ok], pts2[ok],
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
        if M is None:
            return np.zeros(3)
        return np.array([M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])],
                        dtype=np.float64)


# ---------------------------------------------------------------------------
# Stage 1c: PTZ camera motion sensor  (unchanged)
# ---------------------------------------------------------------------------

class CameraMotionSensor:
    MOVING   = "MOVING"
    SETTLING = "SETTLING"
    ACTIVE   = "ACTIVE"
    _LR: Dict[str, float] = {MOVING: 0.0, SETTLING: 0.05, ACTIVE: -1.0}

    def __init__(
        self,
        motion_thresh: float = 5.0,
        settle_frames: int = 125,
        max_corners: int = 150,
        quality_level: float = 0.01,
        min_distance: float = 30.0,
    ):
        self._motion_thresh = motion_thresh
        self._settle_frames = settle_frames
        self._max_corners = max_corners
        self._quality = quality_level
        self._min_dist = min_distance
        self._prev_gray: Optional[np.ndarray] = None
        self._state: str = self.SETTLING
        self._settle_counter: int = 0
        self.last_motion: float = 0.0

    def update(self, gray: np.ndarray) -> str:
        if self._prev_gray is None:
            self._prev_gray = gray
            return self._state
        motion = self._estimate_global_motion(self._prev_gray, gray)
        self._prev_gray = gray
        self.last_motion = motion
        if motion > self._motion_thresh:
            if self._state != self.MOVING:
                self._state = self.MOVING
                self._settle_counter = 0
        else:
            if self._state == self.MOVING:
                self._state = self.SETTLING
                self._settle_counter = 0
            elif self._state == self.SETTLING:
                self._settle_counter += 1
                if self._settle_counter >= self._settle_frames:
                    self._state = self.ACTIVE
        return self._state

    @property
    def learning_rate(self) -> float:
        return self._LR[self._state]

    @property
    def settling_progress(self) -> float:
        if self._state == self.ACTIVE:
            return 1.0
        if self._state == self.MOVING:
            return 0.0
        return min(1.0, self._settle_counter / max(self._settle_frames, 1))

    def reset(self) -> None:
        self._prev_gray = None
        self._settle_counter = 0
        self._state = self.SETTLING

    def _estimate_global_motion(self, prev: np.ndarray, curr: np.ndarray) -> float:
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self._max_corners,
                                      qualityLevel=self._quality,
                                      minDistance=self._min_dist, blockSize=7)
        if pts is None or len(pts) < 4:
            return 0.0
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
        if pts2 is None:
            return 0.0
        ok = status.ravel() == 1
        if ok.sum() < 4:
            return 0.0
        M, _ = cv2.estimateAffinePartial2D(pts[ok], pts2[ok],
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
        if M is None:
            return 0.0
        dx, dy = float(M[0, 2]), float(M[1, 2])
        return (dx * dx + dy * dy) ** 0.5


# ---------------------------------------------------------------------------
# Shared geometry helper
# ---------------------------------------------------------------------------

def _make_affine(dx: float, dy: float, da: float,
                 cx: float, cy: float) -> np.ndarray:
    c, s = np.cos(da), np.sin(da)
    return np.array([
        [c, -s, (1 - c) * cx + s * cy + dx],
        [s,  c, (1 - c) * cy - s * cx + dy],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 2: Motion detection backends  (unchanged)
# ---------------------------------------------------------------------------

class MotionDetector:
    def __init__(
        self,
        method: str = "mog2",
        history: int = 200,
        threshold: float = 12.0,
        diff_frames: int = 5,
    ):
        self.method = method.lower()
        self._history = history
        self._threshold = threshold
        self._diff_frames = diff_frames
        self._mog2: Optional[cv2.BackgroundSubtractorMOG2] = None
        self._knn:  Optional[cv2.BackgroundSubtractorKNN]  = None
        self._frame_buffer: List[np.ndarray] = []
        self._build()

    def _build(self) -> None:
        if self.method == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self._history, varThreshold=self._threshold,
                detectShadows=False)
        elif self.method == "knn":
            self._knn = cv2.createBackgroundSubtractorKNN(
                history=self._history,
                dist2Threshold=self._threshold * self._threshold,
                detectShadows=False)

    def apply(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        if self.method == "mog2":
            return (self._mog2.apply(gray, learningRate=learning_rate) > 200).astype(np.uint8) * 255
        elif self.method == "knn":
            return (self._knn.apply(gray, learningRate=learning_rate) > 200).astype(np.uint8) * 255
        elif self.method == "diff":
            return self._apply_diff(gray)
        elif self.method == "farneback":
            return self._apply_farneback(gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._build()

    def _apply_diff(self, gray: np.ndarray) -> np.ndarray:
        self._frame_buffer.append(gray.copy())
        if len(self._frame_buffer) > self._diff_frames + 1:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) <= self._diff_frames:
            return np.zeros_like(gray)
        ref = self._frame_buffer[-(self._diff_frames + 1)]
        diff = cv2.absdiff(gray, ref)
        _, mask = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _apply_farneback(self, gray: np.ndarray) -> np.ndarray:
        self._frame_buffer.append(gray.copy())
        if len(self._frame_buffer) > 2:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) < 2:
            return np.zeros_like(gray)
        prev, curr = self._frame_buffer[0], self._frame_buffer[1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=7,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        _, mask = cv2.threshold(mag, max(self._threshold * 0.1, 0.3), 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3: Post-processing + tracker  (v3 bbox smoothing + velocity, unchanged)
# ---------------------------------------------------------------------------

class MaskPostprocessor:
    def __init__(
        self,
        morph_close: int = 5,
        morph_open: int = 3,
        min_area: int = 4,
        max_area: int = 5000,
        min_solidity: float = 0.3,
        persistence: int = 3,
        spatial_tol: int = 15,
        min_displacement: float = 0.0,
        min_density: float = 0.0,
        max_absent: int = 2,
        bbox_smooth_alpha: float = 0.35,
        vel_smooth_alpha: float = 0.5,
    ):
        self._min_area = min_area
        self._max_area = max_area
        self._min_solidity = min_solidity
        self._persistence = persistence
        self._spatial_tol = spatial_tol
        self._min_displacement = min_displacement
        self._min_density = min_density
        self._max_absent = max_absent
        self._bbox_alpha = bbox_smooth_alpha
        self._vel_alpha  = vel_smooth_alpha

        ks_c = max(morph_close, 1);  ks_o = max(morph_open, 1)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_c, ks_c))
        self._open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_o, ks_o))

        self._tracked: Dict[int, TrackedBlob] = {}
        self._next_id: int = 0

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel),
            cv2.MORPH_OPEN, self._open_kernel)

    def extract_blobs(self, mask: np.ndarray) -> List[Blob]:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        blobs: List[Blob] = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self._min_area <= area <= self._max_area):
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT]);   y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH]);   h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if self._min_density > 0.0 and w > 0 and h > 0:
                if area / (w * h) < self._min_density:
                    continue
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            if solidity < self._min_solidity:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            blobs.append(Blob(bbox=(x, y, w, h), centroid=(cx, cy),
                              area=area, solidity=solidity))
        return blobs

    def update_tracker(
        self, blobs: List[Blob]
    ) -> Tuple[List[TrackedBlob], List[TrackedBlob]]:
        matched: set = set()

        for blob in blobs:
            best_id: Optional[int] = None
            best_dist: float = self._spatial_tol

            for tid, track in self._tracked.items():
                n_steps = track.frames_absent + 1
                pred_cx, pred_cy = track.predicted_centroid(n_steps)
                dx = blob.centroid[0] - pred_cx
                dy = blob.centroid[1] - pred_cy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist;  best_id = tid

            if best_id is not None:
                t = self._tracked[best_id]
                raw_vx = blob.centroid[0] - t.centroid[0]
                raw_vy = blob.centroid[1] - t.centroid[1]
                va = self._vel_alpha
                t.velocity = (va * raw_vx + (1 - va) * t.velocity[0],
                              va * raw_vy + (1 - va) * t.velocity[1])
                ba = self._bbox_alpha
                t.smooth_bbox = (
                    ba * blob.bbox[0] + (1 - ba) * t.smooth_bbox[0],
                    ba * blob.bbox[1] + (1 - ba) * t.smooth_bbox[1],
                    ba * blob.bbox[2] + (1 - ba) * t.smooth_bbox[2],
                    ba * blob.bbox[3] + (1 - ba) * t.smooth_bbox[3],
                )
                t.centroid = blob.centroid;  t.bbox = blob.bbox
                t.frames_seen += 1;          t.frames_absent = 0
                t.confirmed = (t.frames_seen >= self._persistence
                               and t.displacement >= self._min_displacement)
                matched.add(best_id)
            else:
                nid = self._next_id;  self._next_id += 1
                self._tracked[nid] = TrackedBlob(
                    blob_id=nid, centroid=blob.centroid, bbox=blob.bbox,
                    initial_centroid=blob.centroid,
                    smooth_bbox=(float(blob.bbox[0]), float(blob.bbox[1]),
                                 float(blob.bbox[2]), float(blob.bbox[3])),
                    velocity=(0.0, 0.0),
                )

        stale = [tid for tid, t in self._tracked.items()
                 if tid not in matched and t.frames_absent + 1 > self._max_absent]
        for tid in self._tracked:
            if tid not in matched:
                self._tracked[tid].frames_absent += 1
        for tid in stale:
            del self._tracked[tid]

        confirmed  = [t for t in self._tracked.values() if t.confirmed]
        candidates = [t for t in self._tracked.values() if not t.confirmed]
        return confirmed, candidates

    def reset(self) -> None:
        self._tracked.clear();  self._next_id = 0


# ---------------------------------------------------------------------------
# Visualisation  (unchanged from v3)
# ---------------------------------------------------------------------------

class MotionVisualizer:
    COLOR_CONFIRMED = (0, 230,   0)
    COLOR_CANDIDATE = (0, 220, 255)

    def __init__(self, persistence: int = 3):
        self._persistence = persistence

    def draw_boxes(
        self,
        frame: np.ndarray,
        confirmed: List[TrackedBlob],
        candidates: List[TrackedBlob],
    ) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        lw = max(1, int(min(h, w) * 0.002))
        fs = max(0.35, min(h, w) / 1800)
        ft = max(1, lw // 2)

        def _draw(track: TrackedBlob, color: tuple, label: str) -> None:
            bx, by, bw, bh = track.smooth_bbox_int()
            bx = max(0, min(bx, w - 1));  by = max(0, min(by, h - 1))
            bw = max(1, min(bw, w - bx)); bh = max(1, min(bh, h - by))
            if bw * bh < 16:
                pad = 4
                bx = max(0, bx - pad);  by = max(0, by - pad)
                bw = min(w - bx, bw + pad * 2);  bh = min(h - by, bh + pad * 2)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, lw)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            ly = max(by - th - 6, 0)
            cv2.rectangle(img, (bx, ly), (bx + tw + 4, ly + th + 6), color, -1)
            cv2.putText(img, label, (bx + 2, ly + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft)

        for t in candidates:
            _draw(t, self.COLOR_CANDIDATE, f"?{t.frames_seen}/{self._persistence}")
        for t in confirmed:
            _draw(t, self.COLOR_CONFIRMED,
                  f"#{t.blob_id} T:{t.frames_seen} D:{t.displacement:.0f}px V:{t.speed:.1f}px/f")
        return img

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        method: str,
        n_candidates: int,
        n_confirmed: int,
        warming_up: bool,
        n_zones: int = 0,
        ptz_state: Optional[str] = None,
        ptz_motion: float = 0.0,
        ptz_settle_pct: float = 0.0,
    ) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = max(0.4, min(h, w) / 1600)
        fthick = max(1, int(fscale * 1.5))
        line_h = int(fscale * 28) + 6

        if ptz_state == CameraMotionSensor.MOVING:
            lines  = [f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}  Method: {method}",
                      f"PTZ MOVING  {ptz_motion:.1f} px/f  — detections paused"]
            colors = [(220, 220, 220), (0, 60, 255)]
        elif ptz_state == CameraMotionSensor.SETTLING:
            bw    = max(1, int(ptz_settle_pct * 20))
            bar   = "[" + "#" * bw + "-" * (20 - bw) + "]"
            lines  = [f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}  Method: {method}",
                      f"SETTLING BG  {bar}  {ptz_settle_pct*100:.0f}%"]
            colors = [(220, 220, 220), (0, 165, 255)]
        elif warming_up:
            lines  = [f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}",
                      f"Method: {method}", "*** WARMING UP ***"]
            colors = [(220, 220, 220), (220, 220, 220), (0, 60, 255)]
        else:
            zone_str = f"  Zones: {n_zones}" if n_zones > 0 else ""
            lines  = [f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}",
                      f"Method: {method}   Conf: {n_confirmed}   Cand: {n_candidates}{zone_str}"]
            colors = [(220, 220, 220), (220, 220, 220)]

        max_tw = max(cv2.getTextSize(l, font, fscale, fthick)[0][0] for l in lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (max_tw + 12, len(lines) * line_h + 6), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        for i, (line, color) in enumerate(zip(lines, colors)):
            cv2.putText(img, line, (6, (i + 1) * line_h),
                        font, fscale, color, fthick, cv2.LINE_AA)
        return img

    @staticmethod
    def build_debug_panel(preprocessed: np.ndarray,
                          mask: np.ndarray,
                          annotated: np.ndarray) -> np.ndarray:
        h, w = annotated.shape[:2]
        def _bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        panel_pre  = cv2.resize(_bgr(preprocessed), (w, h))
        panel_mask = cv2.resize(_bgr(mask),          (w, h))
        for panel, text in [(panel_pre, "PREPROCESSED"),
                            (panel_mask, "MOTION MASK (post-zone)"),
                            (annotated, "DETECTIONS")]:
            cv2.putText(panel, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return np.hstack([panel_pre, panel_mask, annotated])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TI Motion Detection v4 — exclusion zones + v3 tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument("--input",   "-i", required=True)
    io.add_argument("--output",  "-o", default=None,
                    help="Output path (auto-named <input>_v4.mp4 if omitted)")
    io.add_argument("--display",       action="store_true")
    io.add_argument("--debug",         action="store_true",
                    help="3-panel debug view: preprocessed | mask | detections")
    io.add_argument("--scale",         type=float, default=1.0)

    # Detection method
    det = parser.add_argument_group("Detection method")
    det.add_argument("--method",      default="mog2",
                     choices=["mog2", "knn", "diff", "farneback"])
    det.add_argument("--history",     type=int,   default=200)
    det.add_argument("--threshold",   type=float, default=12.0)
    det.add_argument("--diff-frames", type=int,   default=5)
    det.add_argument("--freeze-after-warmup", action="store_true", default=False)

    # TI preprocessing
    pre = parser.add_argument_group("TI preprocessing")
    pre.add_argument("--clahe",              dest="clahe", action="store_true",  default=True)
    pre.add_argument("--no-clahe",           dest="clahe", action="store_false")
    pre.add_argument("--clahe-clip",         type=float, default=2.5)
    pre.add_argument("--clahe-tile",         type=int,   default=12)
    pre.add_argument("--temporal-smooth",    dest="temporal_smooth", action="store_true",  default=True)
    pre.add_argument("--no-temporal-smooth", dest="temporal_smooth", action="store_false")
    pre.add_argument("--alpha",              type=float, default=0.4)

    # Post-processing
    post = parser.add_argument_group("Post-processing")
    post.add_argument("--morph-close",      type=int,   default=5)
    post.add_argument("--morph-open",       type=int,   default=3)
    post.add_argument("--min-area",         type=int,   default=4)
    post.add_argument("--max-area",         type=int,   default=5000)
    post.add_argument("--min-solidity",     type=float, default=0.3)
    post.add_argument("--persistence",      type=int,   default=3)
    post.add_argument("--min-displacement", type=float, default=0.0,
                      help="Min px a blob must travel from birth to be CONFIRMED")
    post.add_argument("--min-density",      type=float, default=0.0)
    post.add_argument("--spatial-tol",      type=int,   default=15,
                      help="Max px to predicted centroid for track match")
    post.add_argument("--no-candidates",    dest="show_candidates", action="store_false",
                      default=True)
    post.add_argument("--max-absent",       type=int,   default=2)

    # v3 tracking
    trk = parser.add_argument_group("v3 tracking  (alpha=1.0 disables each)")
    trk.add_argument("--bbox-smooth-alpha", type=float, default=0.35,
                     help="EMA weight for bbox smoothing (0.35=default, 1.0=raw)")
    trk.add_argument("--vel-smooth-alpha",  type=float, default=0.5,
                     help="EMA weight for velocity estimation")

    # v4 exclusion zones
    zones_grp = parser.add_argument_group(
        "v4 exclusion zones  — draw once, auto-loaded every run")
    zones_grp.add_argument(
        "--draw-zones", action="store_true", default=False,
        help="Open interactive zone editor before processing. "
             "Draw polygons over vegetation / water / noise areas. "
             "Zones saved to --zone-file (auto-named if omitted).")
    zones_grp.add_argument(
        "--zone-file", default=None,
        help="Path to zones JSON file. "
             "Default: <input_stem>_zones.json next to the input video. "
             "Auto-loaded if it exists; auto-created by --draw-zones.")
    zones_grp.add_argument(
        "--zone-frame", type=int, default=0,
        help="Video frame index to display in the zone editor (default: 0). "
             "Use a later frame if the first frame does not show all vegetation.")
    zones_grp.add_argument(
        "--zone-overlay", action="store_true", default=False,
        help="Draw semi-transparent red exclusion zones on every output frame. "
             "Useful for verifying zone placement in the output video.")
    zones_grp.add_argument(
        "--no-zone-overlay", dest="zone_overlay", action="store_false")

    # Camera stabilisation
    stab = parser.add_argument_group("Camera stabilisation  (--stabilize)")
    stab.add_argument("--stabilize",        action="store_true", default=False)
    stab.add_argument("--stabilize-radius", type=int, default=30)

    # PTZ-aware gating
    ptz = parser.add_argument_group("PTZ-aware background gating  (--ptz-aware)")
    ptz.add_argument("--ptz-aware",         action="store_true", default=False)
    ptz.add_argument("--ptz-motion-thresh", type=float, default=5.0)
    ptz.add_argument("--ptz-settle-time",   type=float, default=5.0)

    args = parser.parse_args()

    # ------------------------------------------------------------------ paths
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}"); sys.exit(1)
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[ERROR] Unsupported extension: {input_path.suffix}"); sys.exit(1)

    out_path  = Path(args.output) if args.output else \
                input_path.parent / f"{input_path.stem}_v4.mp4"
    zone_file = args.zone_file if args.zone_file else \
                str(input_path.parent / f"{input_path.stem}_zones.json")

    # ------------------------------------------------------------------ video open
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}"); sys.exit(1)

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warmup_frames = args.history // 2
    settle_frames = max(1, int(args.ptz_settle_time * fps_in))

    # ------------------------------------------------------------------ exclusion zones
    zone_mgr = ExclusionZoneManager(width, height)

    if args.draw_zones:
        # Seek to the requested frame for the editor background
        target_frame = min(args.zone_frame, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, editor_frame = cap.read()
        if not ret:
            print("[WARN] Could not read zone editor frame — using blank canvas")
            editor_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

        # Load any existing zones first so the editor shows them pre-populated
        existing = zone_mgr.load(zone_file)
        if existing:
            print(f"  [Zones] Loaded {existing} existing zone(s) from {zone_file} — shown in editor")

        print("\n  Zone editor open. Draw exclusion polygons over vegetation / noise areas.")
        print(f"  Zones will be saved to: {zone_file}\n")
        zone_mgr.draw_interactive(editor_frame, zone_file)
        cap.release()
        sys.exit(0)

    else:
        # Auto-load zone file if it exists
        n_loaded = zone_mgr.load(zone_file)
        if n_loaded:
            print(f"  [Zones] Auto-loaded {n_loaded} zone(s) from {zone_file}")

    # ------------------------------------------------------------------ print config
    print(f"\nInput        : {input_path}")
    print(f"Output       : {out_path}")
    print(f"Frames       : {total_frames}  |  FPS: {fps_in:.1f}  |  {width}×{height}")
    print(f"Method       : {args.method.upper()}")
    print(f"Stabilise    : {'ON  (radius=' + str(args.stabilize_radius) + ')' if args.stabilize else 'OFF'}")
    print(f"PTZ-aware    : {'ON  (thresh=' + str(args.ptz_motion_thresh) + ' px/f)' if args.ptz_aware else 'OFF'}")
    print(f"Warmup       : {warmup_frames} frames  ({warmup_frames / fps_in:.1f}s)")
    print(f"Persistence  : {args.persistence}   min_disp: {args.min_displacement}px")
    print(f"BBox alpha   : {args.bbox_smooth_alpha}   Vel alpha: {args.vel_smooth_alpha}")
    print(f"Excl zones   : {zone_mgr.count}  ({'file: ' + zone_file if zone_mgr.count else 'none loaded'})")
    print(f"Zone overlay : {'ON' if args.zone_overlay and zone_mgr.count else 'OFF'}")
    print()

    # ------------------------------------------------------------------ pipeline objects
    preprocessor = TIPreprocessor(
        use_clahe=args.clahe, clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
        use_temporal=args.temporal_smooth, alpha=args.alpha)
    detector = MotionDetector(
        method=args.method, history=args.history,
        threshold=args.threshold, diff_frames=args.diff_frames)
    postproc = MaskPostprocessor(
        morph_close=args.morph_close, morph_open=args.morph_open,
        min_area=args.min_area, max_area=args.max_area,
        min_solidity=args.min_solidity, persistence=args.persistence,
        spatial_tol=args.spatial_tol, min_displacement=args.min_displacement,
        min_density=args.min_density, max_absent=args.max_absent,
        bbox_smooth_alpha=args.bbox_smooth_alpha,
        vel_smooth_alpha=args.vel_smooth_alpha)
    vis = MotionVisualizer(persistence=args.persistence)

    stabilizer: Optional[CameraStabilizer]   = (
        CameraStabilizer(smooth_radius=args.stabilize_radius) if args.stabilize else None)
    ptz_sensor: Optional[CameraMotionSensor] = (
        CameraMotionSensor(motion_thresh=args.ptz_motion_thresh, settle_frames=settle_frames)
        if args.ptz_aware else None)

    # ------------------------------------------------------------------ video writer
    out_w    = width * 3 if args.debug else width
    scaled_w = max(1, int(out_w  * args.scale))
    scaled_h = max(1, int(height * args.scale))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (scaled_w, scaled_h))
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output: {out_path}"); cap.release(); sys.exit(1)

    # ------------------------------------------------------------------ main loop
    frame_idx      = 0
    t_start        = time.time()
    fps_counter    = 0
    live_fps       = 0.0
    t_fps          = time.time()
    empty_mask     = np.zeros((height, width), dtype=np.uint8)
    prev_ptz_state: Optional[str] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 1. PTZ motion sensor ----
        ptz_state     = None
        learning_rate = -1.0
        if ptz_sensor is not None:
            raw_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ptz_state = ptz_sensor.update(raw_gray)
            if prev_ptz_state == CameraMotionSensor.MOVING and \
               ptz_state      == CameraMotionSensor.SETTLING:
                detector.reset(); postproc.reset(); preprocessor.reset()
                print(f"\n  [PTZ] Camera stopped at frame {frame_idx} — resetting BG model")
            learning_rate  = ptz_sensor.learning_rate
            prev_ptz_state = ptz_state

        if args.freeze_after_warmup and frame_idx >= warmup_frames:
            if ptz_state is None or ptz_state == CameraMotionSensor.ACTIVE:
                learning_rate = 0.0

        # ---- 2. Camera-shake stabilisation ----
        if stabilizer is not None:
            frame = stabilizer.stabilize(frame)

        # ---- 3. TI preprocessing ----
        preprocessed = preprocessor.process(frame)

        # ---- 4. Background subtraction ----
        raw_mask = detector.apply(preprocessed, learning_rate=learning_rate)

        # ---- 5. Detection pipeline ----
        warming_up = frame_idx < warmup_frames
        active     = (ptz_state is None or ptz_state == CameraMotionSensor.ACTIVE)

        if warming_up or not active:
            confirmed, candidates = [], []
            clean_mask = empty_mask
        else:
            clean_mask = postproc.apply_morphology(raw_mask)

            # v4: apply exclusion zones AFTER morphology, BEFORE blob extraction.
            # Morphology is allowed to operate on the full mask (so it can fill
            # holes and remove noise near zone edges correctly).  Only then do we
            # zero out the excluded regions so those blobs never reach the tracker.
            if zone_mgr.count > 0:
                clean_mask = zone_mgr.apply(clean_mask)

            blobs = postproc.extract_blobs(clean_mask)
            confirmed, candidates = postproc.update_tracker(blobs)

        # ---- 6. Visualise ----
        # Zone overlay goes ON the frame before boxes so boxes always show on top
        display_frame = frame.copy()
        if args.zone_overlay and zone_mgr.count > 0:
            display_frame = zone_mgr.draw_overlay(display_frame)

        annotated = vis.draw_boxes(display_frame, confirmed,
                                   candidates if args.show_candidates else [])
        annotated = vis.draw_hud(
            annotated, frame_idx, live_fps, args.method.upper(),
            len(candidates), len(confirmed), warming_up,
            n_zones=zone_mgr.count,
            ptz_state=ptz_state,
            ptz_motion=ptz_sensor.last_motion if ptz_sensor else 0.0,
            ptz_settle_pct=ptz_sensor.settling_progress if ptz_sensor else 0.0,
        )

        if args.debug:
            out_frame = vis.build_debug_panel(preprocessed, clean_mask, annotated)
        else:
            out_frame = annotated

        if args.scale != 1.0:
            out_frame = cv2.resize(out_frame, (scaled_w, scaled_h))

        writer.write(out_frame)

        if args.display or args.debug:
            cv2.imshow("TI Motion v4  [q to quit]", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ---- FPS counter ----
        fps_counter += 1
        now = time.time()
        if now - t_fps >= 1.0:
            live_fps = fps_counter / (now - t_fps);  fps_counter = 0;  t_fps = now

        # ---- Progress ----
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            pct = frame_idx / max(total_frames, 1) * 100
            if ptz_state == CameraMotionSensor.MOVING:
                status = f"PTZ-MOVING  {ptz_sensor.last_motion:.1f}px/f"
            elif ptz_state == CameraMotionSensor.SETTLING:
                status = f"SETTLING {ptz_sensor.settling_progress*100:.0f}%"
            elif warming_up:
                status = "WARMING-UP"
            else:
                avg = (sum(t.bbox[2] * t.bbox[3] for t in confirmed) / len(confirmed)
                       if confirmed else 0)
                status = (f"cand:{len(candidates):3d}  conf:{len(confirmed):3d}  "
                          f"avg_area:{avg:5.1f}px")
            print(f"  [{pct:5.1f}%] Frame {frame_idx:5d}/{total_frames}"
                  f"  {status}  fps:{live_fps:4.1f}    ", end="\r")

        frame_idx += 1

    # ------------------------------------------------------------------ cleanup
    elapsed = time.time() - t_start
    cap.release();  writer.release()
    if args.display or args.debug:
        cv2.destroyAllWindows()
    print(f"\n\nDone. {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps avg)")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
