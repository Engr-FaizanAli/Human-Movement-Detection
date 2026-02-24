#!/usr/bin/env python3
"""
TI Motion Detection v2
======================
Extends v1 with two new optional stages:

  1. Camera-shake stabilisation  (--stabilize)
     Tracks sparse features with Lucas-Kanade optical flow, estimates the
     dominant affine motion (translation + rotation) via RANSAC, accumulates
     the camera trajectory, and smooths it with an EMA filter.  Each frame is
     warped by the correction transform before being fed to the background
     subtractor.  Removes handheld/mount vibration without affecting object
     detection accuracy.

  2. PTZ-aware background gating  (--ptz-aware)
     Detects whether the camera platform is panning / tilting by measuring the
     magnitude of the RANSAC-estimated global motion per frame.  RANSAC
     discards pixels belonging to foreground objects, so only true camera
     motion triggers the state change.

     Three-state machine
     -------------------
     MOVING   — pan/tilt detected  (global motion > --ptz-motion-thresh px/f)
                BG model frozen (learningRate=0); detections suppressed.
     SETTLING — camera just stopped; BG model rebuilt fast (learningRate=0.05);
                detections suppressed; progress bar shown in HUD.
     ACTIVE   — static for --ptz-settle-time s; full detection pipeline.

     On MOVING→SETTLING transition the background model and tracker are reset,
     so the model is learnt entirely on static footage.

Why RANSAC for camera-motion estimation?
-----------------------------------------
Dense optical flow sees *everything* — objects + camera.  RANSAC affine fit
models the dominant (majority) motion, which on a surveillance camera is the
camera itself.  Moving objects are statistical outliers and are discarded.
This gives a clean separation between "camera moved" and "object moved".

How military/professional systems differ
-----------------------------------------
  • Hardware IMU/encoder feeds pan-tilt angles directly — no CV needed.
  • Full spherical panoramic background model — detection *continues* during a
    pan by registering each frame to the panorama.
  • This script is the pragmatic software-only equivalent that works well when
    the camera is mostly static with occasional manual repositioning.

Usage
-----
# Shaky handheld phone video — stabilise then detect
python scripts/ti_motion_detect_v2.py --input video.mp4 --stabilize

# PTZ cam that rotates every 30 min — gate background on static periods
python scripts/ti_motion_detect_v2.py --input video.mp4 --ptz-aware

# Both together
python scripts/ti_motion_detect_v2.py --input video.mp4 --stabilize --ptz-aware \\
    --ptz-motion-thresh 4.0 --ptz-settle-time 5.0

# Full debug panel (3-pane) with live display
python scripts/ti_motion_detect_v2.py --input video.mp4 --stabilize --ptz-aware \\
    --debug --display --scale 0.6
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") is None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Data structures
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

    @property
    def displacement(self) -> float:
        dx = self.centroid[0] - self.initial_centroid[0]
        dy = self.centroid[1] - self.initial_centroid[1]
        return (dx * dx + dy * dy) ** 0.5


# ---------------------------------------------------------------------------
# Stage 1: TI-specific preprocessing  (unchanged from v1)
# ---------------------------------------------------------------------------

class TIPreprocessor:
    """
    Grayscale + optional CLAHE + optional IIR temporal smoothing.
    Reduces thermal speckle and enhances faint distant blobs.
    """

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
# Stage 1b: Camera-shake stabilisation  (NEW)
# ---------------------------------------------------------------------------

class CameraStabilizer:
    """
    Removes per-frame camera shake before background subtraction.

    Algorithm
    ---------
    1. goodFeaturesToTrack — Shi-Tomasi corners on the previous grayscale frame.
    2. calcOpticalFlowPyrLK — track those corners into the current frame.
    3. estimateAffinePartial2D (RANSAC) — fit a 4-DOF model (translation +
       rotation, no shear/scale).  RANSAC rejects moving-object points so only
       the dominant camera motion is estimated.
    4. Accumulate the raw trajectory [cum_dx, cum_dy, cum_da] and smooth it
       with an Exponential Moving Average (EMA).
    5. Apply (smooth − raw) as a correction warp via warpAffine with
       BORDER_REPLICATE to avoid black border artefacts.

    Tuning
    ------
    smooth_radius large (50-100) → very stable, corrects slow drift,
                                   may slightly over-crop on large pans.
    smooth_radius small (10-20)  → faster response to intentional moves,
                                   less aggressive on slow shake.
    """

    def __init__(
        self,
        smooth_radius: int = 30,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 30.0,
    ):
        self._alpha = 2.0 / (smooth_radius + 1)   # EMA coefficient
        self._max_corners = max_corners
        self._quality = quality_level
        self._min_dist = min_distance

        self._prev_gray: Optional[np.ndarray] = None
        self._cum = np.zeros(3, dtype=np.float64)     # [dx, dy, da]  raw trajectory
        self._smooth = np.zeros(3, dtype=np.float64)  # EMA-smoothed trajectory

    # ------------------------------------------------------------------ public

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Input:  BGR (or gray) frame  — the raw camera frame.
        Output: Warp-corrected frame of identical shape.
        On the very first frame returns the original unchanged.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame

        delta = self._estimate_transform(self._prev_gray, gray)  # [dx, dy, da]
        self._prev_gray = gray

        self._cum    += delta
        self._smooth += self._alpha * (self._cum - self._smooth)

        corr = self._smooth - self._cum    # correction = where we want − where we are
        h, w = frame.shape[:2]
        M = _make_affine(corr[0], corr[1], corr[2], w / 2.0, h / 2.0)

        return cv2.warpAffine(frame, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def reset(self) -> None:
        self._prev_gray = None
        self._cum[:] = 0.0
        self._smooth[:] = 0.0

    # ----------------------------------------------------------------- private

    def _estimate_transform(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(prev,
                                      maxCorners=self._max_corners,
                                      qualityLevel=self._quality,
                                      minDistance=self._min_dist,
                                      blockSize=7)
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
# Stage 1c: PTZ camera motion sensor  (NEW)
# ---------------------------------------------------------------------------

class CameraMotionSensor:
    """
    Detects whether the camera platform is panning or tilting.

    Uses the same RANSAC-affine approach as CameraStabilizer to estimate the
    dominant global translation magnitude in pixels/frame.  Moving objects are
    RANSAC-rejected outliers; only actual camera motion is measured.

    State machine
    -------------
    MOVING   — camera pan/tilt detected (motion_px > ptz_motion_thresh).
               BG model should be frozen: learningRate = 0.
    SETTLING — camera just stopped moving.  BG model should be rebuilt fast:
               learningRate = 0.05.  Detections suppressed until fully settled.
    ACTIVE   — camera static for settle_frames frames.  Full detection.
               learningRate = -1 (MOG2/KNN adaptive default).

    MOVING → SETTLING triggers a background model + tracker reset so the model
    is trained exclusively on static footage.
    """

    MOVING   = "MOVING"
    SETTLING = "SETTLING"
    ACTIVE   = "ACTIVE"

    # Background learningRate for MOG2 / KNN per state
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
        self._state: str = self.SETTLING    # start in SETTLING so BG is built on first N frames
        self._settle_counter: int = 0
        self.last_motion: float = 0.0       # last estimated global motion (px/frame)

    # ------------------------------------------------------------------ public

    def update(self, gray: np.ndarray) -> str:
        """
        Feed the current raw (pre-stabilisation) grayscale frame.
        Returns the current state string.
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return self._state

        motion = self._estimate_global_motion(self._prev_gray, gray)
        self._prev_gray = gray
        self.last_motion = motion

        if motion > self._motion_thresh:
            # Camera moving — transition to MOVING (or stay)
            if self._state != self.MOVING:
                self._state = self.MOVING
                self._settle_counter = 0
        else:
            # Camera appears static
            if self._state == self.MOVING:
                # Just stopped — begin settle period
                self._state = self.SETTLING
                self._settle_counter = 0
            elif self._state == self.SETTLING:
                self._settle_counter += 1
                if self._settle_counter >= self._settle_frames:
                    self._state = self.ACTIVE
            # ACTIVE → stays ACTIVE until motion detected again

        return self._state

    @property
    def learning_rate(self) -> float:
        """Recommended MOG2/KNN learningRate for the current state."""
        return self._LR[self._state]

    @property
    def settling_progress(self) -> float:
        """0.0–1.0 fraction through the settle period (1.0 = fully settled)."""
        if self._state == self.ACTIVE:
            return 1.0
        if self._state == self.MOVING:
            return 0.0
        return min(1.0, self._settle_counter / max(self._settle_frames, 1))

    def reset(self) -> None:
        self._prev_gray = None
        self._settle_counter = 0
        self._state = self.SETTLING

    # ----------------------------------------------------------------- private

    def _estimate_global_motion(self, prev: np.ndarray, curr: np.ndarray) -> float:
        pts = cv2.goodFeaturesToTrack(prev,
                                      maxCorners=self._max_corners,
                                      qualityLevel=self._quality,
                                      minDistance=self._min_dist,
                                      blockSize=7)
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
    """Build a 2×3 affine correction matrix (rotation around centre + translation)."""
    c, s = np.cos(da), np.sin(da)
    return np.array([
        [c, -s, (1 - c) * cx + s * cy + dx],
        [s,  c, (1 - c) * cy - s * cx + dy],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 2: Motion detection backends  (learningRate param added)
# ---------------------------------------------------------------------------

class MotionDetector:
    """
    Same four backends as v1.
    apply() now accepts an explicit learning_rate for PTZ gating:
      -1   → adaptive default (MOG2/KNN internal schedule)
       0   → freeze — model is not updated  (use during camera pan)
      0.05 → fast rebuild  (use during settle period)
    """

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
                history=self._history,
                varThreshold=self._threshold,
                detectShadows=False,
            )
        elif self.method == "knn":
            self._knn = cv2.createBackgroundSubtractorKNN(
                history=self._history,
                dist2Threshold=self._threshold * self._threshold,
                detectShadows=False,
            )

    def apply(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        if self.method == "mog2":
            return self._apply_mog2(gray, learning_rate)
        elif self.method == "knn":
            return self._apply_knn(gray, learning_rate)
        elif self.method == "diff":
            return self._apply_diff(gray)
        elif self.method == "farneback":
            return self._apply_farneback(gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def reset(self) -> None:
        """
        Rebuild background models from scratch.
        Call this when the scene changes (e.g. after a PTZ pan).
        """
        self._frame_buffer.clear()
        self._build()

    def _apply_mog2(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        raw = self._mog2.apply(gray, learningRate=learning_rate)
        return (raw > 200).astype(np.uint8) * 255

    def _apply_knn(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        raw = self._knn.apply(gray, learningRate=learning_rate)
        return (raw > 200).astype(np.uint8) * 255

    def _apply_diff(self, gray: np.ndarray) -> np.ndarray:
        self._frame_buffer.append(gray.copy())
        max_buf = self._diff_frames + 1
        if len(self._frame_buffer) > max_buf:
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
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_thresh = max(self._threshold * 0.1, 0.3)
        _, mask = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3: Post-processing + temporal persistence tracker  (min_displacement added)
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
    ):
        self._min_area = min_area
        self._max_area = max_area
        self._min_solidity = min_solidity
        self._persistence = persistence
        self._spatial_tol = spatial_tol
        self._min_displacement = min_displacement
        self._min_density = min_density
        self._max_absent = max_absent

        ks_close = max(morph_close, 1)
        ks_open  = max(morph_open,  1)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_close, ks_close))
        self._open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_open,  ks_open))

        self._tracked: Dict[int, TrackedBlob] = {}
        self._next_id: int = 0

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  self._open_kernel)
        return opened

    def extract_blobs(self, mask: np.ndarray) -> List[Blob]:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs: List[Blob] = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self._min_area <= area <= self._max_area):
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Density: ratio of foreground pixels to bounding-box area.
            # Compact solid blobs (bike, person): 0.3–0.8.
            # Sparse noise merged by morph-close into a large bbox: 0.05–0.15.
            # This directly limits oversized bboxes from scattered pixels.
            if self._min_density > 0.0 and w > 0 and h > 0:
                if area / (w * h) < self._min_density:
                    continue

            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            if solidity < self._min_solidity:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            blobs.append(Blob(bbox=(x, y, w, h), centroid=(cx, cy), area=area, solidity=solidity))
        return blobs

    def update_tracker(self, blobs: List[Blob]) -> Tuple[List[TrackedBlob], List[TrackedBlob]]:
        matched_track_ids: set = set()

        for blob in blobs:
            best_id: Optional[int] = None
            best_dist: float = self._spatial_tol

            for tid, track in self._tracked.items():
                dx = blob.centroid[0] - track.centroid[0]
                dy = blob.centroid[1] - track.centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                t = self._tracked[best_id]
                t.centroid = blob.centroid
                t.bbox = blob.bbox
                t.frames_seen += 1
                t.frames_absent = 0
                t.confirmed = (
                    t.frames_seen >= self._persistence
                    and t.displacement >= self._min_displacement
                )
                matched_track_ids.add(best_id)
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracked[new_id] = TrackedBlob(
                    blob_id=new_id,
                    centroid=blob.centroid,
                    bbox=blob.bbox,
                    initial_centroid=blob.centroid,
                )

        stale = []
        for tid, track in self._tracked.items():
            if tid not in matched_track_ids:
                track.frames_absent += 1
                if track.frames_absent > self._max_absent:
                    stale.append(tid)
        for tid in stale:
            del self._tracked[tid]

        confirmed  = [t for t in self._tracked.values() if t.confirmed]
        candidates = [t for t in self._tracked.values() if not t.confirmed]
        return confirmed, candidates

    def reset(self) -> None:
        self._tracked.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Visualisation  (PTZ state overlay added to draw_hud)
# ---------------------------------------------------------------------------

class MotionVisualizer:

    COLOR_CONFIRMED = (0,   230,   0)
    COLOR_CANDIDATE = (0,   220, 255)

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
        line_width = max(1, int(min(h, w) * 0.002))
        font_scale = max(0.35, min(h, w) / 1800)
        font_thick = max(1, line_width // 2)

        def _draw_single(track: TrackedBlob, color: tuple, label: str) -> None:
            x, y, bw, bh = track.bbox
            if bw * bh < 16:
                pad = 4
                x  = max(0, x - pad);  y  = max(0, y - pad)
                bw = min(w - x, bw + pad * 2);  bh = min(h - y, bh + pad * 2)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, line_width)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            label_y1 = max(y - th - 6, 0)
            cv2.rectangle(img, (x, label_y1), (x + tw + 4, label_y1 + th + 6), color, -1)
            cv2.putText(img, label, (x + 2, label_y1 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick)

        for t in candidates:
            _draw_single(t, self.COLOR_CANDIDATE, f"?{t.frames_seen}/{self._persistence}")

        for t in confirmed:
            _draw_single(t, self.COLOR_CONFIRMED,
                         f"#{t.blob_id} T:{t.frames_seen} D:{t.displacement:.0f}px")

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

        # Build HUD lines based on state priority
        if ptz_state == CameraMotionSensor.MOVING:
            lines  = [
                f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}  Method: {method}",
                f"PTZ MOVING  {ptz_motion:.1f} px/f  — detections paused",
            ]
            colors = [(220, 220, 220), (0, 60, 255)]

        elif ptz_state == CameraMotionSensor.SETTLING:
            bar_w  = max(1, int(ptz_settle_pct * 20))
            bar    = "[" + "#" * bar_w + "-" * (20 - bar_w) + "]"
            lines  = [
                f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}  Method: {method}",
                f"SETTLING BG  {bar}  {ptz_settle_pct*100:.0f}%",
            ]
            colors = [(220, 220, 220), (0, 165, 255)]

        elif warming_up:
            lines  = [
                f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}",
                f"Method: {method}",
                "*** WARMING UP ***",
            ]
            colors = [(220, 220, 220), (220, 220, 220), (0, 60, 255)]

        else:
            lines  = [
                f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}",
                f"Method: {method}   Conf: {n_confirmed}   Cand: {n_candidates}",
            ]
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

        def _bgr(img: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

        panel_pre  = cv2.resize(_bgr(preprocessed), (w, h))
        panel_mask = cv2.resize(_bgr(mask),          (w, h))

        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        for panel, text in [(panel_pre,  "PREPROCESSED"),
                            (panel_mask, "MOTION MASK"),
                            (annotated,  "DETECTIONS")]:
            cv2.putText(panel, text, (6, 20), font, fs, (200, 200, 200), ft, cv2.LINE_AA)

        return np.hstack([panel_pre, panel_mask, annotated])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TI Motion Detection v2 — stabilisation + PTZ-aware background gating",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument("--input",   "-i", required=True)
    io.add_argument("--output",  "-o", default=None,
                    help="Output path (auto-named <input>_v2.mp4 if omitted)")
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
    det.add_argument("--freeze-after-warmup", action="store_true", default=False,
                     help="Freeze the background model (learningRate=0) once warmup "
                          "completes. Prevents MOG2/KNN from absorbing slow-moving "
                          "objects into the background during long journeys. "
                          "Best for static cameras with short clips.")

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
    post.add_argument("--persistence",      type=int,   default=3,
                      help="Frames a blob must persist to become CONFIRMED")
    post.add_argument("--min-displacement", type=float, default=0.0,
                      help="Min px a blob must travel from origin to be CONFIRMED "
                           "(0=disabled; 30–80 eliminates stationary shimmer)")
    post.add_argument("--min-density",     type=float, default=0.0,
                      help="Min ratio of blob pixels to bbox area (0=disabled). "
                           "Compact objects: 0.3-0.5. Filters large sparse bboxes "
                           "caused by morph-close merging nearby noise pixels. "
                           "NOTE: min/max-area = blob pixel count, NOT bbox size. "
                           "Use this to reject oversized boxes from sparse blobs.")
    post.add_argument("--spatial-tol",     type=int,   default=15,
                      help="Max px centroid can move between frames and still match "
                           "the same track (default 15). Raise to 30-50 for fast "
                           "objects like a bike — if too small the tracker drops the "
                           "track mid-journey and frames_seen resets to 1.")
    post.add_argument("--no-candidates",   dest="show_candidates", action="store_false",
                      default=True,
                      help="Hide yellow candidate boxes — only confirmed (green) boxes "
                           "are shown. Useful when persistence is high and the screen "
                           "is cluttered with yellow noise that never gets confirmed.")
    post.add_argument("--max-absent",      type=int, default=2,
                      help="Frames a track can be undetected before being deleted "
                           "(default 2). If bike blob vanishes for 3-5 frames due to "
                           "noise/occlusion the track resets. Raise to 5-8 for "
                           "difficult videos where detection is intermittent.")

    # Camera stabilisation  (NEW)
    stab = parser.add_argument_group("Camera stabilisation  (--stabilize)")
    stab.add_argument("--stabilize",        action="store_true", default=False,
                      help="Enable shake stabilisation (LK optical flow + EMA smoothing)")
    stab.add_argument("--stabilize-radius", type=int, default=30,
                      help="EMA smoothing window in frames "
                           "(larger=more stable, smaller=less border crop)")

    # PTZ-aware gating  (NEW)
    ptz = parser.add_argument_group("PTZ-aware background gating  (--ptz-aware)")
    ptz.add_argument("--ptz-aware",         action="store_true", default=False,
                     help="Enable PTZ motion state machine "
                          "(freezes BG model during pan, rebuilds on settle)")
    ptz.add_argument("--ptz-motion-thresh", type=float, default=5.0,
                     help="Global motion px/frame threshold to classify as MOVING "
                          "(lower=more sensitive to pan; try 3–8 depending on cam)")
    ptz.add_argument("--ptz-settle-time",   type=float, default=5.0,
                     help="Seconds of static footage required before re-enabling "
                          "detections after a pan (background rebuild window)")

    args = parser.parse_args()

    # ------------------------------------------------------------------ paths
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}"); sys.exit(1)
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[ERROR] Unsupported extension: {input_path.suffix}"); sys.exit(1)

    out_path = Path(args.output) if args.output else \
               input_path.parent / f"{input_path.stem}_v2.mp4"

    # ------------------------------------------------------------------ video open
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}"); sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warmup_frames = args.history // 2

    settle_frames = max(1, int(args.ptz_settle_time * fps_in))

    print(f"\nInput      : {input_path}")
    print(f"Output     : {out_path}")
    print(f"Frames     : {total_frames}  |  FPS: {fps_in:.1f}  |  {width}×{height}")
    print(f"Method     : {args.method.upper()}")
    print(f"Stabilise  : {'ON  (radius=' + str(args.stabilize_radius) + ')' if args.stabilize else 'OFF'}")
    print(f"PTZ-aware  : {'ON  (thresh=' + str(args.ptz_motion_thresh) + 'px/f  settle=' + str(args.ptz_settle_time) + 's)' if args.ptz_aware else 'OFF'}")
    print(f"Warmup     : {warmup_frames} frames  ({warmup_frames / fps_in:.1f}s)")
    print(f"Persistence: {args.persistence} frames   min_disp: {args.min_displacement}px")
    print()

    # ------------------------------------------------------------------ pipeline objects
    preprocessor = TIPreprocessor(
        use_clahe=args.clahe, clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
        use_temporal=args.temporal_smooth, alpha=args.alpha,
    )
    detector = MotionDetector(
        method=args.method, history=args.history,
        threshold=args.threshold, diff_frames=args.diff_frames,
    )
    postproc = MaskPostprocessor(
        morph_close=args.morph_close, morph_open=args.morph_open,
        min_area=args.min_area, max_area=args.max_area,
        min_solidity=args.min_solidity, persistence=args.persistence,
        spatial_tol=args.spatial_tol,
        min_displacement=args.min_displacement,
        min_density=args.min_density,
        max_absent=args.max_absent,
    )
    vis = MotionVisualizer(persistence=args.persistence)

    stabilizer:  Optional[CameraStabilizer]   = CameraStabilizer(smooth_radius=args.stabilize_radius) if args.stabilize else None
    ptz_sensor:  Optional[CameraMotionSensor] = CameraMotionSensor(motion_thresh=args.ptz_motion_thresh, settle_frames=settle_frames) if args.ptz_aware else None

    # ------------------------------------------------------------------ video writer
    out_w = width * 3 if args.debug else width
    scaled_w = max(1, int(out_w    * args.scale))
    scaled_h = max(1, int(height   * args.scale))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (scaled_w, scaled_h))
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output: {out_path}"); cap.release(); sys.exit(1)

    # ------------------------------------------------------------------ main loop
    frame_idx    = 0
    t_start      = time.time()
    fps_counter  = 0
    live_fps     = 0.0
    t_fps        = time.time()
    empty_mask   = np.zeros((height, width), dtype=np.uint8)
    prev_ptz_state: Optional[str] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 1. PTZ motion sensor (on raw frame, before any correction) ----
        ptz_state   = None
        learning_rate = -1.0

        if ptz_sensor is not None:
            raw_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ptz_state = ptz_sensor.update(raw_gray)

            # Transition MOVING→SETTLING: reset BG model + tracker so they
            # rebuild cleanly on static footage only.
            if prev_ptz_state == CameraMotionSensor.MOVING and \
               ptz_state      == CameraMotionSensor.SETTLING:
                detector.reset()
                postproc.reset()
                preprocessor.reset()
                print(f"\n  [PTZ] Camera stopped at frame {frame_idx} — resetting background model")

            learning_rate   = ptz_sensor.learning_rate
            prev_ptz_state  = ptz_state

        # --freeze-after-warmup: once the background is built, stop all adaptation.
        # Overrides PTZ learning rate only when ACTIVE (not during MOVING/SETTLING).
        if args.freeze_after_warmup and frame_idx >= warmup_frames:
            if ptz_state is None or ptz_state == CameraMotionSensor.ACTIVE:
                learning_rate = 0.0

        # ---- 2. Camera-shake stabilisation ----
        if stabilizer is not None:
            frame = stabilizer.stabilize(frame)

        # ---- 3. TI preprocessing ----
        preprocessed = preprocessor.process(frame)

        # ---- 4. Background subtraction ----
        # Always call detector.apply() so the BG model is updated according to
        # the learning_rate:  0=freeze, 0.05=fast rebuild, -1=adaptive.
        raw_mask = detector.apply(preprocessed, learning_rate=learning_rate)

        # ---- 5. Detection pipeline (only when ACTIVE or no PTZ gating) ----
        warming_up = frame_idx < warmup_frames
        active     = (ptz_state is None or ptz_state == CameraMotionSensor.ACTIVE)

        if warming_up or not active:
            confirmed, candidates = [], []
            clean_mask = empty_mask
        else:
            clean_mask = postproc.apply_morphology(raw_mask)
            blobs      = postproc.extract_blobs(clean_mask)
            confirmed, candidates = postproc.update_tracker(blobs)

        # ---- 6. Visualise ----
        annotated = vis.draw_boxes(frame, confirmed,
                                   candidates if args.show_candidates else [])
        annotated = vis.draw_hud(
            annotated, frame_idx, live_fps, args.method.upper(),
            len(candidates), len(confirmed), warming_up,
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
            cv2.imshow("TI Motion v2  [q to quit]", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ---- FPS counter ----
        fps_counter += 1
        now = time.time()
        if now - t_fps >= 1.0:
            live_fps    = fps_counter / (now - t_fps)
            fps_counter = 0
            t_fps       = now

        # ---- Progress ----
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            pct = frame_idx / max(total_frames, 1) * 100
            if ptz_state == CameraMotionSensor.MOVING:
                status = f"PTZ-MOVING  motion:{ptz_sensor.last_motion:.1f}px/f"
            elif ptz_state == CameraMotionSensor.SETTLING:
                status = f"SETTLING {ptz_sensor.settling_progress*100:.0f}%"
            elif warming_up:
                status = "WARMING-UP"
            else:
                avg = (sum(t.bbox[2]*t.bbox[3] for t in confirmed) / len(confirmed)
                       if confirmed else 0)
                status = f"cand:{len(candidates):3d}  conf:{len(confirmed):3d}  avg_area:{avg:5.1f}px"
            print(f"  [{pct:5.1f}%] Frame {frame_idx:5d}/{total_frames}"
                  f"  {status}  fps:{live_fps:4.1f}    ", end="\r")

        frame_idx += 1

    # ------------------------------------------------------------------ cleanup
    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    if args.display or args.debug:
        cv2.destroyAllWindows()

    print(f"\n\nDone. {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps avg)")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
