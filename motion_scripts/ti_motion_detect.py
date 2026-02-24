#!/usr/bin/env python3
"""
TI Motion Detection
===================
Standalone motion detection for thermal infrared (TI) video footage.
No YOLO or any ML model — pure classical CV.

Detects moving regions and draws bounding boxes:
  Green  = confirmed motion (persisted >= --persistence frames)
  Yellow = candidate motion (not yet confirmed)

Usage examples
--------------
# Quick run with live display
python scripts/ti_motion_detect.py --input data/test.mp4 --display

# Debug view (3-panel: preprocessed | mask | annotated) — useful for tuning
python scripts/ti_motion_detect.py --input data/test.mp4 --debug --display --scale 0.5

# Difficult far-range video, small blobs only
python scripts/ti_motion_detect.py --input data/white_spots_only.mp4 \\
    --min-area 4 --max-area 100 --debug --display

# KNN method (more accurate on low-contrast TI)
python scripts/ti_motion_detect.py --input data/white_spots_only.mp4 --method knn

# Farneback optical flow (best for sub-pixel far targets)
python scripts/ti_motion_detect.py --input data/white_spots_only.mp4 --method farneback

# Tune sensitivity — more sensitive
python scripts/ti_motion_detect.py --input data/test.mp4 --threshold 8 --persistence 2

# Tune sensitivity — more specific
python scripts/ti_motion_detect.py --input data/test.mp4 --threshold 16 --persistence 5 --min-solidity 0.45

# No TI preprocessing (raw comparison baseline)
python scripts/ti_motion_detect.py --input data/test.mp4 --no-clahe --no-temporal-smooth
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prevent Qt from crashing in headless environments (no display server).
# Must be set before cv2 is imported.  If the user has a real display and
# QT_QPA_PLATFORM is already set, we leave it alone.
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
    bbox: Tuple[int, int, int, int]      # x, y, w, h  (top-left origin)
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
# Stage 1: TI-specific preprocessing
# ---------------------------------------------------------------------------

class TIPreprocessor:
    """
    Converts a raw BGR frame into a preprocessed grayscale frame suitable
    for background subtraction on thermal infrared footage.

    Two operations (both individually toggleable):
      1. CLAHE  — enhances local contrast of faint 2–8px distant blobs
      2. IIR temporal smoothing — reduces per-frame thermal speckle noise
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
        self._smooth: Optional[np.ndarray] = None  # float32 accumulator

        # CLAHE tile guard: tile must not exceed image dimension / 4
        # (enforced in process() once we know the frame size)
        self._clahe_tile_base = clahe_tile
        self._clahe_clip = clahe_clip
        self._clahe: Optional[cv2.CLAHE] = None  # built lazily on first frame

    def _build_clahe(self, height: int, width: int) -> None:
        tile = min(self._clahe_tile_base, min(width, height) // 4)
        tile = max(tile, 1)
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip,
            tileGridSize=(tile, tile),
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Input:  BGR uint8 frame  (H, W, 3)
        Output: grayscale uint8  (H, W)  — preprocessed, ready for detector
        """
        # --- grayscale ---
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # --- CLAHE (lazy init on first frame) ---
        if self._use_clahe:
            if self._clahe is None:
                self._build_clahe(gray.shape[0], gray.shape[1])
            gray = self._clahe.apply(gray)

        # --- IIR temporal smoothing ---
        if self._use_temporal:
            if self._smooth is None:
                self._smooth = gray.astype(np.float32)
            else:
                self._smooth = (
                    self._alpha * gray.astype(np.float32)
                    + (1.0 - self._alpha) * self._smooth
                )
            gray = np.clip(self._smooth, 0, 255).astype(np.uint8)

        return gray

    def reset(self) -> None:
        """Clear IIR accumulator (call on scene cuts or reinitialisation)."""
        self._smooth = None


# ---------------------------------------------------------------------------
# Stage 2: Motion detection backends
# ---------------------------------------------------------------------------

class MotionDetector:
    """
    Unified interface around four motion detection algorithms.

    All return a binary mask (H, W) uint8 with 255 = foreground.
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
        self._knn: Optional[cv2.BackgroundSubtractorKNN] = None
        self._frame_buffer: List[np.ndarray] = []

        self._build()

    def _build(self) -> None:
        if self.method == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=self._threshold,
                detectShadows=False,  # TI has no meaningful shadow semantics
            )
        elif self.method == "knn":
            # KNN uses squared distance; scale threshold accordingly
            self._knn = cv2.createBackgroundSubtractorKNN(
                history=self._history,
                dist2Threshold=self._threshold * self._threshold,
                detectShadows=False,
            )
        # diff and farneback are stateless — only the frame buffer is needed

    def apply(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        """
        Input:  preprocessed grayscale frame (H, W) uint8
        Output: binary foreground mask        (H, W) uint8  {0, 255}
        learning_rate: -1=adaptive, 0=freeze model, 0.05=fast rebuild
        """
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

    def _apply_mog2(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        raw = self._mog2.apply(gray, learningRate=learning_rate)
        # MOG2 returns: 255=foreground, 127=shadow (disabled), 0=background
        return (raw > 200).astype(np.uint8) * 255

    def _apply_knn(self, gray: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        raw = self._knn.apply(gray, learningRate=learning_rate)
        return (raw > 200).astype(np.uint8) * 255

    def _apply_diff(self, gray: np.ndarray) -> np.ndarray:
        self._frame_buffer.append(gray.copy())
        # Keep only as many frames as we need
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
        # Farneback needs exactly the previous frame and current frame
        self._frame_buffer.append(gray.copy())
        if len(self._frame_buffer) > 2:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) < 2:
            return np.zeros_like(gray)

        prev, curr = self._frame_buffer[0], self._frame_buffer[1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5,
            levels=3,
            winsize=7,        # small window — localises flow on 2–4px blobs
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Flow magnitude is in pixels/frame; threshold scales differently from intensity
        flow_thresh = max(self._threshold * 0.1, 0.3)
        _, mask = cv2.threshold(mag, flow_thresh, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3: Post-processing + temporal persistence tracker
# ---------------------------------------------------------------------------

class MaskPostprocessor:
    """
    Cleans the raw foreground mask and returns confirmed/candidate detections.

    Pipeline:
      1. Morphological closing  — fill holes in blobs
      2. Morphological opening  — remove isolated noise specks
      3. Connected component analysis with area + solidity filtering
      4. Temporal persistence tracker — blobs must persist >= N frames
    """

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

        # Elliptical kernels — better preserve round thermal blobs than rectangles
        ks_close = max(morph_close, 1)
        ks_open  = max(morph_open,  1)
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ks_close, ks_close)
        )
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ks_open, ks_open)
        )

        self._tracked: Dict[int, TrackedBlob] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------ morph

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Closing first (fills holes), then opening (removes noise).
        Order matters: closing before opening avoids destroying small blobs.
        """
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  self._open_kernel)
        return opened

    # ----------------------------------------------- connected components

    def extract_blobs(self, mask: np.ndarray) -> List[Blob]:
        """Connected component analysis with area + solidity filtering."""
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        blobs: List[Blob] = []

        for i in range(1, n_labels):  # label 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self._min_area <= area <= self._max_area):
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Density: blob pixel count / bbox area.  Compact blobs: 0.3-0.8.
            # Sparse noise merged by morph-close into a large bbox: 0.05-0.15.
            if self._min_density > 0.0 and w > 0 and h > 0:
                if area / (w * h) < self._min_density:
                    continue

            # Solidity: area / convex hull area
            # Noise streaks ~0.1–0.2, compact blobs ~0.4–0.8
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0

            if solidity < self._min_solidity:
                continue

            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            blobs.append(Blob(bbox=(x, y, w, h), centroid=(cx, cy),
                              area=area, solidity=solidity))

        return blobs

    # ----------------------------------------------------------- tracker

    def update_tracker(
        self, blobs: List[Blob]
    ) -> Tuple[List[TrackedBlob], List[TrackedBlob]]:
        """
        Greedy nearest-neighbour association.
        Returns (confirmed_blobs, candidate_blobs).
        """
        matched_track_ids = set()

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
                # Update existing track
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
                # New track
                new_id = self._next_id
                self._next_id += 1
                self._tracked[new_id] = TrackedBlob(
                    blob_id=new_id,
                    centroid=blob.centroid,
                    bbox=blob.bbox,
                    initial_centroid=blob.centroid,
                    frames_seen=1,
                    frames_absent=0,
                    confirmed=False,
                )

        # Age unmatched tracks; remove stale ones
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
# Visualisation
# ---------------------------------------------------------------------------

class MotionVisualizer:
    """All drawing operations. Stateless."""

    # Colours (BGR)
    COLOR_CONFIRMED  = (0,   230,  0)    # green
    COLOR_CANDIDATE  = (0,   220, 255)   # yellow
    COLOR_HUD_BG     = (0,     0,   0)   # black semi-transparent

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
        font_thick  = max(1, line_width // 2)

        def _draw_single(track: TrackedBlob, color: Tuple, label: str) -> None:
            x, y, bw, bh = track.bbox
            # Expand tiny boxes so they're visible to the operator
            if bw * bh < 16:
                pad = 4
                x = max(0, x - pad)
                y = max(0, y - pad)
                bw = min(w - x, bw + pad * 2)
                bh = min(h - y, bh + pad * 2)

            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, line_width)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
            )
            label_y1 = max(y - th - 6, 0)
            cv2.rectangle(
                img,
                (x, label_y1),
                (x + tw + 4, label_y1 + th + 6),
                color, -1,
            )
            cv2.putText(
                img, label,
                (x + 2, label_y1 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), font_thick,
            )

        for t in candidates:
            label = f"?{t.frames_seen}/{self._persistence}"
            _draw_single(t, self.COLOR_CANDIDATE, label)

        for t in confirmed:
            label = f"#{t.blob_id} T:{t.frames_seen} D:{t.displacement:.0f}px"
            _draw_single(t, self.COLOR_CONFIRMED, label)

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
    ) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale = max(0.4, min(h, w) / 1600)
        fthick = max(1, int(fscale * 1.5))
        line_h = int(fscale * 28) + 6

        if warming_up:
            lines = [
                f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}",
                f"Method: {method}",
                "*** WARMING UP ***",
            ]
        else:
            lines = [
                f"Frame: {frame_idx:5d}  FPS: {fps:4.1f}",
                f"Method: {method}   Conf: {n_confirmed}   Cand: {n_candidates}",
            ]

        # Semi-transparent background
        max_tw = max(
            cv2.getTextSize(l, font, fscale, fthick)[0][0] for l in lines
        )
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (max_tw + 12, len(lines) * line_h + 6),
                      (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        for i, line in enumerate(lines):
            color = (0, 60, 255) if (warming_up and i == 2) else (220, 220, 220)
            cv2.putText(
                img, line,
                (6, (i + 1) * line_h),
                font, fscale, color, fthick, cv2.LINE_AA,
            )

        return img

    @staticmethod
    def build_debug_panel(
        preprocessed: np.ndarray,
        mask: np.ndarray,
        annotated: np.ndarray,
    ) -> np.ndarray:
        """3-panel horizontal: [preprocessed gray | foreground mask | annotated]"""
        h, w = annotated.shape[:2]

        def _to_bgr(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        panel_pre  = cv2.resize(_to_bgr(preprocessed), (w, h))
        panel_mask = cv2.resize(_to_bgr(mask),          (w, h))

        # Label each panel
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        for panel, text in [
            (panel_pre,  "PREPROCESSED"),
            (panel_mask, "MOTION MASK"),
            (annotated,  "DETECTIONS"),
        ]:
            cv2.putText(panel, text, (6, 20), font, fscale,
                        (200, 200, 200), fthick, cv2.LINE_AA)

        return np.hstack([panel_pre, panel_mask, annotated])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TI Motion Detection — background subtraction for thermal IR footage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument("--input",  "-i", required=True,  help="Input video file")
    io.add_argument("--output", "-o", default=None,
                    help="Output video path (auto-named <input>_motion.mp4 if omitted)")
    io.add_argument("--display",      action="store_true",
                    help="Show live OpenCV window")
    io.add_argument("--debug",        action="store_true",
                    help="Write/show 3-panel debug view (3× frame width)")
    io.add_argument("--scale",        type=float, default=1.0,
                    help="Resize factor for display/output")

    # Detection method
    det = parser.add_argument_group("Detection method")
    det.add_argument("--method",      default="mog2",
                     choices=["mog2", "knn", "diff", "farneback"],
                     help="mog2=fast background model, knn=accurate, "
                          "diff=frame differencing, farneback=optical flow")
    det.add_argument("--history",     type=int,   default=200,
                     help="Background model history frames (mog2/knn)")
    det.add_argument("--threshold",   type=float, default=12.0,
                     help="Foreground threshold — lower=more sensitive. "
                          "For farneback this scales to pixel/frame units")
    det.add_argument("--diff-frames", type=int,   default=5,
                     help="Frame lookback gap for frame-diff method "
                          "(increase to 8–10 for slow distant targets)")
    det.add_argument("--freeze-after-warmup", action="store_true", default=False,
                     help="Freeze the background model (learningRate=0) once warmup "
                          "completes. Prevents MOG2/KNN from absorbing slow-moving "
                          "objects during long journeys. Best for static cameras.")

    # TI preprocessing
    pre = parser.add_argument_group("TI preprocessing")
    pre.add_argument("--clahe",           dest="clahe",
                     action="store_true",  default=True,
                     help="Apply CLAHE contrast enhancement")
    pre.add_argument("--no-clahe",        dest="clahe",   action="store_false")
    pre.add_argument("--clahe-clip",      type=float, default=2.5,
                     help="CLAHE clip limit (2–3 for TI; >4 amplifies sensor noise)")
    pre.add_argument("--clahe-tile",      type=int,   default=12,
                     help="CLAHE tile grid size in pixels")
    pre.add_argument("--temporal-smooth", dest="temporal_smooth",
                     action="store_true",  default=True,
                     help="Apply IIR temporal smoothing (reduces thermal speckle)")
    pre.add_argument("--no-temporal-smooth", dest="temporal_smooth",
                     action="store_false")
    pre.add_argument("--alpha",           type=float, default=0.4,
                     help="IIR current-frame weight (0.2=heavy smooth, 0.6=light)")

    # Post-processing
    post = parser.add_argument_group("Post-processing")
    post.add_argument("--morph-close",  type=int,   default=5,
                      help="Morphological closing kernel size (fills blob holes)")
    post.add_argument("--morph-open",   type=int,   default=3,
                      help="Morphological opening kernel size (removes noise specs)")
    post.add_argument("--min-area",     type=int,   default=4,
                      help="Min blob area px² (4=2×2px — human footprint at 3km)")
    post.add_argument("--max-area",     type=int,   default=5000,
                      help="Max blob area px² (remove large vegetation/sky artifacts)")
    post.add_argument("--min-solidity", type=float, default=0.3,
                      help="Min blob solidity 0–1 (noise streaks ~0.1, humans ~0.5+)")
    post.add_argument("--persistence",  type=int,   default=3,
                      help="Frames a blob must persist to become CONFIRMED "
                           "(thermal noise lasts 1–2 frames; 3 eliminates it)")
    post.add_argument("--min-displacement", type=float, default=0.0,
                      help="Min pixels a blob must travel from its origin to be CONFIRMED "
                           "(0=disabled; 30–80 eliminates stationary shimmer/noise, "
                           "keeps only objects that actually cross the frame)")
    post.add_argument("--min-density",     type=float, default=0.0,
                      help="Min ratio of blob pixels to bbox area (0=disabled). "
                           "Compact objects: 0.3-0.5. Filters large sparse bboxes "
                           "from morph-close merging nearby noise pixels. "
                           "NOTE: min/max-area = blob PIXEL COUNT not bbox size.")
    post.add_argument("--spatial-tol",     type=int,   default=15,
                      help="Max px centroid can move between frames to match same track "
                           "(default 15). Raise to 30-50 for fast objects like a bike — "
                           "if too small the tracker resets frames_seen mid-journey.")
    post.add_argument("--no-candidates",   dest="show_candidates", action="store_false",
                      default=True,
                      help="Hide yellow candidate boxes — only confirmed (green) shown. "
                           "Use when persistence is high and screen is cluttered.")
    post.add_argument("--max-absent",      type=int, default=2,
                      help="Frames a track can be undetected before being deleted "
                           "(default 2). Raise to 5-8 for difficult videos where "
                           "detection is intermittent and tracks reset mid-journey.")

    args = parser.parse_args()

    # ---------------------------------------------------------------- paths
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[ERROR] Unsupported extension '{input_path.suffix}'. "
              f"Supported: {SUPPORTED_EXTENSIONS}")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.parent / f"{input_path.stem}_motion.mp4"

    # ---------------------------------------------------------------- video open
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warmup_frames = args.history // 2

    print(f"\nInput   : {input_path}")
    print(f"Output  : {out_path}")
    print(f"Frames  : {total_frames}  |  FPS: {fps_in:.1f}  |  {width}×{height}")
    print(f"Method  : {args.method.upper()}")
    print(f"Warmup  : {warmup_frames} frames  ({warmup_frames / fps_in:.1f}s)")
    print(f"CLAHE   : {'ON' if args.clahe else 'OFF'}  "
          f"  Temporal: {'ON' if args.temporal_smooth else 'OFF'}")
    print(f"Persist : {args.persistence} frames  "
          f"  min_area: {args.min_area}px²  "
          f"  threshold: {args.threshold}  "
          f"  min_displacement: {args.min_displacement}px")
    print()

    # ---------------------------------------------------------------- instantiate pipeline
    preprocessor = TIPreprocessor(
        use_clahe=args.clahe,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        use_temporal=args.temporal_smooth,
        alpha=args.alpha,
    )
    detector = MotionDetector(
        method=args.method,
        history=args.history,
        threshold=args.threshold,
        diff_frames=args.diff_frames,
    )
    postproc = MaskPostprocessor(
        morph_close=args.morph_close,
        morph_open=args.morph_open,
        min_area=args.min_area,
        max_area=args.max_area,
        min_solidity=args.min_solidity,
        persistence=args.persistence,
        spatial_tol=args.spatial_tol,
        min_displacement=args.min_displacement,
        min_density=args.min_density,
        max_absent=args.max_absent,
    )
    vis = MotionVisualizer(persistence=args.persistence)

    # ---------------------------------------------------------------- video writer
    # Debug panel is 3× wide; compute BEFORE creating writer
    out_w = width * 3 if args.debug else width
    out_h = height
    scaled_w = max(1, int(out_w * args.scale))
    scaled_h = max(1, int(out_h * args.scale))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (scaled_w, scaled_h),
    )
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output video: {out_path}")
        cap.release()
        sys.exit(1)

    # ---------------------------------------------------------------- main loop
    frame_idx   = 0
    t_start     = time.time()
    fps_counter = 0
    live_fps    = 0.0
    t_fps       = time.time()

    empty_mask = np.zeros((height, width), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warming_up = frame_idx < warmup_frames

        # --- pipeline ---
        preprocessed = preprocessor.process(frame)
        lr = 0.0 if (args.freeze_after_warmup and not warming_up) else -1.0
        raw_mask     = detector.apply(preprocessed, learning_rate=lr)

        if warming_up:
            confirmed, candidates = [], []
            clean_mask = empty_mask
        else:
            clean_mask = postproc.apply_morphology(raw_mask)
            blobs      = postproc.extract_blobs(clean_mask)
            confirmed, candidates = postproc.update_tracker(blobs)

        # --- visualise ---
        annotated = vis.draw_boxes(frame, confirmed,
                                   candidates if args.show_candidates else [])
        annotated = vis.draw_hud(
            annotated, frame_idx, live_fps,
            args.method.upper(), len(candidates), len(confirmed), warming_up,
        )

        # --- assemble output frame ---
        if args.debug:
            out_frame = vis.build_debug_panel(preprocessed, clean_mask, annotated)
        else:
            out_frame = annotated

        if args.scale != 1.0:
            out_frame = cv2.resize(out_frame, (scaled_w, scaled_h))

        writer.write(out_frame)

        if args.display or args.debug:
            cv2.imshow("TI Motion Detection  [q to quit]", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # --- FPS counter ---
        fps_counter += 1
        now = time.time()
        if now - t_fps >= 1.0:
            live_fps    = fps_counter / (now - t_fps)
            fps_counter = 0
            t_fps       = now

        # --- progress ---
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            pct = frame_idx / max(total_frames, 1) * 100
            if warming_up:
                status = "WARMING UP"
            else:
                avg = (
                    sum(
                        (t.bbox[2] * t.bbox[3]) for t in confirmed
                    ) / len(confirmed)
                    if confirmed else 0
                )
                status = f"cand:{len(candidates):3d}  conf:{len(confirmed):3d}  avg_area:{avg:5.1f}px"
            print(
                f"  [{pct:5.1f}%] Frame {frame_idx:5d}/{total_frames}"
                f"  {status}  fps:{live_fps:4.1f}    ",
                end="\r",
            )

        frame_idx += 1

    # ---------------------------------------------------------------- cleanup
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
