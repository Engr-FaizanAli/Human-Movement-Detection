#!/usr/bin/env python3
"""
TI Motion Compare — Live 4-Method Side-by-Side Tuner
=====================================================
Plays your video in real-time across four OpenCV windows simultaneously,
one for each detection method: MOG2 / KNN / Diff / Farneback.

Each window has its own independent set of trackbars so you can tune
every parameter per-method and compare results in real time.

A two-line command bar at the bottom of every frame shows the exact
`ti_motion_detect.py` CLI command that reproduces current settings —
copy it straight to a terminal when you're happy.

A small motion-mask inset sits in the top-right corner of each frame
so you can see both the detection result AND the raw mask quality at once.

Controls
--------
  q / Esc   Quit
  r         Restart video + reset all background models  (re-warms from frame 0)
  Space     Pause / resume

Warmup note
-----------
MOG2 and KNN build a per-pixel Gaussian/KNN background model.  During the
first `history // 2` frames (warmup), the model is still initialising.
Warmup IS background registration — the model learns what "empty scene"
looks like.  Detection output is suppressed during warmup (shown as
"*** WARMING UP ***") and resumes once the model is stable.
For Diff, warmup = filling the frame buffer (diff_frames + 1 frames).
For Farneback, warmup = 2 frames (just needs prev + current).

Usage
-----
  python scripts/ti_motion_compare.py --input data/test.mp4
  python scripts/ti_motion_compare.py --input data/test.mp4 --scale 0.5
"""

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Must be set before cv2 import in headless environments
if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") is None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np

# Import the pipeline classes from the sibling detection script
_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))
from ti_motion_detect import (
    MaskPostprocessor,
    MotionDetector,
    MotionVisualizer,
    TIPreprocessor,
    TrackedBlob,
)

# ============================================================
# Parameter descriptors
# ============================================================

@dataclass
class Param:
    """One tunable parameter + its OpenCV trackbar representation."""
    flag:     str    # CLI flag, e.g. "--threshold"
    label:    str    # Trackbar label shown in window  (keep ≤ 32 chars)
    tmax:     int    # Trackbar integer maximum  (range: 0 .. tmax)
    tdef:     int    # Default trackbar integer position
    scale:    float  # real_value = max(real_min,  tpos * scale)
    real_min: float  # Floor clamp for the real value
    is_int:   bool = False   # True → display as integer in command string


def _real(p: Param, tpos: int) -> float:
    """Convert trackbar integer position → real parameter value."""
    return max(p.real_min, tpos * p.scale)


# ── Individual parameter singletons ──────────────────────────────────────────
#  tmax   tdef  scale  real_min  is_int
P_HISTORY     = Param("--history",      "History (frames)",      500, 200, 1.0,  10,  True)
P_THRESHOLD   = Param("--threshold",    "Threshold (x 0.5)",     100,  24, 0.5,  0.5       )
P_DIFF_FRAMES = Param("--diff-frames",  "Diff Frame Gap",         30,   5, 1.0,  1,   True)
P_CLAHE_CLIP  = Param("--clahe-clip",   "CLAHE Clip (x 0.1)",    100,  25, 0.1,  0.5       )
P_CLAHE_TILE  = Param("--clahe-tile",   "CLAHE Tile",             64,  12, 1.0,  4,   True)
P_ALPHA       = Param("--alpha",        "IIR Alpha (x 0.1)",       9,   4, 0.1,  0.1       )
P_MORPH_CLOSE = Param("--morph-close",  "Morph Close",            21,   5, 1.0,  1,   True)
P_MORPH_OPEN  = Param("--morph-open",   "Morph Open",             21,   3, 1.0,  1,   True)
P_MIN_AREA    = Param("--min-area",     "Min Area (px²)",        500,   4, 1.0,  1,   True)
P_PERSISTENCE = Param("--persistence",  "Persistence (frames)",   15,   3, 1.0,  1,   True)
P_SOLIDITY    = Param("--min-solidity", "Min Solidity (x 0.1)",    9,   3, 0.1,  0.0       )

_COMMON: List[Param] = [
    P_THRESHOLD, P_CLAHE_CLIP, P_CLAHE_TILE, P_ALPHA,
    P_MORPH_CLOSE, P_MORPH_OPEN, P_MIN_AREA, P_PERSISTENCE, P_SOLIDITY,
]

# Parameters available per method
METHOD_PARAMS: Dict[str, List[Param]] = {
    "mog2":      [P_HISTORY] + _COMMON,
    "knn":       [P_HISTORY] + _COMMON,
    "diff":      [P_DIFF_FRAMES] + _COMMON,
    "farneback": _COMMON,
}

METHODS = ["mog2", "knn", "diff", "farneback"]

# Which CLI flags belong to which pipeline stage (determines what to rebuild)
_PRE_FLAGS  = {"--clahe-clip", "--clahe-tile", "--alpha"}
_DET_FLAGS  = {"--history", "--threshold", "--diff-frames"}
_POST_FLAGS = {"--morph-close", "--morph-open", "--min-area", "--persistence", "--min-solidity"}


# ============================================================
# Pipeline instance  (one per method / window)
# ============================================================

class PipelineInstance:
    """
    Complete detection pipeline for a single method.
    Reads live trackbar values and rebuilds only the affected stage on change.
    """

    def __init__(self, method: str, win_name: str) -> None:
        self.method   = method
        self.win_name = win_name
        self.pdefs    = METHOD_PARAMS[method]

        self._cache: Dict[str, int] = {}  # last-seen trackbar positions
        self.frame_count = 0              # frames processed since last reset
        # Components are built after calling build_all() (requires trackbars to exist)

    # ---------------------------------------------------------------- internal helpers

    def _tb(self, p: Param) -> int:
        """Read trackbar position; fall back to default if not yet initialised."""
        try:
            v = cv2.getTrackbarPos(p.label, self.win_name)
            return v if v >= 0 else p.tdef
        except Exception:
            return p.tdef

    def _v(self, flag: str) -> float:
        """Real value of a parameter by CLI flag name."""
        p = next((x for x in self.pdefs if x.flag == flag), None)
        if p is None:
            # Sensible defaults for flags not present in this method's param list
            return {"--history": 200.0, "--diff-frames": 5.0}.get(flag, 0.0)
        return _real(p, self._tb(p))

    def _has(self, flag: str) -> bool:
        return any(p.flag == flag for p in self.pdefs)

    # ---------------------------------------------------------------- build

    def build_all(self) -> None:
        """Full build of all three pipeline stages (call after trackbars exist)."""
        self._build_pre()
        self._build_det()
        self._build_post()

    def _build_pre(self) -> None:
        self.preprocessor = TIPreprocessor(
            use_clahe    = True,
            clahe_clip   = self._v("--clahe-clip"),
            clahe_tile   = int(self._v("--clahe-tile")),
            use_temporal = True,
            alpha        = self._v("--alpha"),
        )

    def _build_det(self) -> None:
        self.detector = MotionDetector(
            method      = self.method,
            history     = int(self._v("--history")),
            threshold   = self._v("--threshold"),
            diff_frames = int(self._v("--diff-frames")),
        )
        self.frame_count = 0   # reset warmup counter whenever detector is rebuilt

    def _build_post(self) -> None:
        pers = int(self._v("--persistence"))
        self.postproc = MaskPostprocessor(
            morph_close  = int(self._v("--morph-close")),
            morph_open   = int(self._v("--morph-open")),
            min_area     = int(self._v("--min-area")),
            max_area     = 5000,
            min_solidity = self._v("--min-solidity"),
            persistence  = pers,
        )
        self.vis = MotionVisualizer(persistence=pers)

    # ---------------------------------------------------------------- sync (trackbar polling)

    def sync(self) -> None:
        """
        Poll all trackbars; detect changes; rebuild only the affected stage(s).
        Call once per frame before process().
        """
        cur = {p.flag: self._tb(p) for p in self.pdefs}

        # First call: seed the cache without rebuilding (already built with defaults)
        if not self._cache:
            self._cache = cur
            return

        changed = {flag for flag, v in cur.items() if v != self._cache[flag]}
        self._cache = cur

        if not changed:
            return

        if changed & _PRE_FLAGS:
            self._build_pre()
        if changed & _DET_FLAGS:
            self._build_det()   # also resets frame_count → warmup restarts
        if changed & _POST_FLAGS:
            self._build_post()

    # ---------------------------------------------------------------- reset

    def reset(self) -> None:
        """Full reset — call on video loop or manual restart."""
        self.build_all()

    # ---------------------------------------------------------------- process

    @property
    def warmup_len(self) -> int:
        if self.method in ("mog2", "knn"):
            return max(5, int(self._v("--history")) // 2)
        if self.method == "diff":
            return max(2, int(self._v("--diff-frames")) + 1)
        return 2  # farneback

    def process(
        self, frame: np.ndarray
    ) -> Tuple[List[TrackedBlob], List[TrackedBlob], np.ndarray, bool]:
        """
        Run the full pipeline on one frame.
        Returns: (confirmed, candidates, clean_mask, warming_up)
        """
        warming = self.frame_count < self.warmup_len

        gray    = self.preprocessor.process(frame)
        raw     = self.detector.apply(gray)

        if warming:
            confirmed, candidates = [], []
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            mask      = self.postproc.apply_morphology(raw)
            blobs     = self.postproc.extract_blobs(mask)
            confirmed, candidates = self.postproc.update_tracker(blobs)

        self.frame_count += 1
        return confirmed, candidates, mask, warming

    # ---------------------------------------------------------------- command string

    def command_lines(self, input_path: str) -> Tuple[str, str]:
        """
        Returns two display lines representing the equivalent CLI command.
        Line 1: base command (script, input, method, --display)
        Line 2: all tunable parameters
        """
        fname = Path(input_path).name
        line1 = (
            f"python ti_motion_detect.py  --input {fname}"
            f"  --method {self.method}  --display"
        )
        parts = []
        for p in self.pdefs:
            v = _real(p, self._tb(p))
            s = str(int(v)) if p.is_int else f"{v:.2f}"
            parts.append(f"{p.flag} {s}")
        line2 = "  " + "  ".join(parts)
        return line1, line2


# ============================================================
# Window / trackbar setup
# ============================================================

def create_window(
    win_name: str, method: str, x: int, y: int, w: int, h: int
) -> None:
    """Create an OpenCV window at (x, y), sized w×h, with method's trackbars."""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)
    cv2.moveWindow(win_name, x, y)
    for p in METHOD_PARAMS[method]:
        cv2.createTrackbar(p.label, win_name, p.tdef, p.tmax, lambda _: None)


# ============================================================
# Overlay helpers
# ============================================================

def _scale_blobs(blobs: List[TrackedBlob], sx: float, sy: float) -> List[TrackedBlob]:
    """
    Return a scaled copy of tracked blobs so bounding boxes drawn on the
    display-scaled frame align correctly when detection ran on full-res.
    """
    out = []
    for b in blobs:
        x, y, bw, bh = b.bbox
        out.append(TrackedBlob(
            blob_id      = b.blob_id,
            centroid     = (b.centroid[0] * sx, b.centroid[1] * sy),
            bbox         = (int(x * sx), int(y * sy),
                            max(1, int(bw * sx)), max(1, int(bh * sy))),
            frames_seen  = b.frames_seen,
            frames_absent= b.frames_absent,
            confirmed    = b.confirmed,
        ))
    return out


def draw_mask_inset(
    frame: np.ndarray, mask: np.ndarray, frac: float = 0.22
) -> np.ndarray:
    """Stamp a small greyscale mask preview into the top-right corner."""
    h, w = frame.shape[:2]
    mh   = max(1, int(h * frac))
    mw   = max(1, int(w * frac))
    small = cv2.resize(mask, (mw, mh))
    bgr   = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (mw - 1, mh - 1), (70, 70, 70), 1)
    cv2.putText(bgr, "MASK", (3, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    out = frame.copy()
    out[0:mh, w - mw:w] = bgr
    return out


def draw_cmd_bar(
    frame: np.ndarray, line1: str, line2: str
) -> np.ndarray:
    """
    Draw a two-line semi-transparent command bar at the bottom of the frame.
    The text uses the smallest legible size so it doesn't eat too much space.
    """
    img    = frame.copy()
    h, w   = img.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.28, min(w, h) / 1400.0)
    lh     = int(fscale * 26) + 3
    bar_h  = lh * 2 + 8

    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.70, img, 0.30, 0)

    color = (140, 215, 140)
    cv2.putText(img, line1, (4, h - bar_h + lh),
                font, fscale, color, 1, cv2.LINE_AA)
    cv2.putText(img, line2, (4, h - bar_h + lh * 2),
                font, fscale, color, 1, cv2.LINE_AA)
    return img


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 4-method side-by-side motion detection tuner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input video file")
    parser.add_argument("--scale",        type=float, default=0.5,
                        help="Display scale factor (0.5 = half resolution). "
                             "Use 0.4 on 1080p screens to fit all 4 windows.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"[ERROR] File not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    disp_w = max(1, int(src_w * args.scale))
    disp_h = max(1, int(src_h * args.scale))
    sx     = args.scale   # blob coordinate scale factors
    sy     = args.scale

    print(f"\n  Video  : {input_path.name}")
    print(f"  Source : {src_w}×{src_h}  @  {fps_in:.1f} fps  ({total} frames)")
    print(f"  Display: {disp_w}×{disp_h} per window  (scale {args.scale})")
    print()
    print("  Controls")
    print("  --------")
    print("  q / Esc   Quit")
    print("  r         Restart video + reset all background models")
    print("  Space     Pause / resume")
    print()
    print("  Warmup note: MOG2/KNN suppress detections for the first")
    print("  history//2 frames while the background model initialises.")
    print("  The model re-warms automatically on video loop or after r.")
    print()

    # ── Estimate trackbar control-area height for window positioning ─────────
    # Each OpenCV trackbar is ~28 px; title bar ~30 px.
    max_tb     = max(len(v) for v in METHOD_PARAMS.values())
    ctrl_h_est = max_tb * 28 + 30
    gap        = 6

    # Window names (double as unique OpenCV identifiers)
    WIN = {m: f"[{m.upper()}]  q=quit  r=reset  Space=pause" for m in METHODS}

    # 2×2 grid layout
    # MOG2 → top-left   KNN → top-right
    # Diff → bot-left   Farneback → bot-right
    grid = [
        (0,                 0),
        (disp_w + gap,      0),
        (0,                 disp_h + ctrl_h_est + gap),
        (disp_w + gap,      disp_h + ctrl_h_est + gap),
    ]

    # Create windows + trackbars (must happen before PipelineInstance reads them)
    for i, method in enumerate(METHODS):
        create_window(WIN[method], method, grid[i][0], grid[i][1], disp_w, disp_h)

    # Build pipeline instances (they read trackbars which now exist with defaults)
    pipes: Dict[str, PipelineInstance] = {}
    for method in METHODS:
        p = PipelineInstance(method, WIN[method])
        p.build_all()
        # Seed the cache so the first sync() call doesn't trigger a spurious rebuild
        p._cache = {param.flag: param.tdef for param in p.pdefs}
        pipes[method] = p

    # ── Main loop ────────────────────────────────────────────────────────────
    frame_idx = 0
    paused    = False
    fps_live  = 0.0
    fps_cnt   = 0
    t_fps     = time.time()

    def _restart() -> None:
        """Seek video to start and reset all pipeline background models."""
        nonlocal frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        for p in pipes.values():
            p.reset()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):          # q or Esc → quit
            break
        if key == ord("r"):                # r → restart
            _restart()
            print("\r[RESET]  Video restarted, background models cleared.       ", end="")
            continue
        if key == ord(" "):                # Space → pause/resume
            paused = not paused
            print(f"\r[{'PAUSED  ' if paused else 'RESUMED '}]                  ", end="")

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            # End of video → auto-loop
            _restart()
            print("\r[LOOP]   Video restarted.                                   ", end="")
            continue

        # FPS measurement
        fps_cnt += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps_live = fps_cnt / (now - t_fps)
            fps_cnt  = 0
            t_fps    = now

        # Display-scaled frame (for rendering only; detection runs full-res)
        disp_frame = cv2.resize(frame, (disp_w, disp_h))

        # ── Per-method pipeline ──────────────────────────────────────────────
        for method in METHODS:
            pipe = pipes[method]

            # 1. Detect trackbar changes → rebuild affected stages
            pipe.sync()

            # 2. Detection on full-resolution frame (accuracy matters)
            confirmed, candidates, mask, warming = pipe.process(frame)

            # 3. Scale blob coordinates to the display resolution
            confirmed_d  = _scale_blobs(confirmed,  sx, sy)
            candidates_d = _scale_blobs(candidates, sx, sy)

            # 4. Annotate the display-scaled frame
            annotated = pipe.vis.draw_boxes(disp_frame.copy(), confirmed_d, candidates_d)
            annotated = pipe.vis.draw_hud(
                annotated, frame_idx, fps_live,
                method.upper(), len(candidates), len(confirmed), warming,
            )

            # 5. Mask inset (top-right corner; resize mask to display dims first)
            mask_disp = cv2.resize(mask, (disp_w, disp_h))
            annotated = draw_mask_inset(annotated, mask_disp)

            # 6. Command bar (bottom two lines)
            l1, l2 = pipe.command_lines(str(input_path))
            annotated = draw_cmd_bar(annotated, l1, l2)

            cv2.imshow(WIN[method], annotated)

        frame_idx += 1

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("\n\nTuner closed.")


if __name__ == "__main__":
    main()
