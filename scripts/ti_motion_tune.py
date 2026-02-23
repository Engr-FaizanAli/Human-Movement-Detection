#!/usr/bin/env python3
"""
TI Motion Detection — Interactive Parameter Tuner
==================================================
Streamlit app: load a video, scrub through frames, and tune ALL motion
detection parameters in real-time with instant visual feedback.

4-panel view per frame:
  [Original] | [Preprocessed] | [Raw Mask] | [Final Detections]

The "Raw Mask coverage %" metric tells you immediately whether the problem
is the background model (too noisy at the source) or the post-processing
filters (not filtering enough).

Run:
    streamlit run scripts/ti_motion_tune.py
    streamlit run scripts/ti_motion_tune.py --server.port 8502

Then open  http://localhost:8501  in your browser.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Headless fix — must be before cv2
if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") is None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import streamlit as st

# ── import the pipeline classes from ti_motion_detect ──────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from ti_motion_detect import (
    Blob, MaskPostprocessor, MotionDetector,
    MotionVisualizer, TIPreprocessor, TrackedBlob,
)

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

MAX_FRAMES = 700   # cap memory usage — plenty for tuning

def _bg_key(path, method, history, threshold, diff_frames,
            use_clahe, clahe_clip, clahe_tile, use_temporal, alpha) -> str:
    """Hash of background-model parameters — used to detect when re-processing is needed."""
    raw = f"{path}|{method}|{history}|{threshold}|{diff_frames}|" \
          f"{use_clahe}|{clahe_clip}|{clahe_tile}|{use_temporal}|{alpha}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def load_video(path: str) -> Tuple[Optional[List], float, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0, 0, 0
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(min(tot, MAX_FRAMES)):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps, w, h


def build_background_model(
    frames, method, history, threshold, diff_frames,
    use_clahe, clahe_clip, clahe_tile, use_temporal, alpha,
    progress_placeholder,
) -> Tuple[List, List]:
    """
    Process every frame through preprocessor + motion detector.
    Returns (preprocessed_frames, raw_masks).
    Cached in st.session_state — only re-runs when bg params change.
    """
    preprocessor = TIPreprocessor(
        use_clahe=use_clahe, clahe_clip=clahe_clip, clahe_tile=clahe_tile,
        use_temporal=use_temporal, alpha=alpha,
    )
    detector = MotionDetector(
        method=method, history=history,
        threshold=threshold, diff_frames=diff_frames,
    )

    preprocessed: List[np.ndarray] = []
    raw_masks:    List[np.ndarray] = []
    n = len(frames)
    pbar = progress_placeholder.progress(0, text="Building background model…")

    for i, frame in enumerate(frames):
        pre  = preprocessor.process(frame)
        mask = detector.apply(pre)
        preprocessed.append(pre)
        raw_masks.append(mask)
        if i % 30 == 0:
            pbar.progress(i / n, text=f"Building background model… {i}/{n} frames")

    pbar.progress(1.0, text="Done!")
    time.sleep(0.25)
    progress_placeholder.empty()
    return preprocessed, raw_masks


def get_frame_result(
    frame_idx: int,
    frames: List,
    raw_masks: List,
    warmup: int,
    morph_close: int,
    morph_open: int,
    min_area: int,
    max_area: int,
    min_solidity: float,
    persistence: int,
) -> Tuple[List[TrackedBlob], List[TrackedBlob], np.ndarray]:
    """
    Run the temporal persistence tracker over a window ending at frame_idx
    so the confirmed/candidate state is realistic.
    Returns (confirmed, candidates, clean_mask_at_frame_idx).
    """
    window_start = max(warmup, frame_idx - persistence * 4)

    postproc = MaskPostprocessor(
        morph_close=morph_close, morph_open=morph_open,
        min_area=min_area, max_area=max_area,
        min_solidity=min_solidity, persistence=persistence,
    )

    clean_mask_out: np.ndarray = np.zeros_like(raw_masks[0])
    for i in range(window_start, frame_idx + 1):
        cm    = postproc.apply_morphology(raw_masks[i])
        blobs = postproc.extract_blobs(cm)
        postproc.update_tracker(blobs)
        if i == frame_idx:
            clean_mask_out = cm

    confirmed  = [t for t in postproc._tracked.values() if t.confirmed]
    candidates = [t for t in postproc._tracked.values() if not t.confirmed]
    return confirmed, candidates, clean_mask_out


def export_full_video(
    frames, raw_masks, warmup, fps, video_path,
    morph_close, morph_open, min_area, max_area, min_solidity, persistence,
    progress_placeholder,
) -> Path:
    """Process and write the full annotated video. Returns output path."""
    out_path = Path(video_path).parent / f"{Path(video_path).stem}_tuned.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    postproc = MaskPostprocessor(
        morph_close=morph_close, morph_open=morph_open,
        min_area=min_area, max_area=max_area,
        min_solidity=min_solidity, persistence=persistence,
    )
    vis  = MotionVisualizer(persistence=persistence)
    n    = len(frames)
    pbar = progress_placeholder.progress(0, text="Exporting video…")

    for i, frame in enumerate(frames):
        warming_up = i < warmup
        if warming_up:
            confirmed, candidates = [], []
            annotated = frame.copy()
        else:
            cm        = postproc.apply_morphology(raw_masks[i])
            blobs     = postproc.extract_blobs(cm)
            confirmed, candidates = postproc.update_tracker(blobs)
            annotated = vis.draw_boxes(frame, confirmed, candidates)

        annotated = vis.draw_hud(
            annotated, i, fps, "TUNED",
            len(candidates), len(confirmed), warming_up,
        )
        writer.write(annotated)

        if i % 30 == 0:
            pbar.progress(i / n, text=f"Exporting… {i}/{n}")

    writer.release()
    pbar.progress(1.0, text="Export complete!")
    time.sleep(0.3)
    progress_placeholder.empty()
    return out_path


def build_cli_command(
    video_path, method, history, threshold, diff_frames,
    use_clahe, clahe_clip, clahe_tile, use_temporal, alpha,
    morph_close, morph_open, min_area, max_area, min_solidity, persistence,
) -> str:
    parts = [
        "python scripts/ti_motion_detect.py",
        f"--input  {video_path}",
        f"--method {method}",
        f"--history {history}",
        f"--threshold {threshold}",
        f"--diff-frames {diff_frames}",
        "--clahe"         if use_clahe    else "--no-clahe",
        f"--clahe-clip {clahe_clip}"  if use_clahe    else "",
        f"--clahe-tile {clahe_tile}"  if use_clahe    else "",
        "--temporal-smooth" if use_temporal else "--no-temporal-smooth",
        f"--alpha {alpha}"           if use_temporal else "",
        f"--morph-close {morph_close}",
        f"--morph-open {morph_open}",
        f"--min-area {min_area}",
        f"--max-area {max_area}",
        f"--min-solidity {min_solidity}",
        f"--persistence {persistence}",
    ]
    return " \\\n    ".join(p for p in parts if p)


# ────────────────────────────────────────────────────────────────────────────
# Main app
# ────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        layout="wide",
        page_title="TI Motion — Tuner",
        initial_sidebar_state="expanded",
    )

    st.title("TI Motion Detection — Interactive Parameter Tuner")
    st.caption(
        "Tune every knob in real-time. The **Raw Mask coverage %** tells you "
        "if your problem is the background model (source too noisy) "
        "or the post-processing filters (not filtering enough)."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:

        # ---- Video --------------------------------------------------------
        st.header("📁 Video")
        video_path = st.text_input("Video file path", value="test_motion_1.mp4",
                                   help="Absolute or relative path to your video")

        # ---- Background model params --------------------------------------
        st.markdown("---")
        st.header("🔍 Detection Method")
        st.caption("⚠️ Changing these requires rebuilding the background model")

        method = st.selectbox(
            "Method", ["mog2", "knn", "diff", "farneback"],
            help=(
                "**mog2** — Fast Gaussian mixture background model. Best default.  \n"
                "**knn** — K-Nearest Neighbours. More accurate for low-contrast TI.  \n"
                "**diff** — Simple N-frame difference. Useful as baseline.  \n"
                "**farneback** — Dense optical flow. Best for very far, tiny blobs."
            ),
        )
        history = st.slider(
            "History (frames)", 30, 600, 200, 10,
            help="Background model memory. 200 = ~8s at 25 fps. "
                 "Reduce to 50–100 if you see slow adaptation.",
        )
        threshold = st.slider(
            "Threshold", 1, 200, 25, 1,
            help=(
                "Foreground sensitivity.  \n"
                "**Regular compressed video**: start at 20–40.  \n"
                "**TI video (native)**: 8–16 is typical.  \n"
                "If raw mask is >20%% coverage → increase this first."
            ),
        )
        diff_frames = st.slider(
            "Diff frame gap (diff method only)", 1, 30, 5, 1,
            help="Frame lookback for the diff method. "
                 "Increase to 8–15 for slow far targets.",
        )

        st.markdown("---")
        st.header("🌡️ TI Preprocessing")
        st.caption("⚠️ Changing these also requires rebuilding")

        use_clahe = st.checkbox(
            "CLAHE enhancement", value=False,
            help="Enhances local contrast. "
                 "**Turn OFF for regular video** — CLAHE amplifies H.264 block artifacts. "
                 "Turn ON for native TI footage.",
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            clahe_clip = st.slider("Clip limit", 0.5, 8.0, 2.5, 0.5,
                                   disabled=not use_clahe)
        with col_c2:
            clahe_tile = st.slider("Tile size", 4, 128, 32, 4,
                                   disabled=not use_clahe,
                                   help="Larger tile = person affects smaller fraction of tile")

        use_temporal = st.checkbox(
            "Temporal IIR smoothing", value=False,
            help="Frame-blending noise reduction. "
                 "**Turn OFF for regular video** — often makes things worse on H.264.",
        )
        alpha = st.slider(
            "IIR alpha", 0.05, 0.95, 0.40, 0.05,
            disabled=not use_temporal,
            help="Weight of current frame. 0.2 = heavy smoothing, 0.8 = light smoothing.",
        )

        # Compute bg key to detect when rebuild is needed
        bg_key = _bg_key(
            video_path, method, history, threshold, diff_frames,
            use_clahe, clahe_clip, clahe_tile, use_temporal, alpha,
        )
        needs_rebuild = (
            "bg_key" not in st.session_state
            or st.session_state.bg_key != bg_key
            or "frames" not in st.session_state
        )

        rebuild_btn = st.button(
            "▶  Load Video + Build Background Model",
            type="primary",
            use_container_width=True,
            help="Run this after changing any parameter above, or on first load.",
        )

        # ---- Post-processing params (instant) ----------------------------
        st.markdown("---")
        st.header("🔬 Post-Processing  ✨ *instant*")
        st.caption("These update the view immediately — no rebuild needed")

        morph_close = st.slider(
            "Morphological closing kernel", 1, 21, 5, 2,
            help="Fills holes within blobs. Larger = more aggressive hole-filling.",
        )
        morph_open = st.slider(
            "Morphological opening kernel", 1, 21, 3, 2,
            help="Removes small noise specks. "
                 "⚠️ Keep ≤ morph-close to avoid destroying small blobs.",
        )

        st.markdown("**Blob filters**")
        min_area = st.slider(
            "Min blob area (px²)", 1, 20000, 500, 10,
            help=(
                "Minimum connected blob size to consider:  \n"
                "- Person at **1 km** (TI): ~150–600 px²  \n"
                "- Person at **2–3 km** (TI): 4–30 px²  \n"
                "- Person in **regular video, close range**: 500–5000 px²  \n"
                "Start high, reduce gradually until you detect the target."
            ),
        )
        max_area = st.slider(
            "Max blob area (px²)", 100, 200000, 80000, 500,
            help="Filters out sky, ground, large vegetation blobs.",
        )
        min_solidity = st.slider(
            "Min solidity", 0.0, 1.0, 0.30, 0.05,
            help="Blob compactness (area / convex hull area).  \n"
                 "Noise streaks ≈ 0.05–0.2.  \n"
                 "Humans ≈ 0.4–0.9.",
        )
        persistence = st.slider(
            "Persistence (frames to confirm)", 1, 15, 3, 1,
            help="Frames a blob must appear consecutively to become **confirmed (green)**.  \n"
                 "Thermal noise: 1–2 frames.  \n"
                 "Set to 3–5 to reject noise; reduce to 1–2 if targets are fast.",
        )

    # ── Main area ─────────────────────────────────────────────────────────────

    # Warn about needing rebuild
    if needs_rebuild and not rebuild_btn:
        if "frames" not in st.session_state:
            st.info("👈  Enter a video path and click **Load Video + Build Background Model** to start.")
        else:
            st.warning(
                "⚠️  Background model parameters changed.  "
                "Click **Load Video + Build Background Model** to apply them."
            )

    # Build background model
    if rebuild_btn:
        if not Path(video_path).is_file():
            st.error(f"Video not found: `{video_path}`")
            st.stop()

        ph = st.empty()
        with ph.container():
            st.info("Loading video…")
        frames, fps, w, h = load_video(video_path)
        ph.empty()

        if frames is None or len(frames) == 0:
            st.error("Could not open the video file.")
            st.stop()

        ph2 = st.empty()
        preprocessed, raw_masks = build_background_model(
            frames, method, history, threshold, diff_frames,
            use_clahe, clahe_clip, clahe_tile, use_temporal, alpha,
            ph2,
        )

        st.session_state.update({
            "frames":       frames,
            "preprocessed": preprocessed,
            "raw_masks":    raw_masks,
            "fps":          fps,
            "w":            w,
            "h":            h,
            "video_path":   video_path,
            "bg_key":       bg_key,
        })
        st.rerun()

    if "frames" not in st.session_state:
        st.stop()

    # Retrieve cached data
    frames       = st.session_state.frames
    preprocessed = st.session_state.preprocessed
    raw_masks    = st.session_state.raw_masks
    fps          = st.session_state.fps
    n_frames     = len(frames)
    warmup       = history // 2

    # ── Frame scrubber ───────────────────────────────────────────────────────
    st.markdown("---")
    default_frame = min(warmup + 20, n_frames - 1)
    frame_idx = st.slider(
        "Frame scrubber", 0, n_frames - 1, default_frame, 1,
        help="Drag to step through the video. Results update instantly.",
    )

    # Warmup notice
    if frame_idx < warmup:
        st.warning(
            f"Frame {frame_idx} is inside the warmup window "
            f"(first **{warmup}** frames). The background model is still "
            f"initialising — move the scrubber past frame **{warmup}** to see detections."
        )

    # ── 4-panel display ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # Panel 1 — original
    with col1:
        st.image(
            cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB),
            caption=f"Original  (frame {frame_idx})",
            use_container_width=True,
        )

    # Panel 2 — preprocessed
    with col2:
        pre = preprocessed[frame_idx]
        pre_disp = pre if pre.ndim == 2 else cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
        st.image(pre_disp, caption="Preprocessed (CLAHE + IIR)",
                 use_container_width=True, clamp=True)

    # Panel 3 — raw mask (colourised for visibility)
    raw_mask     = raw_masks[frame_idx]
    raw_coverage = float(np.mean(raw_mask > 0) * 100)
    with col3:
        mask_coloured = cv2.applyColorMap(raw_mask, cv2.COLORMAP_HOT)
        st.image(
            cv2.cvtColor(mask_coloured, cv2.COLOR_BGR2RGB),
            caption=f"Raw Motion Mask  ({raw_coverage:.1f}% coverage)",
            use_container_width=True,
        )

    # Panel 4 — final detections
    if frame_idx < warmup:
        confirmed, candidates, clean_mask = [], [], np.zeros_like(raw_mask)
        annotated = frames[frame_idx].copy()
    else:
        confirmed, candidates, clean_mask = get_frame_result(
            frame_idx, frames, raw_masks, warmup,
            morph_close, morph_open, min_area, max_area, min_solidity, persistence,
        )
        vis = MotionVisualizer(persistence=persistence)
        annotated = vis.draw_boxes(frames[frame_idx], confirmed, candidates)
        annotated = vis.draw_hud(
            annotated, frame_idx, fps, method.upper(),
            len(candidates), len(confirmed), False,
        )

    clean_coverage = float(np.mean(clean_mask > 0) * 100)
    with col4:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=(
                f"Detections  ✅ {len(confirmed)} confirmed   "
                f"🟡 {len(candidates)} candidates"
            ),
            use_container_width=True,
        )

    # ── Metrics row ──────────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("✅ Confirmed",     len(confirmed))
    m2.metric("🟡 Candidates",    len(candidates))
    m3.metric("Raw mask %",       f"{raw_coverage:.1f}%",
              help="% of pixels the background subtractor flagged as foreground")
    m4.metric("Clean mask %",     f"{clean_coverage:.1f}%",
              help="% after morphological filtering")
    m5.metric("Frame",            f"{frame_idx} / {n_frames-1}")
    m6.metric("Warmup ends",      f"frame {warmup}")

    # ── Diagnostic guidance ───────────────────────────────────────────────────
    if raw_coverage > 30:
        st.error(
            f"🚨 **Raw mask is {raw_coverage:.0f}%** — the background subtractor is flagging "
            f"almost the entire frame. No post-processing filter can fix this downstream.  \n"
            f"**Root causes & fixes:**  \n"
            f"- **Threshold too low** → increase the Threshold slider (try 40–80 for compressed video)  \n"
            f"- **CLAHE ON with compressed video** → turn CLAHE OFF  \n"
            f"- **Camera is moving / shaking** → background subtraction assumes a fixed camera  \n"
            f"- **Video has lighting changes** → reduce History to 50–100 frames for faster adaptation  \n"
            f"- **IIR smoothing ON** → turn it OFF and see if the raw mask improves"
        )
    elif raw_coverage > 10:
        st.warning(
            f"⚠️ **Raw mask is {raw_coverage:.1f}%** — moderately noisy. "
            f"If you expect only 1–2 moving targets, try increasing Threshold to {int(threshold * 1.5)}.  \n"
            f"If clean mask % is much lower, your morphological filters are working."
        )
    elif raw_coverage > 0:
        st.success(
            f"✅ Raw mask coverage: **{raw_coverage:.1f}%** — looks healthy. "
            f"Use the post-processing sliders to isolate the target blobs."
        )

    # ── Export section ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Export")

    tab_cli, tab_video = st.tabs(["CLI Command", "Export Full Video"])

    with tab_cli:
        st.caption("Copy this command to reproduce the current settings with the CLI script:")
        cli_cmd = build_cli_command(
            video_path, method, history, threshold, diff_frames,
            use_clahe, clahe_clip, clahe_tile, use_temporal, alpha,
            morph_close, morph_open, min_area, max_area, min_solidity, persistence,
        )
        st.code(cli_cmd, language="bash")

    with tab_video:
        st.caption(
            "Writes the full video using the current post-processing parameters "
            "(uses the already-built background model, so this is fast)."
        )
        if st.button("Export Full Video  ▶", type="primary"):
            ph_export = st.empty()
            out_path = export_full_video(
                frames, raw_masks, warmup, fps,
                st.session_state.video_path,
                morph_close, morph_open, min_area, max_area, min_solidity, persistence,
                ph_export,
            )
            st.success(f"Saved → `{out_path}`")


if __name__ == "__main__":
    main()
