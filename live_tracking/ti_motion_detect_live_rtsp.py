#!/usr/bin/env python3
"""
Live RTSP motion detection using the TI Motion Detection v2 pipeline components.

This runner reuses the processing classes from motion_scripts/ti_motion_detect_v2.py
but accepts a live stream URL as input.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import cv2

# Current line
cap = cv2.VideoCapture(rtsp_url)

# Change to force TCP
cap = cv2.VideoCapture(f"{rtsp_url}?tcp")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_scripts.ti_motion_detect_v2 import (  # noqa: E402
    CameraMotionSensor,
    CameraStabilizer,
    MaskPostprocessor,
    MotionDetector,
    MotionVisualizer,
    TIPreprocessor,
)


DEFAULT_RTSP = f"rtsp://{quote('admin', safe='')}:{quote('Tlgcctv@786', safe='')}@192.168.1.121:554/cam/realmonitor?channel=1&subtype=0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live RTSP TI motion detection (v2 pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", default=DEFAULT_RTSP, help="RTSP/HTTP stream URL")
    p.add_argument("--display", action="store_true", default=True, help="Show live window")
    p.add_argument("--no-display", dest="display", action="store_false", help="Disable live window")
    p.add_argument("--output", "-o", default=None, help="Optional output video path")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--method", default="mog2", choices=["mog2", "knn", "diff", "farneback"])
    p.add_argument("--history", type=int, default=200)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--diff-frames", type=int, default=5)
    p.add_argument("--freeze-after-warmup", action="store_true", default=False)

    p.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument("--clahe-clip", type=float, default=2.5)
    p.add_argument("--clahe-tile", type=int, default=12)
    p.add_argument("--temporal-smooth", dest="temporal_smooth", action="store_true", default=True)
    p.add_argument("--no-temporal-smooth", dest="temporal_smooth", action="store_false")
    p.add_argument("--alpha", type=float, default=0.4)

    p.add_argument("--morph-close", type=int, default=5)
    p.add_argument("--morph-open", type=int, default=3)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--max-area", type=int, default=5000)
    p.add_argument("--min-solidity", type=float, default=0.3)
    p.add_argument("--persistence", type=int, default=12)
    p.add_argument("--min-displacement", type=float, default=40.0)
    p.add_argument("--min-density", type=float, default=0.0)
    p.add_argument("--spatial-tol", type=int, default=15)
    p.add_argument("--max-absent", type=int, default=2)
    p.add_argument("--no-candidates", dest="show_candidates", action="store_false", default=True)

    p.add_argument("--stabilize", action="store_true", default=False)
    p.add_argument("--stabilize-radius", type=int, default=30)
    p.add_argument("--ptz-aware", action="store_true", default=False)
    p.add_argument("--ptz-motion-thresh", type=float, default=5.0)
    p.add_argument("--ptz-settle-time", type=float, default=5.0)
    p.add_argument("--reconnect-delay", type=float, default=2.0)
    return p.parse_args()


def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


def maybe_open_writer(path: Optional[str], fps: float, width: int, height: int) -> Optional[cv2.VideoWriter]:
    if not path:
        return None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"[WARN] Cannot create output video: {out_path}")
        return None
    print(f"Output video: {out_path}")
    return writer


def main() -> None:
    args = parse_args()
    print(f"Live input : {args.input}")

    cap = None
    writer = None
    frame_idx = 0
    live_fps = 0.0
    fps_counter = 0
    t_fps = time.time()
    prev_ptz_state: Optional[str] = None
    empty_mask = None

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
    stabilizer = CameraStabilizer(smooth_radius=args.stabilize_radius) if args.stabilize else None
    ptz_sensor = CameraMotionSensor(motion_thresh=args.ptz_motion_thresh, settle_frames=125) if args.ptz_aware else None

    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = open_stream(args.input)
            if not cap.isOpened():
                print(f"[WARN] Could not open stream. Retrying in {args.reconnect_delay:.1f}s...")
                time.sleep(args.reconnect_delay)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            settle_frames = max(1, int(args.ptz_settle_time * fps_in))
            if ptz_sensor is not None:
                ptz_sensor = CameraMotionSensor(
                    motion_thresh=args.ptz_motion_thresh,
                    settle_frames=settle_frames,
                )
            warmup_frames = args.history // 2
            out_w = max(1, int(width * args.scale))
            out_h = max(1, int(height * args.scale))
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            writer = maybe_open_writer(args.output, fps_in, out_w, out_h)
            print(f"Connected  : {width}x{height} @ {fps_in:.1f} FPS")

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed. Reconnecting...")
            cap.release()
            cap = None
            continue

        ptz_state = None
        learning_rate = -1.0

        if ptz_sensor is not None:
            raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ptz_state = ptz_sensor.update(raw_gray)
            if prev_ptz_state == CameraMotionSensor.MOVING and ptz_state == CameraMotionSensor.SETTLING:
                detector.reset()
                postproc.reset()
                preprocessor.reset()
            learning_rate = ptz_sensor.learning_rate
            prev_ptz_state = ptz_state

        if args.freeze_after_warmup and frame_idx >= warmup_frames:
            if ptz_state is None or ptz_state == CameraMotionSensor.ACTIVE:
                learning_rate = 0.0

        if stabilizer is not None:
            frame = stabilizer.stabilize(frame)

        preprocessed = preprocessor.process(frame)
        raw_mask = detector.apply(preprocessed, learning_rate=learning_rate)

        warming_up = frame_idx < warmup_frames
        active = (ptz_state is None or ptz_state == CameraMotionSensor.ACTIVE)

        if warming_up or not active:
            confirmed, candidates = [], []
            clean_mask = empty_mask
        else:
            clean_mask = postproc.apply_morphology(raw_mask)
            blobs = postproc.extract_blobs(clean_mask)
            confirmed, candidates = postproc.update_tracker(blobs)

        annotated = vis.draw_boxes(frame, confirmed, candidates if args.show_candidates else [])
        annotated = vis.draw_hud(
            annotated,
            frame_idx,
            live_fps,
            args.method.upper(),
            len(candidates),
            len(confirmed),
            warming_up,
            ptz_state=ptz_state,
            ptz_motion=ptz_sensor.last_motion if ptz_sensor else 0.0,
            ptz_settle_pct=ptz_sensor.settling_progress if ptz_sensor else 0.0,
        )

        out_frame = annotated
        if args.scale != 1.0:
            out_frame = cv2.resize(out_frame, (out_w, out_h))

        if writer is not None:
            writer.write(out_frame)

        if args.display:
            cv2.imshow("TI Motion v2 Live [q to quit]", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        fps_counter += 1
        now = time.time()
        if now - t_fps >= 1.0:
            live_fps = fps_counter / (now - t_fps)
            fps_counter = 0
            t_fps = now

        frame_idx += 1

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
