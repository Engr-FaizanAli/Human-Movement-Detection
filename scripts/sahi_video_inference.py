#!/usr/bin/env python3
"""
SAHI Video Inference Pipeline
==============================
Slicing Aided Hyper Inference (SAHI) for detecting small/distant objects
in thermal surveillance footage. Critical for humans/vehicles at 2-5km range
where objects appear as 4-10 pixel blobs in the full frame.

How SAHI works:
    Each frame is sliced into overlapping 640×640 patches.
    YOLO runs on each patch independently.
    A 6px human in the full frame = ~30px in a slice → detectable.
    Detections are merged back to original frame coordinates via NMS.

Install dependencies:
    pip install sahi supervision

Usage:
    # Basic (default slice=640, overlap=0.3)
    python sahi_video_inference.py --model best.pt --input video.mp4

    # Aggressive slicing for very distant objects (4-5km)
    python sahi_video_inference.py --model best.pt --input video.mp4 --slice 320 --overlap 0.4

    # Lower confidence threshold (recommended for distant/blurry detections)
    python sahi_video_inference.py --model best.pt --input video.mp4 --conf 0.2 --slice 320

    # Detection only, no tracking
    python sahi_video_inference.py --model best.pt --input video.mp4 --no-track

    # Folder of videos
    python sahi_video_inference.py --model best.pt --input ./videos/ --output ./sahi_results/
"""

import argparse
import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_colors(n: int):
    """Generate n visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def draw_detections(
    image: np.ndarray,
    xyxy: np.ndarray,
    class_ids: np.ndarray,
    confs: np.ndarray,
    track_ids,          # np.ndarray or None
    class_names: dict,
    class_colors: list,
    track_colors: list,
) -> np.ndarray:
    img = image.copy()
    if len(xyxy) == 0:
        return img

    h, w = img.shape[:2]
    line_width   = max(2, int(min(h, w) * 0.002))
    font_scale   = max(0.4, min(h, w) / 1800)
    font_thick   = max(1, line_width // 2)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        cls_id  = int(class_ids[i])
        conf    = float(confs[i])
        tid     = int(track_ids[i]) if track_ids is not None else None

        if tid is not None:
            color = track_colors[tid % len(track_colors)]
            label = f"#{tid} {class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        else:
            color = class_colors[cls_id % len(class_colors)]
            label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ly1 = max(y1 - th - 10, 0)
        cv2.rectangle(img, (x1, ly1), (x1 + tw + 6, ly1 + th + 10), color, -1)
        cv2.putText(img, label, (x1 + 3, ly1 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

    return img


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_video(
    sahi_model,
    video_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    slice_size: int,
    overlap: float,
    use_tracker: bool,
    class_names: dict,
    class_colors: list,
    track_colors: list,
) -> None:
    from sahi.predict import get_sliced_prediction

    # --- optional tracker ---
    tracker = None
    if use_tracker:
        try:
            import supervision as sv
            tracker = sv.ByteTrack()
        except ImportError:
            print("  [WARN] supervision not installed — running without tracking.")
            print("         Install with: pip install supervision")

    # --- open video ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {video_path.name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{video_path.stem}_sahi.mp4"
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    cols = max(1, int(width  / (slice_size * (1.0 - overlap))))
    rows = max(1, int(height / (slice_size * (1.0 - overlap))))
    print(f"  {total_frames} frames  |  {fps:.1f} fps  |  {width}×{height}")
    print(f"  Slices per frame: ~{cols * rows}  (slice={slice_size}px, overlap={overlap})")
    if tracker:
        print("  Tracker: ByteTrack ON")

    frame_idx        = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- SAHI sliced prediction ---
        result = get_sliced_prediction(
            frame,
            sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=True,     # also run full-frame pass for closer objects
            postprocess_match_threshold=iou,
            verbose=0,
        )

        preds = result.object_prediction_list
        n_det = len(preds)
        total_detections += n_det

        if n_det > 0:
            xyxy      = np.array([[*p.bbox.to_xyxy()] for p in preds], dtype=float)
            confs_arr = np.array([p.score.value for p in preds],       dtype=float)
            cls_arr   = np.array([p.category.id  for p in preds],      dtype=int)
            track_ids = None

            if tracker is not None:
                import supervision as sv
                detections = sv.Detections(xyxy=xyxy, confidence=confs_arr, class_id=cls_arr)
                tracked    = tracker.update_with_detections(detections)

                if len(tracked) > 0:
                    xyxy      = tracked.xyxy
                    cls_arr   = tracked.class_id   if tracked.class_id   is not None else cls_arr
                    confs_arr = tracked.confidence if tracked.confidence is not None else confs_arr
                    track_ids = tracked.tracker_id

            annotated = draw_detections(
                frame, xyxy, cls_arr, confs_arr, track_ids,
                class_names, class_colors, track_colors,
            )
        else:
            annotated = frame

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  [{pct:5.1f}%]  frame {frame_idx}/{total_frames}  dets: {n_det}    ", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Saved  -> {out_path}")
    print(f"  Total detections across all frames: {total_detections}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAHI video inference — sliced detection for small/distant thermal objects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    "-m", required=True,              help="Path to YOLO .pt model")
    parser.add_argument("--input",    "-i", required=True,              help="Input video file or folder")
    parser.add_argument("--output",   "-o", default="./sahi_output",   help="Output directory")
    parser.add_argument("--conf",           type=float, default=0.25,  help="Confidence threshold (go as low as 0.15 for extreme range)")
    parser.add_argument("--iou",            type=float, default=0.45,  help="IoU NMS threshold for merging slice detections")
    parser.add_argument("--slice",          type=int,   default=640,   help="Slice size in pixels. Use 320 for objects at 4-5km")
    parser.add_argument("--overlap",        type=float, default=0.3,   help="Slice overlap ratio (0.2–0.5). Higher = fewer missed edge objects")
    parser.add_argument("--no-track",       action="store_true",       help="Disable ByteTrack (detection only)")
    parser.add_argument("--device",         default="cuda",            help="Inference device: cuda or cpu")

    args = parser.parse_args()

    # --- check SAHI ---
    try:
        from sahi import AutoDetectionModel
    except ImportError:
        print("[ERROR] SAHI not installed. Run:  pip install sahi")
        sys.exit(1)

    # --- load model twice: once for class names, once wrapped in SAHI ---
    try:
        from ultralytics import YOLO
        _yolo       = YOLO(args.model)
        class_names = _yolo.names
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    class_colors = generate_colors(len(class_names))
    track_colors = generate_colors(100)

    print(f"Model:   {args.model}")
    print(f"Classes: {class_names}")
    print(f"Conf: {args.conf}  |  IoU: {args.iou}  |  Slice: {args.slice}px  |  Overlap: {args.overlap}")
    print()

    # --- collect input files ---
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    if not files:
        print(f"[ERROR] No supported video files found. Supported: {SUPPORTED_EXTENSIONS}")
        sys.exit(1)

    print(f"Found {len(files)} video(s) to process.\n")

    for video_path in files:
        print(f"--- {video_path.name} ---")
        process_video(
            sahi_model, video_path, output_dir,
            args.conf, args.iou, args.slice, args.overlap,
            not args.no_track,
            class_names, class_colors, track_colors,
        )
        print()

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
