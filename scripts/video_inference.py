#!/usr/bin/env python3
"""
Video YOLO Inference with Tracking
====================================
Run any YOLO model on video files with ByteTrack/BotSort tracking.
Outputs annotated video with bounding boxes and persistent track IDs.

Usage:
    # Single video
    python video_inference.py --model path/to/best.pt --input video.mp4

    # Folder of videos
    python video_inference.py --model path/to/best.pt --input ./videos/

    # Custom settings
    python video_inference.py --model best.pt --input video.mp4 --conf 0.4 --output ./results/

    # Use BotSort tracker (more accurate, slower)
    python video_inference.py --model best.pt --input video.mp4 --tracker botsort.yaml
"""

import argparse
import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def generate_colors(n: int):
    """Generate n visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


def draw_boxes(image: np.ndarray, boxes, class_names: dict, class_colors: list, track_colors: list) -> np.ndarray:
    """Draw bounding boxes with class name, confidence, and track ID.
    Colors by track ID so each object has a consistent color across frames.
    """
    img = image.copy()
    if boxes is None or len(boxes) == 0:
        return img

    h, w = img.shape[:2]
    line_width = max(2, int(min(h, w) * 0.002))
    font_scale = max(0.4, min(h, w) / 1800)
    font_thickness = max(1, line_width // 2)

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        track_id = int(box.id[0]) if box.id is not None else None

        # Color by track ID when available (consistent per object), else by class
        if track_id is not None:
            color = track_colors[track_id % len(track_colors)]
            label = f"#{track_id} {class_names.get(cls_id, cls_id)} {conf:.2f}"
        else:
            color = class_colors[cls_id % len(class_colors)]
            label = f"{class_names.get(cls_id, cls_id)} {conf:.2f}"

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 6, label_y1 + th + 10), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 3, label_y1 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return img


def process_video(model, video_path: Path, output_dir: Path, conf: float, iou: float, tracker: str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Could not open {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{video_path.stem}_tracked.mp4"
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    class_colors = generate_colors(len(model.names))   # one color per class
    track_colors = generate_colors(100)                 # 100 colors for track IDs

    print(f"  {total_frames} frames  |  {fps:.1f} fps  |  {width}×{height}")

    frame_idx        = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            conf=conf,
            iou=iou,
            tracker=tracker,
            persist=True,   # keeps tracker state across frames
            verbose=False,
        )
        result = results[0]
        n_det  = len(result.boxes) if result.boxes is not None else 0
        total_detections += n_det

        annotated = draw_boxes(frame, result.boxes, model.names, class_colors, track_colors)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  [{pct:5.1f}%] Frame {frame_idx}/{total_frames}  detections: {n_det}    ", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Saved -> {out_path}")
    print(f"  Total detections across all frames: {total_detections}")
    return total_detections


def run_inference(model_path: str, input_path: str, output_dir: str, conf: float, iou: float, tracker: str):
    from ultralytics import YOLO

    model = YOLO(model_path)

    print(f"Model:   {model_path}")
    print(f"Classes: {model.names}")
    print(f"Conf:    {conf}  |  IoU: {iou}  |  Tracker: {tracker}")
    print()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
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
        process_video(model, video_path, output_dir, conf, iou, tracker)
        print()

    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO video inference with ByteTrack/BotSort tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",   "-m", required=True,                   help="Path to YOLO .pt model")
    parser.add_argument("--input",   "-i", required=True,                   help="Input video file or folder")
    parser.add_argument("--output",  "-o", default="./video_output",        help="Output directory")
    parser.add_argument("--conf",          type=float, default=0.4,         help="Confidence threshold")
    parser.add_argument("--iou",           type=float, default=0.45,        help="IoU threshold for NMS")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="bytetrack.yaml = faster  |  botsort.yaml = more accurate (uses ReID features)",
    )

    args = parser.parse_args()
    run_inference(args.model, args.input, args.output, args.conf, args.iou, args.tracker)


if __name__ == "__main__":
    main()
