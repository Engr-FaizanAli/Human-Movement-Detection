#!/usr/bin/env python
"""Run YOLOv8 inference on images or a folder.

Examples:
  python scripts/infer_yolov8.py --model yolov8x.pt --source "D:\data\images" --output "D:\data\pred"
  python scripts/infer_yolov8.py --model "runs\yolov8x_humans_cars\weights\best.pt" --source "D:\data\images" --output "D:\data\pred"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8 inference for humans and cars")
    parser.add_argument("--model", type=Path, default=Path("yolov8x.pt"), help="Model path (.pt)")
    parser.add_argument("--source", required=True, type=Path, help="Image file or folder")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for predictions")
    parser.add_argument("--img", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format txt predictions")

    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        return 2
    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    # COCO: person=0, car=2
    model.predict(
        source=str(args.source),
        imgsz=args.img,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=[0, 2],
        project=str(args.output),
        name="pred",
        save=True,
        save_txt=args.save_txt,
    )

    print(f"Predictions saved under: {args.output}\\pred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
