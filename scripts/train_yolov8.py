#!/usr/bin/env python
"""Train YOLOv8x with metrics logging, checkpoints, and early stopping.

Example:
  python scripts/train_yolov8.py --data "D:\Projects\Human Movement Detection\yolo_split\data.yaml" --epochs 100 --batch 16 --img 640 --project runs --name yolo8x_humans_cars
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)):
            flat[k] = v
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8x training with CSV metrics and checkpoints")
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", type=str, default="yolov8x_humans_cars")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Optional directory to store checkpoint copies",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--save_period", type=int, default=5, help="Checkpoint save period (epochs)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    # LR schedule
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.1, help="Final LR factor (lr0 * lrf)")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in the run directory",
    )
    # Augmentation controls (YOLOv8)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--fliplr", type=float, default=0.5)

    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: data.yaml not found: {args.data}")
        return 2

    project_dir = args.project
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir = project_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    ckpt_dir = args.ckpt_dir or (run_dir / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_exists = metrics_csv.exists()
    csv_file = metrics_csv.open("a" if csv_exists else "w", newline="", encoding="utf-8")
    writer = None

    def on_fit_epoch_end(trainer):
        nonlocal writer
        metrics = _flatten_metrics(trainer.metrics)
        # Add epoch index
        metrics["epoch"] = int(trainer.epoch)

        if writer is None:
            headers = sorted(metrics.keys())
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            if not csv_exists:
                writer.writeheader()

        writer.writerow(metrics)
        csv_file.flush()

        # Copy checkpoint each epoch (resume-capable)
        epoch = int(trainer.epoch)
        weights_dir = Path(trainer.save_dir) / "weights"
        last_pt = weights_dir / "last.pt"
        if last_pt.exists():
            dest = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
            dest.write_bytes(last_pt.read_bytes())
        if args.save_period and args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            best_pt = weights_dir / "best.pt"
            if best_pt.exists():
                best_dest = ckpt_dir / "best.pt"
                if not best_dest.exists():
                    best_dest.write_bytes(best_pt.read_bytes())

    model = YOLO("yolov8x.pt")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        resume=args.resume,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        fliplr=args.fliplr,
        # disable other augmentations
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        flipud=0.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
    )

    csv_file.close()
    print(f"Metrics saved to: {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
