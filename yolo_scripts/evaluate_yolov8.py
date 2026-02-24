#!/usr/bin/env python3
"""
YOLOv8 Test-Set Evaluation
===========================
Computes mAP50, mAP50-95, Precision, Recall and saves the Confusion Matrix
against the held-out test split.

Usage (RunPod):
    python evaluate_yolov8.py

    # Override any default:
    python evaluate_yolov8.py \
        --model  /workspace/POD_OCR/runs/yolov8x_humans_v16/weights/best.pt \
        --data   /workspace/POD_OCR/yolo_split/data.yaml \
        --output /workspace/POD_OCR/eval_results \
        --imgsz 640 \
        --batch  16 \
        --conf   0.001 \
        --iou    0.6 \
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Defaults (match your RunPod layout)
# ---------------------------------------------------------------------------
DEFAULT_MODEL  = "/workspace/POD_OCR/runs/yolov8x_humans_v16/weights/best.pt"
DEFAULT_DATA   = "/workspace/POD_OCR/yolo_split/data.yaml"
DEFAULT_OUTPUT = "/workspace/POD_OCR/eval_results/yolov8x_humans_v16"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help="Path to best.pt")
    parser.add_argument("--data",   default=DEFAULT_DATA,   help="Path to data.yaml (must have a 'test:' key)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Directory to save results")
    parser.add_argument("--imgsz",  type=int,   default=640,   help="Inference image size")
    parser.add_argument("--batch",  type=int,   default=16,    help="Batch size")
    parser.add_argument("--conf",   type=float, default=0.001, help="Confidence threshold (keep low for mAP)")
    parser.add_argument("--iou",    type=float, default=0.6,   help="IoU threshold for NMS / mAP matching")
    parser.add_argument("--device", default="0",              help="CUDA device or 'cpu'")
    parser.add_argument("--workers", type=int,  default=8,    help="DataLoader workers")
    args = parser.parse_args()

    model_path  = Path(args.model)
    data_path   = Path(args.data)
    output_dir  = Path(args.output)

    # ---- sanity checks -------------------------------------------------------
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLOv8 Test-Set Evaluation")
    print("=" * 60)
    print(f"  Model  : {model_path}")
    print(f"  Data   : {data_path}")
    print(f"  Output : {output_dir}")
    print(f"  Split  : test")
    print(f"  imgsz  : {args.imgsz}  |  batch: {args.batch}")
    print(f"  conf   : {args.conf}  |  iou:   {args.iou}")
    print(f"  device : {args.device}")
    print("=" * 60)

    model = YOLO(str(model_path))

    # ---- run validation on the TEST split ------------------------------------
    metrics = model.val(
        data=str(data_path),
        split="test",               # uses the 'test:' key in data.yaml
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=args.workers,
        project=str(output_dir.parent),
        name=output_dir.name,
        plots=True,                 # saves confusion_matrix.png, PR curve, etc.
        save_json=False,
        verbose=True,
        exist_ok=True,
    )

    # ---- extract per-class and overall metrics --------------------------------
    box = metrics.box

    # Overall (mean across classes)
    precision_overall   = float(box.mp)     # mean precision
    recall_overall      = float(box.mr)     # mean recall
    map50_overall       = float(box.map50)  # mAP@0.50
    map50_95_overall    = float(box.map)    # mAP@0.50:0.95

    class_names = model.names  # {0: 'person', ...}

    # Per-class breakdown (if available)
    per_class: list[dict] = []
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for idx, cls_id in enumerate(box.ap_class_index):
            name = class_names.get(int(cls_id), str(cls_id))
            per_class.append({
                "class_id"  : int(cls_id),
                "class_name": name,
                "precision" : float(box.p[idx])      if box.p      is not None else None,
                "recall"    : float(box.r[idx])      if box.r      is not None else None,
                "mAP50"     : float(box.ap50[idx])   if box.ap50   is not None else None,
                "mAP50-95"  : float(box.ap[idx])     if box.ap     is not None else None,
            })

    # ---- print summary -------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS — Overall")
    print("=" * 60)
    print(f"  Precision  (mean) : {precision_overall:.4f}")
    print(f"  Recall     (mean) : {recall_overall:.4f}")
    print(f"  mAP@0.50         : {map50_overall:.4f}")
    print(f"  mAP@0.50:0.95    : {map50_95_overall:.4f}")

    if per_class:
        print()
        print("=" * 60)
        print("RESULTS — Per Class")
        print("=" * 60)
        header = f"  {'Class':<20} {'Prec':>8} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in per_class:
            p  = f"{row['precision']:.4f}" if row['precision']  is not None else "  N/A  "
            r  = f"{row['recall']:.4f}"    if row['recall']     is not None else "  N/A  "
            m5 = f"{row['mAP50']:.4f}"     if row['mAP50']      is not None else "  N/A  "
            m9 = f"{row['mAP50-95']:.4f}"  if row['mAP50-95']   is not None else "  N/A  "
            print(f"  {row['class_name']:<20} {p:>8} {r:>8} {m5:>8} {m9:>10}")

    # ---- save JSON summary ---------------------------------------------------
    summary = {
        "model" : str(model_path),
        "data"  : str(data_path),
        "split" : "test",
        "overall": {
            "precision" : precision_overall,
            "recall"    : recall_overall,
            "mAP50"     : map50_overall,
            "mAP50-95"  : map50_95_overall,
        },
        "per_class": per_class,
    }

    json_path = output_dir / "test_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"  JSON summary : {json_path}")

    # ---- confusion matrix location -------------------------------------------
    cm_path = output_dir / "confusion_matrix.png"
    cm_norm_path = output_dir / "confusion_matrix_normalized.png"
    if cm_path.exists():
        print(f"  Confusion matrix : {cm_path}")
    if cm_norm_path.exists():
        print(f"  Confusion matrix (normalized) : {cm_norm_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
