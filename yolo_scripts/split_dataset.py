#!/usr/bin/env python
"""Split a YOLO dataset into train/val/test with stratified sampling.

Example:
  python scripts/split_dataset.py --input "D:\Projects\Human Movement Detection\combined" --output "D:\Projects\Human Movement Detection\yolo_split" --train 0.8 --val 0.1 --test 0.1 --seed 42 --names "Human,Car"
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        classes.add(cls)
    return classes


def _group_key(classes: set[int]) -> str:
    if not classes:
        return "none"
    if classes == {0}:
        return "0"
    if classes == {1}:
        return "1"
    if 0 in classes and 1 in classes:
        return "0+1"
    # Any other classes
    return "+".join(str(c) for c in sorted(classes))


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    test_n = n - train_n - val_n
    # Adjust to avoid negative
    if test_n < 0:
        test_n = 0
    return train_n, val_n, test_n


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified train/val/test split for YOLO datasets")
    parser.add_argument("--input", required=True, type=Path, help="Input dataset with images/ and labels/")
    parser.add_argument("--output", required=True, type=Path, help="Output root for split dataset")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--names", type=str, default="Human,Car", help="Comma-separated class names")

    args = parser.parse_args()

    images_dir = args.input / "images"
    labels_dir = args.input / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print("ERROR: input must contain images/ and labels/ folders")
        return 2

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print("ERROR: train/val/test ratios must sum to 1.0")
        return 2

    image_files = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not image_files:
        print("ERROR: no images found")
        return 2

    # Build stratified groups
    groups: Dict[str, List[Path]] = {}
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        classes = _parse_label_classes(label_path)
        key = _group_key(classes)
        groups.setdefault(key, []).append(img_path)

    random.seed(args.seed)
    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        train_n, val_n, _ = _split_indices(n, args.train, args.val)
        train_items = items[:train_n]
        val_items = items[train_n : train_n + val_n]
        test_items = items[train_n + val_n :]

        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)

    # Create output folders
    for split in ("train", "val", "test"):
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files
    for split, items in splits.items():
        for img_path in items:
            label_path = labels_dir / f"{img_path.stem}.txt"
            out_img = args.output / split / "images" / img_path.name
            out_lbl = args.output / split / "labels" / label_path.name
            shutil.copy2(img_path, out_img)
            if label_path.exists():
                shutil.copy2(label_path, out_lbl)

    # Write data.yaml for YOLOv8
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
            ]
            + [f"  {i}: {name}" for i, name in enumerate(names)]
            + [""]
        ),
        encoding="utf-8",
    )

    print("Split complete:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(splits[split])} images")
    print(f"data.yaml: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
