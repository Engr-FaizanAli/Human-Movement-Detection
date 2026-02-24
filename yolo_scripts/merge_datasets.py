#!/usr/bin/env python
"""Merge multiple YOLO datasets under processed_images into one folder.

This script expects each dataset folder to contain an images/ folder and a labels/
folder (or misspelled lables/). It copies files to a single output folder and
prefixes filenames with the dataset name to avoid collisions.

Example:
  python scripts/merge_datasets.py --processed-root "D:\Projects\Human Movement Detection\processed_images" --output "D:\Projects\Human Movement Detection\combined"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_DIR_NAMES = {"labels", "lables"}


def _find_images_labels(root: Path) -> tuple[Path | None, Path | None]:
    if not root.exists():
        return None, None

    image_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "images"]
    label_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in LABEL_DIR_NAMES]

    for img_dir in image_dirs:
        for lbl_dir in label_dirs:
            if img_dir.parent == lbl_dir.parent:
                return img_dir, lbl_dir

    if image_dirs and label_dirs:
        return image_dirs[0], label_dirs[0]

    return None, None


def _find_pairs_under(processed_root: Path) -> List[tuple[Path, Path, Path]]:
    pairs: List[tuple[Path, Path, Path]] = []

    if not processed_root.exists():
        return pairs

    # Check each direct subfolder (dataset)
    for child in sorted([p for p in processed_root.iterdir() if p.is_dir()]):
        img_dir, lbl_dir = _find_images_labels(child)
        if img_dir and lbl_dir:
            pairs.append((child, img_dir, lbl_dir))

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into one folder")
    parser.add_argument("--processed-root", required=True, type=Path, help="Root folder containing datasets")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for merged dataset")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include images even if label file is missing",
    )
    parser.add_argument(
        "--copy-labels-only",
        action="store_true",
        help="Copy only labels (for debugging)",
    )

    args = parser.parse_args()

    pairs = _find_pairs_under(args.processed_root)
    if not pairs:
        print("ERROR: no datasets found under processed root")
        return 2

    images_out = args.output / "images"
    labels_out = args.output / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_images = 0
    copied_images = 0
    copied_labels = 0
    skipped_missing = 0

    for dataset_root, images_dir, labels_dir in pairs:
        dataset_name = dataset_root.name
        image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

        for img_path in image_files:
            total_images += 1
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists() and not args.include_missing:
                skipped_missing += 1
                continue

            new_stem = f"{dataset_name}__{img_path.stem}"
            new_img_path = images_out / f"{new_stem}{img_path.suffix.lower()}"
            new_label_path = labels_out / f"{new_stem}.txt"

            if not args.copy_labels_only:
                shutil.copy2(img_path, new_img_path)
                copied_images += 1

            if label_path.exists():
                shutil.copy2(label_path, new_label_path)
                copied_labels += 1

    print(f"Total images scanned: {total_images}")
    print(f"Copied images: {copied_images}")
    print(f"Copied labels: {copied_labels}")
    print(f"Skipped missing labels: {skipped_missing}")
    print(f"Merged dataset at: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
