#!/usr/bin/env python
"""Check for missing YOLO labels, then visualize labeled images.

Example:
  python scripts/visualize_missing_labels.py --processed-root "D:\Projects\Human Movement Detection\processed_images" --limit 100 --manual
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_images_labels(root: Path) -> tuple[Path | None, Path | None]:
    if not root.exists():
        return None, None

    image_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "images"]
    label_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "labels"]

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

    img_dir, lbl_dir = _find_images_labels(processed_root)
    if img_dir and lbl_dir:
        pairs.append((processed_root, img_dir, lbl_dir))

    for child in sorted([p for p in processed_root.iterdir() if p.is_dir()]):
        img_dir, lbl_dir = _find_images_labels(child)
        if img_dir and lbl_dir:
            pairs.append((child, img_dir, lbl_dir))

    return pairs


def _display_missing(
    dataset_name: str,
    images_dir: Path,
    labels_dir: Path,
    limit: int,
    manual: bool,
    delay: float,
) -> int:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("ERROR: matplotlib is required for live display. Install it with: pip install matplotlib")
        raise SystemExit(2)

    image_files = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not image_files:
        print(f"No images found for {dataset_name}")
        return 0

    missing = [p for p in image_files if not (labels_dir / f"{p.stem}.txt").exists()]
    if missing:
        print(f"{dataset_name}: missing labels for {len(missing)} images")
    else:
        print(f"{dataset_name}: no missing labels")

    labeled = [p for p in image_files if (labels_dir / f"{p.stem}.txt").exists()]
    if not labeled:
        print(f"{dataset_name}: no labeled images to display")
        return len(missing)

    if limit and limit > 0:
        labeled = labeled[:limit]

    plt.ion()
    fig, ax = plt.subplots(1, 1)

    for idx, img_path in enumerate(labeled, start=1):
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        ax.clear()
        ax.imshow(img)
        ax.set_title(f"{dataset_name} labeled ({idx}/{len(labeled)}) - {img_path.name}")
        ax.axis("off")

        if manual:
            plt.waitforbuttonpress(timeout=-1)
        else:
            plt.pause(delay)

    plt.close(fig)
    return len(missing)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show images that are missing YOLO label files")
    parser.add_argument("--processed-root", type=Path, required=True, help="Root folder containing dataset subfolders")
    parser.add_argument("--limit", type=int, default=100, help="Max labeled images per folder to show")
    parser.add_argument("--manual", action="store_true", help="Advance only on key/mouse press")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to pause between images")

    args = parser.parse_args()

    pairs = _find_pairs_under(args.processed_root)
    if not pairs:
        print("ERROR: no images/labels pairs found under processed root")
        return 2

    for dataset_root, images_dir, labels_dir in pairs:
        _display_missing(dataset_root.name, images_dir, labels_dir, args.limit, args.manual, args.delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
