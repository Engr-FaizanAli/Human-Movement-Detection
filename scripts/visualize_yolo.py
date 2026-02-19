#!/usr/bin/env python
"""Visualize YOLO labels by drawing bounding boxes on images.

Examples:
  python scripts/visualize_yolo.py --images "D:\data\images" --labels "D:\data\labels" --output "D:\data\viz"
  python scripts/visualize_yolo.py --root "D:\data\hit_uav_filtered-20260217T111554Z-3-001" --output "D:\data\viz"
  python scripts/visualize_yolo.py --processed-root "D:\Projects\Human Movement Detection\processed_images" --limit 50 --manual
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_MAP = {0: "human", 1: "car"}


def _load_classes(classes_path: Path | None, class_names: str | None) -> Dict[int, str]:
    if class_names:
        names = [c.strip() for c in class_names.split(",") if c.strip()]
        return {i: name for i, name in enumerate(names)}

    if classes_path and classes_path.exists():
        lines = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines()]
        lines = [l for l in lines if l]
        if lines:
            return {i: name for i, name in enumerate(lines)}

    return DEFAULT_CLASS_MAP.copy()


def _parse_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except Exception:
            continue
        boxes.append((cls, x, y, w, h))

    return boxes


def _yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)  # type: ignore[attr-defined]


def _draw_boxes(
    img: Image.Image,
    boxes: List[Tuple[int, float, float, float, float]],
    class_map: Dict[int, str],
) -> Image.Image:
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, img.size[0] // 80))
    except Exception:
        font = ImageFont.load_default()

    for cls, x, y, w, h in boxes:
        x1, y1, x2, y2 = _yolo_to_xyxy(x, y, w, h, img.width, img.height)
        x1 = max(0, min(img.width - 1, x1))
        y1 = max(0, min(img.height - 1, y1))
        x2 = max(0, min(img.width - 1, x2))
        y2 = max(0, min(img.height - 1, y2))

        color = "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 400))

        label = class_map.get(cls, str(cls))
        text_w, text_h = _text_size(draw, label, font)
        text_bg = [x1, max(0, y1 - text_h - 2), x1 + text_w + 4, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 2, max(0, y1 - text_h - 2)), label, fill="white", font=font)

    return img


def _find_images_labels(root: Path) -> tuple[Path | None, Path | None]:
    if not root.exists():
        return None, None

    image_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "images"]
    label_dirs = [
        p
        for p in root.rglob("*")
        if p.is_dir() and p.name.lower() in {"labels", "lables"}
    ]

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

    # If processed_root itself contains images/labels
    img_dir, lbl_dir = _find_images_labels(processed_root)
    if img_dir and lbl_dir:
        pairs.append((processed_root, img_dir, lbl_dir))

    # Check each direct subfolder (dataset)
    for child in sorted([p for p in processed_root.iterdir() if p.is_dir()]):
        img_dir, lbl_dir = _find_images_labels(child)
        if img_dir and lbl_dir:
            pairs.append((child, img_dir, lbl_dir))

    return pairs


def _display_sequence(
    dataset_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_map: Dict[int, str],
    limit: int,
    manual: bool,
    delay: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("ERROR: matplotlib is required for live display. Install it with: pip install matplotlib")
        raise SystemExit(2)

    image_files = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if limit and limit > 0:
        image_files = image_files[:limit]

    if not image_files:
        print(f"No images found for {dataset_name}")
        return

    plt.ion()
    fig, ax = plt.subplots(1, 1)

    for idx, img_path in enumerate(image_files, start=1):
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = _parse_yolo_labels(label_path)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = _draw_boxes(img, boxes, class_map)

        ax.clear()
        ax.imshow(img)
        ax.set_title(f"{dataset_name} ({idx}/{len(image_files)}) - {img_path.name}")
        ax.axis("off")
        if manual:
            plt.waitforbuttonpress(timeout=-1)
        else:
            plt.pause(delay)

    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw YOLO bounding boxes on images")
    parser.add_argument("--processed-root", type=Path, default=None, help="Root folder containing dataset subfolders")
    parser.add_argument("--root", type=Path, default=None, help="Root folder containing images/labels")
    parser.add_argument("--images", type=Path, default=None, help="Path to images folder")
    parser.add_argument("--labels", type=Path, default=None, help="Path to YOLO labels folder")
    parser.add_argument("--output", type=Path, default=None, help="Output folder for visualized images")
    parser.add_argument("--classes", type=Path, default=None, help="Optional classes.txt file")
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class names")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of images per folder")
    parser.add_argument("--manual", action="store_true", help="Advance only on key/mouse press")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to pause between images")

    args = parser.parse_args()

    # Live display across processed_images
    if args.processed_root:
        pairs = _find_pairs_under(args.processed_root)
        if not pairs:
            print("ERROR: no images/labels pairs found under processed root")
            return 2

        class_map = _load_classes(args.classes, args.class_names)
        for dataset_root, images_dir, labels_dir in pairs:
            _display_sequence(
                dataset_root.name, images_dir, labels_dir, class_map, args.limit, args.manual, args.delay
            )
        return 0

    images_dir = args.images
    labels_dir = args.labels

    if args.root:
        images_dir, labels_dir = _find_images_labels(args.root)
        if not images_dir or not labels_dir:
            print("ERROR: could not auto-detect images/labels folders under root")
            return 2

    if not images_dir or not labels_dir:
        print("ERROR: provide --root or both --images and --labels")
        return 2

    if not images_dir.exists() or not labels_dir.exists():
        print("ERROR: images or labels path does not exist")
        return 2

    class_map = _load_classes(args.classes, args.class_names)

    if not args.output:
        print("ERROR: --output is required when not using --processed-root")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.limit and args.limit > 0:
        image_files = image_files[: args.limit]

    if not image_files:
        print("ERROR: no images found")
        return 2

    processed = 0
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = _parse_yolo_labels(label_path)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = _draw_boxes(img, boxes, class_map)
            out_path = args.output / img_path.name
            img.save(out_path)
            processed += 1

    print(f"Processed {processed} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
