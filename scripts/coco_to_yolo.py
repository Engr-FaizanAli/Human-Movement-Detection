#!/usr/bin/env python
"""Convert COCO JSON annotations to YOLO txt format.

Example:
  python scripts/coco_to_yolo.py --coco-json "C:\Users\Administrator\Downloads\coco.json" \
    --labels-dir "C:\Users\Administrator\Downloads\labels" --include-empty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _get_image_size(img: dict, images_dir: Path | None) -> tuple[int, int] | None:
    w = img.get("width")
    h = img.get("height")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h

    if images_dir is None:
        return None

    # Optional Pillow fallback if width/height are missing.
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    file_name = img.get("file_name")
    if not file_name:
        return None
    img_path = images_dir / file_name
    if not img_path.exists():
        return None

    with Image.open(img_path) as im:
        return im.size  # (width, height)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO label txt files")
    parser.add_argument("--coco-json", required=True, type=Path, help="Path to coco.json")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Output directory for YOLO txt files (default: <coco_json_dir>/labels)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional images directory (used if width/height missing; for Pillow fallback)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Create empty txt files for images without annotations",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="Include annotations with iscrowd=true",
    )
    parser.add_argument(
        "--category-order",
        choices=["id", "name"],
        default="id",
        help="Order of classes in classes.txt (default: id ascending)",
    )
    parser.add_argument(
        "--write-class-map",
        action="store_true",
        help="Write class_map.json (category_id -> class_index) next to labels dir",
    )

    args = parser.parse_args()

    coco_json: Path = args.coco_json
    if not coco_json.exists():
        print(f"ERROR: coco json not found: {coco_json}")
        return 2

    labels_dir: Path = args.labels_dir or (coco_json.parent / "labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    data = _load_json(coco_json)

    categories = data.get("categories", []) or []
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []

    if not categories or not images:
        print("ERROR: coco json missing categories or images")
        return 2

    if args.category_order == "id":
        categories_sorted = sorted(categories, key=lambda c: c.get("id", 0))
    else:
        categories_sorted = sorted(categories, key=lambda c: str(c.get("name", "")))

    cat_id_to_index: dict[int, int] = {}
    class_names: list[str] = []
    for i, cat in enumerate(categories_sorted):
        cat_id = cat.get("id")
        if not isinstance(cat_id, int):
            continue
        cat_id_to_index[cat_id] = i
        class_names.append(str(cat.get("name", f"class_{cat_id}")))

    if not cat_id_to_index:
        print("ERROR: no valid categories found")
        return 2

    # image_id -> info
    image_by_id: dict[int, dict] = {}
    for img in images:
        img_id = img.get("id")
        if isinstance(img_id, int):
            image_by_id[img_id] = img

    # collect labels per image_id
    labels_by_image: dict[int, list[str]] = {img_id: [] for img_id in image_by_id.keys()}

    for ann in annotations:
        if not args.include_crowd and ann.get("iscrowd") is True:
            continue

        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")

        if not isinstance(img_id, int) or not isinstance(cat_id, int) or not bbox:
            continue

        if cat_id not in cat_id_to_index:
            continue

        img = image_by_id.get(img_id)
        if not img:
            continue

        size = _get_image_size(img, args.images_dir)
        if size is None:
            print(
                f"ERROR: missing width/height for image_id {img_id}. "
                "Provide --images-dir with Pillow installed or ensure COCO has width/height."
            )
            return 2

        img_w, img_h = size
        try:
            x, y, w, h = bbox
        except Exception:
            continue

        if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
            continue

        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        x_center = _safe_float(float(x_center))
        y_center = _safe_float(float(y_center))
        w_norm = _safe_float(float(w_norm))
        h_norm = _safe_float(float(h_norm))

        class_index = cat_id_to_index[cat_id]
        labels_by_image[img_id].append(
            f"{class_index} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    # write label files
    for img_id, img in image_by_id.items():
        file_name = img.get("file_name")
        if not file_name:
            continue
        stem = Path(file_name).stem
        label_path = labels_dir / f"{stem}.txt"
        labels = labels_by_image.get(img_id, [])

        if labels:
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        elif args.include_empty:
            label_path.write_text("", encoding="utf-8")

    # write classes.txt
    classes_path = labels_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    if args.write_class_map:
        class_map_path = labels_dir / "class_map.json"
        class_map_path.write_text(
            json.dumps(cat_id_to_index, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(f"Wrote labels to: {labels_dir}")
    print(f"Classes: {classes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
