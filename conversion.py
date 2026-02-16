#!/usr/bin/env python
"""Convert COCO JSON or Pascal VOC XML annotations to YOLO format.

Examples:
  python conversion.py --input "C:\Users\Administrator\Downloads" --type coco --output "C:\Users\Administrator\Downloads\labels"
  python conversion.py --input "C:\Users\Administrator\Downloads\XML" --type xml --output "C:\Users\Administrator\Downloads\labels"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import xml.etree.ElementTree as ET


def _safe_float(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_image_size_from_coco(img: dict, images_dir: Path | None) -> Tuple[int, int] | None:
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


def _write_labels(output_dir: Path, stem: str, lines: List[str], include_empty: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / f"{stem}.txt"
    if lines:
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif include_empty:
        label_path.write_text("", encoding="utf-8")


def _convert_coco(
    coco_json: Path,
    output_dir: Path,
    images_dir: Path | None,
    include_empty: bool,
    include_crowd: bool,
    category_order: str,
) -> int:
    data = _load_json(coco_json)

    categories = data.get("categories", []) or []
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []

    if not categories or not images:
        print("ERROR: COCO json missing categories or images")
        return 2

    if category_order == "id":
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

    image_by_id: dict[int, dict] = {}
    for img in images:
        img_id = img.get("id")
        if isinstance(img_id, int):
            image_by_id[img_id] = img

    labels_by_image: dict[int, list[str]] = {img_id: [] for img_id in image_by_id.keys()}

    for ann in annotations:
        if not include_crowd and ann.get("iscrowd") is True:
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

        size = _get_image_size_from_coco(img, images_dir)
        if size is None:
            print(
                f"ERROR: missing width/height for image_id {img_id}. "
                "Provide --images-dir (requires Pillow) or ensure COCO has width/height."
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

    for img_id, img in image_by_id.items():
        file_name = img.get("file_name")
        if not file_name:
            continue
        stem = Path(file_name).stem
        labels = labels_by_image.get(img_id, [])
        _write_labels(output_dir, stem, labels, include_empty)

    classes_path = output_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    class_map_path = output_dir / "class_map.json"
    class_map_path.write_text(
        json.dumps(cat_id_to_index, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote labels to: {output_dir}")
    print(f"Classes: {classes_path}")
    return 0


def _parse_voc_xml(xml_path: Path) -> Tuple[str, Tuple[int, int], List[Tuple[str, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename") or xml_path.stem

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width_text = size_node.findtext("width")
    height_text = size_node.findtext("height")
    if width_text is None or height_text is None:
        raise ValueError(f"Missing width/height in {xml_path}")

    img_w = int(float(width_text))
    img_h = int(float(height_text))

    objects: List[Tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = bnd.findtext("xmin")
        ymin = bnd.findtext("ymin")
        xmax = bnd.findtext("xmax")
        ymax = bnd.findtext("ymax")
        if None in (xmin, ymin, xmax, ymax):
            continue
        objects.append((name, float(xmin), float(ymin), float(xmax), float(ymax)))

    return filename, (img_w, img_h), objects


def _convert_xml(
    input_path: Path,
    output_dir: Path,
    include_empty: bool,
) -> int:
    if input_path.is_file() and input_path.suffix.lower() == ".xml":
        xml_files = [input_path]
    else:
        xml_files = sorted(input_path.rglob("*.xml"))

    if not xml_files:
        print(f"ERROR: no XML files found in {input_path}")
        return 2

    # First pass: collect all class names
    class_names_set = set()
    parsed_cache: dict[Path, Tuple[str, Tuple[int, int], List[Tuple[str, float, float, float, float]]]] = {}
    for xml_path in xml_files:
        try:
            parsed = _parse_voc_xml(xml_path)
        except Exception as e:
            print(f"ERROR parsing {xml_path}: {e}")
            return 2
        parsed_cache[xml_path] = parsed
        for name, *_ in parsed[2]:
            class_names_set.add(name)

    class_names = sorted(class_names_set)
    if not class_names:
        print("ERROR: no object classes found in XML files")
        return 2

    class_to_index = {name: i for i, name in enumerate(class_names)}

    for xml_path, (filename, (img_w, img_h), objects) in parsed_cache.items():
        stem = Path(filename).stem
        lines: List[str] = []

        for name, xmin, ymin, xmax, ymax in objects:
            if img_w <= 0 or img_h <= 0:
                continue
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            if w <= 0 or h <= 0:
                continue

            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            x_center = _safe_float(float(x_center))
            y_center = _safe_float(float(y_center))
            w_norm = _safe_float(float(w_norm))
            h_norm = _safe_float(float(h_norm))

            class_index = class_to_index[name]
            lines.append(
                f"{class_index} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        _write_labels(output_dir, stem, lines, include_empty)

    classes_path = output_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    class_map_path = output_dir / "class_map.json"
    class_map_path.write_text(
        json.dumps(class_to_index, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote labels to: {output_dir}")
    print(f"Classes: {classes_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO JSON or Pascal VOC XML to YOLO labels")
    parser.add_argument("--input", required=True, type=Path, help="Input folder or file")
    parser.add_argument("--type", required=True, choices=["coco", "xml"], help="Input annotation type")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for YOLO labels")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Images directory (COCO only, used if width/height missing; requires Pillow)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Create empty txt files for images/XMLs without annotations",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="Include COCO annotations with iscrowd=true",
    )
    parser.add_argument(
        "--category-order",
        choices=["id", "name"],
        default="id",
        help="COCO class order in classes.txt (default: id)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        return 2

    if args.type == "coco":
        if args.input.is_file() and args.input.suffix.lower() == ".json":
            coco_json = args.input
        else:
            coco_json = args.input / "coco.json"
        if not coco_json.exists():
            print(f"ERROR: COCO json not found: {coco_json}")
            return 2
        return _convert_coco(
            coco_json=coco_json,
            output_dir=args.output,
            images_dir=args.images_dir,
            include_empty=args.include_empty,
            include_crowd=args.include_crowd,
            category_order=args.category_order,
        )

    return _convert_xml(
        input_path=args.input,
        output_dir=args.output,
        include_empty=args.include_empty,
    )


if __name__ == "__main__":
    raise SystemExit(main())
