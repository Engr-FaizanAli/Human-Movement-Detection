#!/usr/bin/env python
"""One-off visualizer for hit_uav_filtered dataset (hardcoded paths)."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ROOT = Path(r"D:\Projects\Human Movement Detection\processed_images\hit_uav_filtered-20260217T111554Z-3-001")
IMAGES = ROOT / "hit_uav_filtered" / "images"
LABELS = ROOT / "hit_uav_filtered" / "labels"
OUTPUT = ROOT / "viz"
CLASSES = LABELS / "classes.txt"
LIMIT = 50


def load_classes() -> dict[int, str]:
    if CLASSES.exists():
        lines = [l.strip() for l in CLASSES.read_text(encoding="utf-8").splitlines()]
        lines = [l for l in lines if l]
        return {i: name for i, name in enumerate(lines)}
    return {}


def parse_labels(label_path: Path):
    boxes = []
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


def yolo_to_xyxy(x, y, w, h, img_w, img_h):
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


def draw_boxes(img, boxes, class_map):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, img.size[0] // 80))
    except Exception:
        font = ImageFont.load_default()

    for cls, x, y, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, img.width, img.height)
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


def main():
    if not IMAGES.exists() or not LABELS.exists():
        print("ERROR: images or labels folder not found")
        return 2

    OUTPUT.mkdir(parents=True, exist_ok=True)

    class_map = load_classes()
    image_files = [p for p in sorted(IMAGES.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not image_files:
        print("ERROR: no images found")
        return 2

    if LIMIT and LIMIT > 0:
        image_files = image_files[:LIMIT]

    processed = 0
    for img_path in image_files:
        label_path = LABELS / f"{img_path.stem}.txt"
        boxes = parse_labels(label_path)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = draw_boxes(img, boxes, class_map)
            out_path = OUTPUT / img_path.name
            img.save(out_path)
            processed += 1

    print(f"Processed {processed} images -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
