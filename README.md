# COCO / VOC XML to YOLO Converter

A small utility to convert **COCO JSON** or **Pascal VOC XML** annotations into **YOLO** label files.

## Features
- Convert COCO `coco.json` to YOLO `.txt` labels
- Convert Pascal VOC XML files to YOLO `.txt` labels
- Writes `classes.txt` and `class_map.json`
- Option to create empty labels for images without annotations

## Requirements
- Python 3.8+
- Optional: Pillow (only needed if COCO JSON is missing `width`/`height`)

Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage
### COCO
If `--input` is a folder, the script looks for `coco.json` inside it.
```powershell
python conversion.py --input "C:\Users\Administrator\Downloads" --type coco --output "C:\Users\Administrator\Downloads\labels"
```

If your COCO JSON does **not** include `width`/`height`, pass an images folder (requires Pillow):
```powershell
python conversion.py --input "C:\Users\Administrator\Downloads" --type coco --output "C:\Users\Administrator\Downloads\labels" --images-dir "E:\TI\LLVIP\infrared"
```

### XML (Pascal VOC)
If `--input` is a folder, all `*.xml` files inside are converted.
```powershell
python conversion.py --input "C:\Users\Administrator\Downloads\XML" --type xml --output "C:\Users\Administrator\Downloads\labels"
```

## Output
The output folder will contain:
- One `.txt` file per image with YOLO annotations
- `classes.txt` (class index order)
- `class_map.json` (class name/id to index mapping)

## Notes
- For COCO, class order in `classes.txt` defaults to category `id` ascending. You can change it with `--category-order name`.
- `--include-empty` creates empty label files for images without objects.
- `--include-crowd` includes COCO annotations with `iscrowd=true`.

## Help
```powershell
python conversion.py --help
```
