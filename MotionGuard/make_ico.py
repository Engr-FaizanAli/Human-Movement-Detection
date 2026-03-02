"""
make_ico.py — Convert logo.png to logo.ico for PyInstaller.

Run once before building:
    python make_ico.py
"""

from pathlib import Path
from PIL import Image

assets = Path(__file__).parent / "assets" / "icons"
png_path = assets / "logo.png"
ico_path = assets / "logo.ico"

if not png_path.exists():
    raise FileNotFoundError(f"logo.png not found: {png_path}")

img = Image.open(png_path).convert("RGBA")

# ICO requires specific sizes
sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
icons = [img.resize(s, Image.LANCZOS) for s in sizes]

icons[0].save(
    str(ico_path),
    format="ICO",
    sizes=[(i.width, i.height) for i in icons],
    append_images=icons[1:],
)

print(f"Created: {ico_path}")
