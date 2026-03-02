# -*- mode: python ; coding: utf-8 -*-
"""
motionguard.spec — PyInstaller build specification for MotionGuard.

Build command (run from MotionGuard/ directory):
    pyinstaller motionguard.spec --clean

Output: dist/MotionGuard/MotionGuard.exe  (one-folder build)
"""

import sys
from pathlib import Path

# Resolve paths relative to this spec file
SPEC_DIR = Path(SPECPATH)   # MotionGuard/
APP_DIR  = SPEC_DIR / "app"
ASSETS   = SPEC_DIR / "assets"

# The v4 motion script (sibling of MotionGuard/)
V4_SCRIPT = SPEC_DIR.parent / "motion_scripts" / "ti_motion_detect_v4.py"

block_cipher = None

a = Analysis(
    [str(APP_DIR / "main.py")],
    pathex=[str(APP_DIR)],
    binaries=[],
    datas=[
        # Bundle all assets
        (str(ASSETS), "assets"),
        # Bundle the motion detection algorithm
        (str(V4_SCRIPT), "motion_scripts"),
    ],
    hiddenimports=[
        # PySide6 platform plugins
        "PySide6.QtSvg",
        "PySide6.QtPrintSupport",
        "PySide6.QtNetwork",
        "PySide6.QtXml",
        # OpenCV + numpy
        "cv2",
        "numpy",
        "numpy.core._multiarray_umath",
        "numpy.core.multiarray",
        # Sound
        "pygame",
        "pygame.mixer",
        # ONVIF discovery (optional; include so app doesn't crash)
        "onvif",
        "wsdiscovery",
        "zeep",
        "zeep.transports",
        # Standard library modules sometimes missed
        "wave",
        "struct",
        "sqlite3",
        "uuid",
        "threading",
        "queue",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages to keep build small
        "streamlit",
        "matplotlib",
        "torch",
        "torchvision",
        "ultralytics",
        "sklearn",
        "scipy",
        "IPython",
        "jupyter",
        "pandas",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MotionGuard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # --windowed (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ASSETS / "icons" / "logo.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MotionGuard",
)
