"""
paths.py — Resolve all application data paths.

Handles both dev mode (running from source) and frozen mode (PyInstaller exe).
All app data lives under %APPDATA%\\MotionGuard\\ on Windows.
"""

import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def app_data_dir() -> Path:
    """Return %APPDATA%\\MotionGuard\\ and create it if needed."""
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    d = base / "MotionGuard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return app_data_dir() / "config.db"


def logs_dir() -> Path:
    d = app_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshots_dir() -> Path:
    d = app_data_dir() / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def assets_dir() -> Path:
    """
    Locate the assets/ folder.
    - Frozen: alongside the exe (sys._MEIPASS/assets)
    - Dev: MotionGuard/assets (two levels above this file's location)
    """
    if _is_frozen():
        return Path(sys._MEIPASS) / "assets"
    # dev: this file is at MotionGuard/app/utils/paths.py
    # assets is at MotionGuard/assets
    return Path(__file__).parent.parent.parent / "assets"


def motion_scripts_dir() -> Path:
    """
    Locate the motion_scripts directory that contains ti_motion_detect_v4.py.
    - Frozen: sys._MEIPASS/motion_scripts
    - Dev: ../../motion_scripts relative to MotionGuard root
    """
    if _is_frozen():
        return Path(sys._MEIPASS) / "motion_scripts"
    # dev: MotionGuard/app/utils/paths.py → MotionGuard → Human-Movement-Detection → motion_scripts
    return Path(__file__).parent.parent.parent.parent / "motion_scripts"


def alarm_wav_path() -> Path:
    return assets_dir() / "sounds" / "alarm.wav"


def logo_path() -> Path:
    return assets_dir() / "icons" / "logo.png"


def logo_ico_path() -> Path:
    return assets_dir() / "icons" / "logo.ico"
