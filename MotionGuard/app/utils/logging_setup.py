"""
logging_setup.py — Configure application-wide rotating file + console logging.
"""

import logging
import logging.handlers
from pathlib import Path

from utils.paths import logs_dir

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_FMT = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level_name: str = "INFO") -> None:
    """
    Call once at startup.  Configures:
      - RotatingFileHandler → %APPDATA%\\MotionGuard\\logs\\motionguard.log
      - StreamHandler (console, only in non-frozen / dev mode)
    """
    level = _LEVELS.get(level_name.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear any handlers already attached (important if called twice)
    root.handlers.clear()

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Rotating file handler (5 MB × 5 backups)
    log_file = logs_dir() / "motionguard.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler (dev only — suppressed in frozen exe)
    import sys
    if not getattr(sys, "frozen", False):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def update_log_level(level_name: str) -> None:
    """Hot-update log level at runtime (e.g. from Settings dialog)."""
    level = _LEVELS.get(level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)
