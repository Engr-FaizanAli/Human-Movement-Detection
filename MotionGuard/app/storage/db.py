"""
db.py — Thread-safe SQLite connection management.

Each thread gets its own connection via threading.local().
WAL mode is enabled for concurrent reads without blocking writes.
"""

import logging
import sqlite3
import threading
from pathlib import Path

from utils.paths import db_path

log = logging.getLogger(__name__)

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """
    Return a per-thread SQLite connection to config.db.
    Creates and configures the connection on first access per thread.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        path = db_path()
        _local.conn = sqlite3.connect(str(path), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        log.debug("Opened SQLite connection on thread %s", threading.current_thread().name)
    return _local.conn


def init_db() -> None:
    """
    Ensure the database file exists and is accessible.
    Call once from the main thread before any workers start.
    """
    path = db_path()
    log.info("Database path: %s", path)
    conn = get_connection()
    conn.execute("SELECT 1")
    log.info("Database initialised")


def close_connection() -> None:
    """Close the connection for the current thread (call on thread exit)."""
    if hasattr(_local, "conn") and _local.conn:
        _local.conn.close()
        _local.conn = None
