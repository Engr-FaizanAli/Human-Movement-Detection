"""
migrations.py — Schema versioning.

Add new versions to MIGRATIONS list.  run_migrations() runs each in order,
skipping those already applied.  Safe to call on every app start.
"""

import logging

from storage.db import get_connection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version table — always created first
# ---------------------------------------------------------------------------
_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Migration list — append new entries; never edit existing ones
# ---------------------------------------------------------------------------
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS recorders (
            id              TEXT PRIMARY KEY,
            brand           TEXT NOT NULL,
            name            TEXT NOT NULL,
            ip              TEXT NOT NULL,
            rtsp_port       INTEGER DEFAULT 554,
            username        TEXT,
            password_enc    TEXT,
            channel_count   INTEGER DEFAULT 16,
            stream_pref     TEXT DEFAULT 'sub',
            template_key    TEXT DEFAULT NULL,
            custom_template TEXT DEFAULT NULL,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS channels (
            id              TEXT PRIMARY KEY,
            recorder_id     TEXT REFERENCES recorders(id) ON DELETE CASCADE,
            channel_num     INTEGER NOT NULL,
            friendly_name   TEXT,
            enabled         INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS cameras (
            id              TEXT PRIMARY KEY,
            source_type     TEXT DEFAULT 'rtsp',
            name            TEXT NOT NULL,
            rtsp_url        TEXT NOT NULL,
            username        TEXT,
            password_enc    TEXT,
            enabled         INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS offline_sources (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            file_path       TEXT NOT NULL,
            loop_enabled    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS exclusion_zones (
            id              TEXT PRIMARY KEY,
            source_id       TEXT NOT NULL,
            source_type     TEXT NOT NULL,
            zone_name       TEXT NOT NULL,
            vertices_json   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detection_params (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id   TEXT NOT NULL,
            source_type TEXT NOT NULL,
            param_name  TEXT NOT NULL,
            param_value TEXT NOT NULL,
            UNIQUE(source_id, source_type, param_name)
        );

        CREATE TABLE IF NOT EXISTS events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id     TEXT NOT NULL,
            source_type   TEXT NOT NULL,
            source_name   TEXT NOT NULL DEFAULT '',
            timestamp     TEXT NOT NULL,
            duration_sec  REAL,
            snapshot_path TEXT
        );

        CREATE TABLE IF NOT EXISTS global_settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """,
    ),
]


def run_migrations() -> None:
    """Apply any pending schema migrations on startup."""
    conn = get_connection()
    conn.execute(_BOOTSTRAP)
    conn.commit()

    applied = {
        row[0] for row in conn.execute("SELECT version FROM schema_version")
    }

    for version, sql in MIGRATIONS:
        if version in applied:
            continue
        log.info("Applying migration v%d", version)
        # Execute each statement separately (executescript commits automatically)
        for statement in sql.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        conn.execute(
            "INSERT INTO schema_version(version) VALUES (?)", (version,)
        )
        conn.commit()
        log.info("Migration v%d applied", version)

    log.info("Database schema up to date")
