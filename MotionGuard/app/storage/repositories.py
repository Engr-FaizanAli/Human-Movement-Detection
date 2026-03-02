"""
repositories.py — Pure-function CRUD layer for all database tables.

All functions use get_connection() which is thread-local, so workers can
safely call these without locking.

Credential handling:
  - Passwords are stored encrypted via _encrypt/_decrypt (lightweight XOR+base64)
  - Never pass raw passwords to logging or return them in list results except
    when explicitly needed for connection building
"""

import base64
import json
import logging
import uuid
from typing import Any

from storage.db import get_connection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential encryption (lightweight; swap to Windows Credential Manager later)
# ---------------------------------------------------------------------------

def _derive_key() -> bytes:
    """16-byte key from MAC address XOR'd with a fixed salt."""
    import uuid as _uuid
    mac = _uuid.getnode()
    mac_bytes = mac.to_bytes(6, "big") * 3  # 18 bytes; trim to 16
    salt = b"\x4d\x47\x37\x61\x62\x63\x64\x65\x31\x32\x33\x34\x35\x36\x37\x38"
    key = bytes(a ^ b for a, b in zip(mac_bytes[:16], salt))
    return key


_KEY = _derive_key()


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    data = plaintext.encode("utf-8")
    # Repeat key to match data length
    key_stream = (_KEY * ((len(data) // len(_KEY)) + 1))[: len(data)]
    encrypted = bytes(a ^ b for a, b in zip(data, key_stream))
    return base64.b64encode(encrypted).decode("ascii")


def _decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    try:
        encrypted = base64.b64decode(ciphertext.encode("ascii"))
        key_stream = (_KEY * ((len(encrypted) // len(_KEY)) + 1))[: len(encrypted)]
        return bytes(a ^ b for a, b in zip(encrypted, key_stream)).decode("utf-8")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------

def add_recorder(
    brand: str,
    name: str,
    ip: str,
    rtsp_port: int = 554,
    username: str = "",
    password: str = "",
    channel_count: int = 16,
    stream_pref: str = "sub",
    template_key: str | None = None,
    custom_template: str | None = None,
) -> str:
    """Insert a recorder and return its UUID."""
    rid = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        """INSERT INTO recorders
           (id, brand, name, ip, rtsp_port, username, password_enc,
            channel_count, stream_pref, template_key, custom_template)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (rid, brand, name, ip, rtsp_port, username, _encrypt(password),
         channel_count, stream_pref, template_key, custom_template),
    )
    conn.commit()
    return rid


def get_recorders() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM recorders ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_recorder(recorder_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM recorders WHERE id=?", (recorder_id,)
    ).fetchone()
    return dict(row) if row else None


def get_recorder_password(recorder_id: str) -> str:
    conn = get_connection()
    row = conn.execute(
        "SELECT password_enc FROM recorders WHERE id=?", (recorder_id,)
    ).fetchone()
    return _decrypt(row["password_enc"]) if row else ""


def update_recorder(recorder_id: str, **fields) -> None:
    if "password" in fields:
        fields["password_enc"] = _encrypt(fields.pop("password"))
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [recorder_id]
    get_connection().execute(
        f"UPDATE recorders SET {sets} WHERE id=?", vals
    )
    get_connection().commit()


def delete_recorder(recorder_id: str) -> None:
    get_connection().execute("DELETE FROM recorders WHERE id=?", (recorder_id,))
    get_connection().commit()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def add_channel(recorder_id: str, channel_num: int, friendly_name: str = "") -> str:
    cid = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO channels (id, recorder_id, channel_num, friendly_name) VALUES (?,?,?,?)",
        (cid, recorder_id, channel_num, friendly_name or f"CH{channel_num}"),
    )
    conn.commit()
    return cid


def get_channels(recorder_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM channels WHERE recorder_id=? ORDER BY channel_num",
        (recorder_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_channel(channel_id: str, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [channel_id]
    conn = get_connection()
    conn.execute(f"UPDATE channels SET {sets} WHERE id=?", vals)
    conn.commit()


def delete_channel(channel_id: str) -> None:
    get_connection().execute("DELETE FROM channels WHERE id=?", (channel_id,))
    get_connection().commit()


def provision_channels(recorder_id: str, count: int) -> None:
    """Create channel entries 1..count for a recorder, replacing existing."""
    conn = get_connection()
    conn.execute("DELETE FROM channels WHERE recorder_id=?", (recorder_id,))
    for ch in range(1, count + 1):
        cid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO channels (id, recorder_id, channel_num, friendly_name) VALUES (?,?,?,?)",
            (cid, recorder_id, ch, f"CH{ch}"),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Cameras (direct IP / ONVIF / manual RTSP)
# ---------------------------------------------------------------------------

def add_camera(
    name: str,
    rtsp_url: str,
    username: str = "",
    password: str = "",
    source_type: str = "rtsp",
) -> str:
    cid = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO cameras (id, source_type, name, rtsp_url, username, password_enc) VALUES (?,?,?,?,?,?)",
        (cid, source_type, name, rtsp_url, username, _encrypt(password)),
    )
    conn.commit()
    return cid


def get_cameras() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_camera(camera_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
    return dict(row) if row else None


def get_camera_password(camera_id: str) -> str:
    conn = get_connection()
    row = conn.execute("SELECT password_enc FROM cameras WHERE id=?", (camera_id,)).fetchone()
    return _decrypt(row["password_enc"]) if row else ""


def update_camera(camera_id: str, **fields) -> None:
    if "password" in fields:
        fields["password_enc"] = _encrypt(fields.pop("password"))
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [camera_id]
    conn = get_connection()
    conn.execute(f"UPDATE cameras SET {sets} WHERE id=?", vals)
    conn.commit()


def delete_camera(camera_id: str) -> None:
    get_connection().execute("DELETE FROM cameras WHERE id=?", (camera_id,))
    get_connection().commit()


# ---------------------------------------------------------------------------
# Offline Sources
# ---------------------------------------------------------------------------

def add_offline_source(name: str, file_path: str, loop_enabled: bool = False) -> str:
    oid = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO offline_sources (id, name, file_path, loop_enabled) VALUES (?,?,?,?)",
        (oid, name, file_path, 1 if loop_enabled else 0),
    )
    conn.commit()
    return oid


def get_offline_sources() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM offline_sources ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_offline_source(source_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM offline_sources WHERE id=?", (source_id,)).fetchone()
    return dict(row) if row else None


def update_offline_source(source_id: str, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [source_id]
    conn = get_connection()
    conn.execute(f"UPDATE offline_sources SET {sets} WHERE id=?", vals)
    conn.commit()


def delete_offline_source(source_id: str) -> None:
    get_connection().execute("DELETE FROM offline_sources WHERE id=?", (source_id,))
    get_connection().commit()


# ---------------------------------------------------------------------------
# Exclusion Zones
# ---------------------------------------------------------------------------

def save_zones(source_id: str, source_type: str, zones: list[dict]) -> None:
    """
    Replace all zones for a source.
    zones: [{"name": str, "vertices": [[x_norm, y_norm], ...]}]
    """
    conn = get_connection()
    conn.execute(
        "DELETE FROM exclusion_zones WHERE source_id=? AND source_type=?",
        (source_id, source_type),
    )
    for zone in zones:
        zid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO exclusion_zones (id, source_id, source_type, zone_name, vertices_json) VALUES (?,?,?,?,?)",
            (zid, source_id, source_type, zone["name"], json.dumps(zone["vertices"])),
        )
    conn.commit()


def get_zones(source_id: str, source_type: str) -> list[dict]:
    """
    Returns [{"name": str, "vertices": [[x_norm, y_norm], ...]}]
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT zone_name, vertices_json FROM exclusion_zones WHERE source_id=? AND source_type=?",
        (source_id, source_type),
    ).fetchall()
    return [
        {"name": r["zone_name"], "vertices": json.loads(r["vertices_json"])}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Detection Parameters (per-source overrides)
# ---------------------------------------------------------------------------

def save_param(source_id: str, source_type: str, key: str, value: Any) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO detection_params (source_id, source_type, param_name, param_value)
           VALUES (?,?,?,?)
           ON CONFLICT(source_id, source_type, param_name) DO UPDATE SET param_value=excluded.param_value""",
        (source_id, source_type, key, str(value)),
    )
    conn.commit()


def get_params(source_id: str, source_type: str) -> dict[str, str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT param_name, param_value FROM detection_params WHERE source_id=? AND source_type=?",
        (source_id, source_type),
    ).fetchall()
    return {r["param_name"]: r["param_value"] for r in rows}


def delete_params(source_id: str, source_type: str) -> None:
    conn = get_connection()
    conn.execute(
        "DELETE FROM detection_params WHERE source_id=? AND source_type=?",
        (source_id, source_type),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def add_event(
    source_id: str,
    source_type: str,
    source_name: str,
    timestamp: str,
    duration_sec: float | None = None,
    snapshot_path: str | None = None,
) -> int:
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO events (source_id, source_type, source_name, timestamp, duration_sec, snapshot_path)
           VALUES (?,?,?,?,?,?)""",
        (source_id, source_type, source_name, timestamp, duration_sec, snapshot_path),
    )
    conn.commit()
    return cur.lastrowid


def get_events(limit: int = 200) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_old_events(older_than_days: int) -> int:
    conn = get_connection()
    cur = conn.execute(
        "DELETE FROM events WHERE timestamp < datetime('now', ?)",
        (f"-{older_than_days} days",),
    )
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Global Settings
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, str] = {
    "detection_fps": "10",
    "preview_fps": "15",
    "stream_pref": "sub",
    "alarm_enabled": "1",
    "alarm_volume": "0.8",
    "snapshot_enabled": "1",
    "snapshot_retention_days": "30",
    "auto_start": "0",
    "start_minimized": "0",
    "log_level": "INFO",
    "detection_mode": "mog2",
    "history": "200",
    "threshold": "15.0",       # --threshold 15
    "diff_frames": "5",
    "use_clahe": "0",           # --no-clahe
    "clahe_clip": "2.5",
    "clahe_tile": "12",
    "use_temporal": "1",
    "alpha": "0.4",
    "morph_close": "5",
    "morph_open": "3",
    "min_area": "10",           # --min-area 10
    "max_area": "5000",
    "min_solidity": "0.3",
    "min_density": "0.0",
    "persistence": "4",         # --persistence 4
    "min_displacement": "15.0", # --min-displacement 15
    "spatial_tol": "20.0",      # --spatial-tol 20
    "max_absent": "5",          # --max-absent 5
    "bbox_smooth_alpha": "0.35",
    "vel_smooth_alpha": "0.5",
    "ptz_aware": "1",           # PTZ motion compensation enabled
    "ptz_motion_thresh": "5.0",
    "ptz_settle_frames": "125",
    "show_candidates": "0",     # --no-candidates
}


def get_setting(key: str, default: str | None = None) -> str:
    conn = get_connection()
    row = conn.execute(
        "SELECT value FROM global_settings WHERE key=?", (key,)
    ).fetchone()
    if row:
        return row["value"]
    return default if default is not None else _DEFAULTS.get(key, "")


def set_setting(key: str, value: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO global_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    conn.commit()


def get_all_settings() -> dict[str, str]:
    result = dict(_DEFAULTS)
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM global_settings").fetchall()
    for r in rows:
        result[r["key"]] = r["value"]
    return result
