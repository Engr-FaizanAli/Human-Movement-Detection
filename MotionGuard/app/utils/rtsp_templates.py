"""
rtsp_templates.py — RTSP URL builders for Dahua, Hikvision, and generic sources.

The template system allows per-recorder override so users can handle non-standard
firmware variants without modifying source code.
"""

import urllib.parse
from typing import Any

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, Any]] = {
    "dahua": {
        "label": "Dahua NVR",
        "pattern": "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype={stream}",
        "stream_map": {"main": 0, "sub": 1},
    },
    "hikvision_v1": {
        "label": "Hikvision (Standard — /Streaming/Channels/)",
        "pattern": "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel_id}",
        # channel_id = ch * 100 + stream_index  (main=1, sub=2)
        "stream_map": {"main": 1, "sub": 2},
    },
    "hikvision_v2": {
        "label": "Hikvision (Alt / h264 path)",
        "pattern": "rtsp://{user}:{password}@{ip}:{port}/h264/ch{ch}/{stream_tag}/av_stream",
        # stream_tag: main=0, sub=1 mapped differently
        "stream_map": {"main": "main", "sub": "sub"},
    },
    "custom": {
        "label": "Custom Template",
        "pattern": None,  # user provides the full template string
        "stream_map": {"main": 0, "sub": 1},
    },
    "generic": {
        "label": "Generic (full RTSP URL)",
        "pattern": "{rtsp_url}",
        "stream_map": {},
    },
}

# Map brand → default template key
BRAND_DEFAULT_TEMPLATE: dict[str, str] = {
    "dahua": "dahua",
    "hikvision": "hikvision_v1",
    "generic": "generic",
}


def get_template_choices() -> list[tuple[str, str]]:
    """Return [(key, label), ...] for UI combo boxes."""
    return [(k, v["label"]) for k, v in TEMPLATES.items()]


def build_channel_url(
    brand: str,
    ip: str,
    port: int,
    username: str,
    password: str,
    channel_num: int,
    stream_pref: str = "sub",
    template_key: str | None = None,
    custom_template: str | None = None,
    rtsp_url: str = "",
) -> str:
    """
    Build and return a complete RTSP URL for a recorder channel.
    Credentials are URL-encoded to handle special characters safely.

    Parameters
    ----------
    brand : str
        "dahua" | "hikvision" | "generic"
    template_key : str | None
        Override the brand default.  "custom" uses custom_template.
    custom_template : str | None
        User-supplied format string when template_key="custom".
    """
    key = template_key or BRAND_DEFAULT_TEMPLATE.get(brand, "generic")
    tmpl = TEMPLATES.get(key, TEMPLATES["generic"])

    pattern = tmpl["pattern"]
    if key == "custom":
        if not custom_template:
            raise ValueError("custom_template is required when template_key='custom'")
        pattern = custom_template

    stream_map = tmpl.get("stream_map", {})
    stream_val = stream_map.get(stream_pref, 0)

    enc_user = urllib.parse.quote(username, safe="")
    enc_pass = urllib.parse.quote(password, safe="")

    # hikvision_v1 uses a compound channel_id
    channel_id = channel_num * 100 + (stream_map.get(stream_pref, 1) if key == "hikvision_v1" else 1)

    url = pattern.format(
        user=enc_user,
        password=enc_pass,
        ip=ip,
        port=port,
        ch=channel_num,
        stream=stream_val,
        channel_id=channel_id,
        stream_tag=stream_val,
        rtsp_url=rtsp_url,
    )
    return url


def build_camera_url(rtsp_url: str, username: str = "", password: str = "") -> str:
    """
    Build URL for a direct camera.  If credentials are provided and not already
    embedded in rtsp_url, they are injected.
    """
    if not username and not password:
        return rtsp_url
    # Check if credentials already embedded
    if "@" in rtsp_url.split("://", 1)[-1]:
        return rtsp_url
    # Inject credentials
    scheme, rest = rtsp_url.split("://", 1)
    enc_user = urllib.parse.quote(username, safe="")
    enc_pass = urllib.parse.quote(password, safe="")
    return f"{scheme}://{enc_user}:{enc_pass}@{rest}"


def sanitize_url_for_log(url: str) -> str:
    """Strip credentials from a URL before logging."""
    try:
        parsed = urllib.parse.urlparse(url)
        safe = parsed._replace(netloc=f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname)
        return urllib.parse.urlunparse(safe)
    except Exception:
        return "<url>"
