"""
net_discovery.py — ONVIF WS-Discovery and profile resolution.

Requires: onvif-zeep  (pip install onvif-zeep)
          wsdiscovery (pip install WSDiscovery)

If either library is missing, discovery returns an empty list with an error
message — the UI should handle this gracefully and show "Add manually" fallback.
"""

import logging
import socket
from typing import Callable

log = logging.getLogger(__name__)


def _local_ip() -> str:
    """Best-effort local IP address (used for diagnostics display)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def discover_onvif_devices(
    timeout: float = 5.0,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[dict], str]:
    """
    Send a WS-Discovery probe and return discovered ONVIF devices.

    Parameters
    ----------
    timeout      : float  seconds to wait for responses
    progress_cb  : called with status strings during discovery

    Returns
    -------
    (devices, error_message)
    devices : list of {
        "ip"         : str,
        "name"       : str,
        "xaddr"      : str   (ONVIF service URL),
        "types"      : str   (device type string)
    }
    error_message : str  (empty on success)
    """

    def _cb(msg: str) -> None:
        log.debug("Discovery: %s", msg)
        if progress_cb:
            progress_cb(msg)

    devices: list[dict] = []

    # --- Try wsdiscovery ---
    try:
        from wsdiscovery import WSDiscovery, QName, Scope  # type: ignore
        _cb("Starting WS-Discovery probe…")

        wsd = WSDiscovery()
        wsd.start()

        # NetworkVideoTransmitter or any ONVIF device
        services = wsd.searchServices(timeout=timeout)
        wsd.stop()

        _cb(f"WS-Discovery found {len(services)} service(s)")

        for svc in services:
            xaddrs = svc.getXAddrs()
            if not xaddrs:
                continue
            xaddr = xaddrs[0]
            # Extract IP from xaddr
            try:
                from urllib.parse import urlparse
                ip = urlparse(xaddr).hostname or ""
            except Exception:
                ip = ""
            types_str = " ".join(str(t) for t in svc.getTypes())
            devices.append(
                {
                    "ip": ip,
                    "name": svc.getEPR() or ip,
                    "xaddr": xaddr,
                    "types": types_str,
                }
            )

        return devices, ""

    except ImportError:
        log.warning("wsdiscovery not installed; ONVIF discovery unavailable")
        return [], "wsdiscovery library not installed. Run: pip install WSDiscovery"
    except Exception as exc:
        log.warning("WS-Discovery failed: %s", exc)
        return [], f"Discovery error: {exc}"


def resolve_rtsp_uri(
    xaddr: str,
    username: str,
    password: str,
    profile_index: int = 0,
) -> tuple[str | None, str]:
    """
    Connect to an ONVIF device and retrieve the RTSP stream URI.

    Returns
    -------
    (rtsp_url or None, error_message)
    """
    try:
        from onvif import ONVIFCamera  # type: ignore
        from urllib.parse import urlparse

        parsed = urlparse(xaddr)
        host = parsed.hostname or xaddr
        port = parsed.port or 80

        log.debug("Connecting ONVIF: %s:%d", host, port)

        cam = ONVIFCamera(host, port, username, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()

        if not profiles:
            return None, "No media profiles found on device"

        idx = min(profile_index, len(profiles) - 1)
        profile = profiles[idx]

        req = media.create_type("GetStreamUri")
        req.StreamSetup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"},
        }
        req.ProfileToken = profile.token
        uri_obj = media.GetStreamUri(req)
        rtsp_url = uri_obj.Uri

        log.info("ONVIF resolved RTSP for %s: %s", host, rtsp_url.split("@")[-1])
        return rtsp_url, ""

    except ImportError:
        return None, "onvif-zeep library not installed. Run: pip install onvif-zeep"
    except Exception as exc:
        log.warning("ONVIF resolution failed: %s", exc)
        return None, f"ONVIF error: {exc}"


def get_network_info() -> dict:
    """Return local network info for the diagnostics screen."""
    local_ip = _local_ip()
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    return {"local_ip": local_ip, "hostname": hostname}
