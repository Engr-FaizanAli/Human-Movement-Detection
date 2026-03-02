"""
rtsp_test.py — Quick RTSP connectivity tester.

Opens a VideoCapture, reads one frame, returns result dict.
Credentials are never logged.
"""

import logging
import time

from utils.rtsp_templates import sanitize_url_for_log

log = logging.getLogger(__name__)


def test_rtsp(url: str, timeout_sec: float = 6.0) -> dict:
    """
    Attempt to open a VideoCapture and read one frame.

    Returns
    -------
    dict with keys:
        ok        : bool
        width     : int
        height    : int
        fps       : float
        error     : str  (empty string on success)
        latency_ms: float  (ms to open + read first frame)
    """
    safe_url = sanitize_url_for_log(url)
    log.debug("Testing RTSP: %s", safe_url)

    result = {
        "ok": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "error": "",
        "latency_ms": 0.0,
    }

    try:
        import cv2

        t0 = time.monotonic()

        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set read timeout via open timeout property where available
        deadline = time.monotonic() + timeout_sec
        opened = cap.isOpened()
        if not opened:
            result["error"] = "Failed to open stream"
            log.warning("RTSP test failed (could not open): %s", safe_url)
            return result

        # Try reading one frame
        ok = False
        frame = None
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok:
                break
            time.sleep(0.1)

        elapsed_ms = (time.monotonic() - t0) * 1000
        result["latency_ms"] = round(elapsed_ms, 1)

        if ok and frame is not None:
            result["ok"] = True
            result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result["fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
            log.info(
                "RTSP test OK: %s  %dx%d %.1ffps  %.0fms",
                safe_url,
                result["width"],
                result["height"],
                result["fps"],
                elapsed_ms,
            )
        else:
            result["error"] = "Stream opened but no frame received within timeout"
            log.warning("RTSP test timeout: %s", safe_url)

        cap.release()

    except Exception as exc:
        result["error"] = str(exc)
        log.warning("RTSP test exception for %s: %s", safe_url, exc)

    return result
