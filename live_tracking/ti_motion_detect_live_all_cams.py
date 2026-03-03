#!/usr/bin/env python3
"""
Launch TI live motion detection on RTSP cameras in the fixed IP range
192.168.1.2 .. 192.168.1.134.

This script starts one process per camera by invoking:
  Live_Tracking/ti_motion_detect_live_rtsp.py
"""

import argparse
import concurrent.futures
import multiprocessing as mp
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import quote

import cv2


RTSP_USERNAME = "admin"
RTSP_PASSWORD = "Tlgcctv@786"
RTSP_PORT = 554
RTSP_CHANNEL = 1
RTSP_SUBTYPE = 0
IP_PREFIX = "192.168.1"
START_OCTET = 2
END_OCTET = 134


def build_urls() -> list[str]:
    urls = []
    user = quote(RTSP_USERNAME, safe="")
    password = quote(RTSP_PASSWORD, safe="")
    for octet in range(START_OCTET, END_OCTET + 1):
        ip = f"{IP_PREFIX}.{octet}"
        url = (
            f"rtsp://{user}:{password}@{ip}:{RTSP_PORT}/cam/realmonitor"
            f"?channel={RTSP_CHANNEL}&subtype={RTSP_SUBTYPE}"
        )
        urls.append(url)
    return urls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TI live motion detection on hardcoded RTSP IP range",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-cams", type=int, default=0, help="Limit number of cameras (0 = all)")
    p.add_argument("--stagger", type=float, default=0.2, help="Delay between process starts (seconds)")
    p.add_argument("--probe-timeout", type=float, default=1.0, help="Seconds for TCP/feed probe timeout")
    p.add_argument("--probe-workers", type=int, default=16, help="Parallel workers for camera feed probing")

    # Defaults match your provided tuning
    p.add_argument("--clahe", action="store_true", default=True)
    p.add_argument("--clahe-clip", type=float, default=2.5)
    p.add_argument("--clahe-tile", type=int, default=12)
    p.add_argument("--no-temporal-smooth", action="store_true", default=True)
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--min-area", type=int, default=500)
    p.add_argument("--max-area", type=int, default=50000)
    p.add_argument("--min-density", type=float, default=0.15)
    p.add_argument("--persistence", type=int, default=4)
    p.add_argument("--min-displacement", type=float, default=30.0)
    p.add_argument("--spatial-tol", type=int, default=20)
    p.add_argument("--max-absent", type=int, default=4)
    p.add_argument("--no-candidates", action="store_true", default=True)
    p.add_argument("--stabilize", action="store_true", default=False)
    p.add_argument("--display", action="store_true", default=True)
    p.add_argument("--no-display", dest="display", action="store_false")
    return p.parse_args()


def _tcp_open(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _stream_has_frame(url: str, timeout_s: float) -> bool:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return False
    try:
        # Backend-dependent; ignored when unsupported.
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_s * 1000))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_s * 1000))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    ok = False
    for _ in range(3):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            ok = True
            break
        time.sleep(0.05)
    cap.release()
    return ok


def is_camera_live(url: str, timeout_s: float) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    port = parsed.port or RTSP_PORT
    if not _tcp_open(parsed.hostname, port, timeout_s):
        return False
    return _stream_has_frame(url, timeout_s)


def worker(url: str, args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "Live_Tracking" / "ti_motion_detect_live_rtsp.py"

    cmd = [
        sys.executable,
        str(runner),
        "--input",
        url,
        "--clahe",
        "--clahe-clip",
        str(args.clahe_clip),
        "--clahe-tile",
        str(args.clahe_tile),
        "--threshold",
        str(args.threshold),
        "--min-area",
        str(args.min_area),
        "--max-area",
        str(args.max_area),
        "--min-density",
        str(args.min_density),
        "--persistence",
        str(args.persistence),
        "--min-displacement",
        str(args.min_displacement),
        "--spatial-tol",
        str(args.spatial_tol),
        "--max-absent",
        str(args.max_absent),
    ]

    if args.no_temporal_smooth:
        cmd.append("--no-temporal-smooth")
    if args.no_candidates:
        cmd.append("--no-candidates")
    if args.stabilize:
        cmd.append("--stabilize")
    if not args.display:
        cmd.append("--no-display")

    subprocess.run(cmd, check=False)


def main() -> None:
    args = parse_args()
    urls = build_urls()

    if args.max_cams > 0:
        urls = urls[: args.max_cams]

    if not urls:
        print("No URLs generated.")
        return

    print(f"Probing {len(urls)} camera(s) for live feed...")
    live_urls = []
    workers = max(1, args.probe_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(is_camera_live, url, args.probe_timeout): url for url in urls}
        for fut in concurrent.futures.as_completed(futures):
            url = futures[fut]
            try:
                if fut.result():
                    live_urls.append(url)
            except Exception:
                pass

    urls = sorted(live_urls)
    if not urls:
        print("No live camera feeds detected in the configured IP range.")
        return

    print(f"Starting motion detection for {len(urls)} live camera(s)")
    print(f"IP range: {IP_PREFIX}.{START_OCTET} -> {IP_PREFIX}.{END_OCTET}")

    procs = []
    for url in urls:
        p = mp.Process(target=worker, args=(url, args))
        p.start()
        procs.append(p)
        if args.stagger > 0:
            time.sleep(args.stagger)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping all camera processes...")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=1.0)


if __name__ == "__main__":
    main()
