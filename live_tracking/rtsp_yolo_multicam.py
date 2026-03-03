import argparse
from collections import defaultdict, deque
import multiprocessing as mp
import time
import cv2

DEFAULT_RTSP_URL = "rtsp://admin:Tlgcctv@786@192.168.1.121:554/cam/realmonitor?channel=1&subtype=0"


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO tracking on RTSP cameras in parallel.")
    p.add_argument("--base-ip", default="192.168.1.121", help="Any IP in the target subnet (used to derive prefix)")
    p.add_argument("--start-octet", type=int, default=118, help="Start of IP range (last octet)")
    p.add_argument("--end-octet", type=int, default=126, help="End of IP range (last octet)")
    p.add_argument("--username", default="admin", help="RTSP username")
    p.add_argument("--password", default="Tlgcctv@786", help="RTSP password")
    p.add_argument("--port", type=int, default=554, help="RTSP port")
    p.add_argument("--channel", type=int, default=1, help="Camera channel")
    p.add_argument("--subtype", type=int, default=0, help="Stream subtype (0 = main, 1 = substream)")
    p.add_argument("--model", default="yolov8x.pt", help="YOLO model path/name")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="", help="Device for inference (e.g. 'cpu', '0', '0,1'). Empty = auto")
    p.add_argument("--half", action="store_true", help="Use FP16 inference when supported")
    p.add_argument("--frame-stride", type=int, default=1, help="Run detection every N frames")
    p.add_argument("--trail-length", type=int, default=30, help="Number of recent tracked points to draw per object")
    p.add_argument("--reconnect-delay", type=float, default=1.0, help="Seconds to wait before reconnecting")
    p.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking backend",
    )
    p.add_argument("--url", default=DEFAULT_RTSP_URL, help="Single RTSP URL (used if --urls is not provided)")
    p.add_argument("--use-generated-range", action="store_true", help="Use generated IP range instead of --url")
    p.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="Optional explicit RTSP URLs (overrides --url and generated range)",
    )
    return p.parse_args()


def build_urls(base_ip, start_octet, end_octet, username, password, port, channel, subtype):
    parts = base_ip.split(".")
    if len(parts) != 4:
        raise ValueError("base-ip must be a valid IPv4 address")
    prefix = ".".join(parts[:3])
    urls = []
    for octet in range(start_octet, end_octet + 1):
        ip = f"{prefix}.{octet}"
        url = (
            f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor"
            f"?channel={channel}&subtype={subtype}"
        )
        urls.append(url)
    return urls


def open_capture(url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


def camera_worker(
    index,
    url,
    model_path,
    conf,
    iou,
    imgsz,
    device,
    half,
    frame_stride,
    trail_length,
    reconnect_delay,
    tracker,
    stop_event,
):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("Failed to import ultralytics. Install with: pip install ultralytics opencv-python")
        raise SystemExit(1) from exc

    window_name = f"Cam {index} {url.split('@')[-1].split('/')[0]}"
    model = YOLO(model_path)

    cap = None
    frame_idx = 0
    track_history = defaultdict(lambda: deque(maxlen=max(2, trail_length)))

    track_kwargs = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
        "persist": True,
        "tracker": tracker,
    }
    if device:
        track_kwargs["device"] = device
    if half:
        track_kwargs["half"] = True

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = open_capture(url)
            if cap is None or not cap.isOpened():
                time.sleep(reconnect_delay)
                continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = None
            time.sleep(reconnect_delay)
            continue

        frame_idx += 1
        if frame_stride > 1 and (frame_idx % frame_stride) != 0:
            display = frame
        else:
            results = model.track(frame, **track_kwargs)
            result = results[0]
            display = result.plot()

            boxes = getattr(result, "boxes", None)
            if boxes is not None and getattr(boxes, "id", None) is not None and len(boxes) > 0:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.int().cpu().tolist()

                for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    track_history[int(track_id)].append((cx, cy))

                for track_id, points in track_history.items():
                    if len(points) < 2:
                        continue
                    pts = list(points)
                    color = (
                        (37 * int(track_id)) % 255,
                        (17 * int(track_id)) % 255,
                        (97 * int(track_id)) % 255,
                    )
                    for j in range(1, len(pts)):
                        p1 = pts[j - 1]
                        p2 = pts[j]
                        thickness = max(1, 4 - (len(pts) - j) // 8)
                        cv2.line(display, p1, p2, color, thickness)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break

    if cap is not None:
        cap.release()
    cv2.destroyWindow(window_name)


def main():
    args = parse_args()

    if args.urls:
        urls = args.urls
    elif args.use_generated_range:
        urls = build_urls(
            args.base_ip,
            args.start_octet,
            args.end_octet,
            args.username,
            args.password,
            args.port,
            args.channel,
            args.subtype,
        )
    elif args.url:
        urls = [args.url]
    else:
        urls = []

    if not urls:
        raise SystemExit("No RTSP URLs provided or generated")

    stop_event = mp.Event()
    processes = []

    for i, url in enumerate(urls, start=1):
        p = mp.Process(
            target=camera_worker,
            args=(
                i,
                url,
                args.model,
                args.conf,
                args.iou,
                args.imgsz,
                args.device,
                args.half,
                args.frame_stride,
                args.trail_length,
                args.reconnect_delay,
                args.tracker,
                stop_event,
            ),
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=1.0)


if __name__ == "__main__":
    main()
