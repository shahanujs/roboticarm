"""
Live camera feed with orange detection overlay.

Usage:
  python scripts/view_cameras.py

Behavior:
  - If GUI display is available, opens OpenCV window.
  - If no DISPLAY is available (headless/SSH), starts a web viewer on :8080.
"""

import argparse
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from hardware.camera_manager import CameraManager
from perception.orange_detector import OrangeDetector


def get_composite_frame(cameras: CameraManager, detector: OrangeDetector,
                        detect: bool = True) -> np.ndarray:
    raw = cameras.get_raw_frames()
    wrist_bgr = raw["wrist"] if raw["wrist"] is not None else np.zeros((480, 640, 3), np.uint8)
    stand_bgr = raw["stand"] if raw["stand"] is not None else np.zeros((480, 640, 3), np.uint8)

    if detect:
        wrist_dets = detector.detect(wrist_bgr)
        stand_dets = detector.detect(stand_bgr)
        wrist_bgr = detector.draw_detections(wrist_bgr, wrist_dets)
        stand_bgr = detector.draw_detections(stand_bgr, stand_dets)

    disp = np.hstack([
        cv2.resize(wrist_bgr, (480, 360)),
        cv2.resize(stand_bgr, (480, 360)),
    ])
    cv2.putText(disp, "Wrist", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(disp, "Stand", (490, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return disp


def run_gui(cameras: CameraManager, detector: OrangeDetector) -> None:
    print("GUI mode: Press Q to quit, D to toggle detection")
    detect = True
    while True:
        disp = get_composite_frame(cameras, detector, detect)
        cv2.imshow("Cameras", disp)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("d"):
            detect = not detect
    cv2.destroyAllWindows()


def run_web(cameras: CameraManager, detector: OrangeDetector,
            host: str = "0.0.0.0", port: int = 8080) -> None:
    print(f"Headless mode: open http://<orin-ip>:{port} in your browser")
    print("Press Ctrl+C to stop")

    state = {
        "jpg": b"",
        "lock": threading.Lock(),
        "detect": True,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = f"""
                <html><head><title>RoboArm Cameras</title></head>
                <body style='background:#111;color:#eee;font-family:sans-serif'>
                  <h2>RoboArm Camera Viewer</h2>
                  <p>Refresh page if stream is blank for first second.</p>
                  <img src='/stream.mjpg' width='960' />
                </body></html>
                """.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    with state["lock"]:
                        jpg = state["jpg"]
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, *args, **kwargs):
            return

    server = HTTPServer((host, port), Handler)
    stop_evt = threading.Event()

    def produce_frames():
        while not stop_evt.is_set():
            disp = get_composite_frame(cameras, detector, state["detect"])
            ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with state["lock"]:
                    state["jpg"] = buf.tobytes()
            time.sleep(0.03)

    producer = threading.Thread(target=produce_frames, daemon=True)
    producer.start()

    try:
        server.serve_forever()
    finally:
        stop_evt.set()
        server.shutdown()
        server.server_close()


def _default_mode() -> str:
    # In SSH/headless sessions, prefer web mode to avoid Qt/X11 crashes.
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return "web"
    if not os.environ.get("DISPLAY"):
        return "web"
    return "gui"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "gui", "web"], default="auto")
    ap.add_argument("--host", default="0.0.0.0", help="Web server host (web mode)")
    ap.add_argument("--port", type=int, default=8080, help="Web server port (web mode)")
    args = ap.parse_args()

    cameras = CameraManager("config/camera_config.yaml")
    detector = OrangeDetector("config/camera_config.yaml")
    cameras.start()
    try:
        mode = _default_mode() if args.mode == "auto" else args.mode
        if mode == "gui":
            try:
                run_gui(cameras, detector)
            except cv2.error:
                print("GUI mode failed; falling back to web mode")
                run_web(cameras, detector, host=args.host, port=args.port)
        else:
            run_web(cameras, detector, host=args.host, port=args.port)
    finally:
        cameras.stop()


if __name__ == "__main__":
    main()
