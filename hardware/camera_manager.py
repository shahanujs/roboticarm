"""
Dual-camera manager for SO-ARM101 setup.
  - wrist_cam  : USB camera mounted on end-effector (/dev/video0)
  - stand_cam  : Fixed overview camera (/dev/video2)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

Frame = np.ndarray          # H x W x 3, uint8, BGR


class SingleCamera:
    """Thread-safe wrapper around a V4L2 / OpenCV camera."""

    def __init__(self, device_id: int, width: int, height: int,
                 fps: int, name: str,
                 rotate_180: bool = False,
                 flip_horizontal: bool = False,
                 flip_vertical: bool = False):
        self.name = name
        self._cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # avoid stale frames
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {name} at /dev/video{device_id}")

        self._rotate_180 = rotate_180
        self._flip_horizontal = flip_horizontal
        self._flip_vertical = flip_vertical

        self._lock   = threading.Lock()
        self._latest: Optional[Frame] = None
        self._ts     = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name=f"cam_{self.name}")
        self._thread.start()
        # Wait for first frame
        t0 = time.monotonic()
        while self._latest is None and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
        logger.info("Camera '%s' started.", self.name)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def get_frame(self) -> Tuple[Optional[Frame], float]:
        """Return (frame_bgr, timestamp_sec). Thread-safe."""
        with self._lock:
            return (self._latest.copy() if self._latest is not None else None,
                    self._ts)

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera '%s' read failed.", self.name)
                time.sleep(0.1)
                continue

            if self._rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self._flip_horizontal:
                frame = cv2.flip(frame, 1)
            if self._flip_vertical:
                frame = cv2.flip(frame, 0)

            with self._lock:
                self._latest = frame
                self._ts = time.monotonic()


class CameraManager:
    """
    Manages wrist camera and stand camera simultaneously.
    Provides synchronised frame pairs for policy input.
    """

    def __init__(self, config_path: str = "config/camera_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        wc = cfg["wrist_camera"]
        sc = cfg["stand_camera"]
        self._obs_size: Tuple[int, int] = tuple(cfg["obs_image_size"])  # (H,W)

        self.wrist_cam = SingleCamera(
            device_id=wc["device_id"],
            width=wc["width"], height=wc["height"], fps=wc["fps"],
            name=wc["name"],
            rotate_180=wc.get("rotate_180", False),
            flip_horizontal=wc.get("flip_horizontal", False),
            flip_vertical=wc.get("flip_vertical", False),
        )
        self.stand_cam = SingleCamera(
            device_id=sc["device_id"],
            width=sc["width"], height=sc["height"], fps=sc["fps"],
            name=sc["name"],
            rotate_180=sc.get("rotate_180", False),
            flip_horizontal=sc.get("flip_horizontal", False),
            flip_vertical=sc.get("flip_vertical", False),
        )

        # Intrinsics (for 3-D projection)
        self._wrist_K = np.array([
            [wc["fx"], 0, wc["cx"]],
            [0, wc["fy"], wc["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)
        self._stand_K = np.array([
            [sc["fx"], 0, sc["cx"]],
            [0, sc["fy"], sc["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)
        T_list = sc.get("T_base_cam", np.eye(4).tolist())
        self._T_base_standcam = np.array(T_list)

    def start(self) -> None:
        self.wrist_cam.start()
        self.stand_cam.start()

    def stop(self) -> None:
        self.wrist_cam.stop()
        self.stand_cam.stop()

    def get_obs_frames(self) -> Dict[str, np.ndarray]:
        """
        Returns dict with keys 'wrist' and 'stand'.
        Each value is an RGB float32 [0,1] image resized to obs_image_size (H,W,3).
        """
        wrist_bgr, _ = self.wrist_cam.get_frame()
        stand_bgr, _ = self.stand_cam.get_frame()

        out = {}
        for name, bgr in [("wrist", wrist_bgr), ("stand", stand_bgr)]:
            if bgr is None:
                out[name] = np.zeros((*self._obs_size, 3), dtype=np.float32)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = self._obs_size
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            out[name] = resized.astype(np.float32) / 255.0
        return out

    def get_raw_frames(self) -> Dict[str, Optional[Frame]]:
        return {
            "wrist": self.wrist_cam.get_frame()[0],
            "stand": self.stand_cam.get_frame()[0],
        }

    def project_3d_to_stand_cam(self, xyz_base: np.ndarray) -> Tuple[int, int]:
        """Project a 3-D point in base frame to stand-camera pixel coords."""
        T_cam_base = np.linalg.inv(self._T_base_standcam)
        p_cam = T_cam_base[:3, :3] @ xyz_base + T_cam_base[:3, 3]
        uv = self._stand_K @ p_cam
        return int(uv[0] / uv[2]), int(uv[1] / uv[2])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
