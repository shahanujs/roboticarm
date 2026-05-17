"""
Orange detection pipeline.
Two modes:
  1. HSV colour segmentation — fast, runs at camera FPS, used at inference time.
  2. YOLOv8-based detection   — accurate, used to bootstrap colour thresholds
                                (requires: pip install ultralytics).

Outputs:
  - Pixel bounding box in camera frame
  - Estimated 3-D centroid in camera frame (using depth from size heuristic)
  - Estimated 3-D centroid in robot base frame (for stand_cam with known extrinsic)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected orange."""
    bbox_xyxy: Tuple[int, int, int, int]   # pixel box [x1,y1,x2,y2]
    centroid_px: Tuple[int, int]           # (u, v)
    radius_px: float                       # approximate radius in pixels
    confidence: float                      # 0–1
    xyz_camera: Optional[np.ndarray] = None   # 3-D in camera frame
    xyz_base: Optional[np.ndarray]   = None   # 3-D in robot base frame


class OrangeDetector:
    """
    Detects oranges in a BGR image via HSV thresholding.
    Optionally uses YOLOv8 for higher accuracy (auto-detected at init).
    """

    ORANGE_REAL_DIAMETER_M = 0.075   # ~7.5 cm diameter orange

    def __init__(self, config_path: str = "config/camera_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        oh = cfg["orange_hsv"]
        self._hsv_lower = np.array(oh["lower"], dtype=np.uint8)
        self._hsv_upper = np.array(oh["upper"], dtype=np.uint8)
        self._min_area  = oh["min_area_px"]
        self._max_area  = oh["max_area_px"]

        # Try to load YOLOv8
        self._yolo = None
        try:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n.pt")   # nano model; auto-downloads
            logger.info("YOLOv8 loaded for orange detection.")
        except Exception:
            logger.info("YOLOv8 not available; using HSV detection only.")

        # Intrinsics for depth estimation (wrist cam defaults)
        wc = cfg["wrist_camera"]
        self._fx = wc["fx"]
        self._fy = wc["fy"]
        self._cx = wc["cx"]
        self._cy = wc["cy"]

    # ── Main detection entry ──────────────────────────────────────────────────

    def detect(self, bgr: np.ndarray,
               camera_K: Optional[np.ndarray] = None,
               T_base_cam: Optional[np.ndarray] = None,
               use_yolo: bool = False) -> List[Detection]:
        """
        Detect oranges in bgr image.

        Args:
            bgr        : OpenCV BGR image (H,W,3 uint8)
            camera_K   : 3×3 intrinsic matrix; if None, uses wrist cam defaults
            T_base_cam : 4×4 transform base←camera; if provided, fills xyz_base
            use_yolo   : Force YOLO detection even if HSV would suffice

        Returns:
            List of Detection objects sorted by confidence (desc).
        """
        if use_yolo and self._yolo is not None:
            detections = self._detect_yolo(bgr)
        else:
            detections = self._detect_hsv(bgr)

        if camera_K is None:
            camera_K = np.array([
                [self._fx, 0, self._cx],
                [0, self._fy, self._cy],
                [0, 0, 1],
            ])

        # Estimate 3-D positions
        for d in detections:
            d.xyz_camera = self._estimate_xyz_camera(d, camera_K)
            if T_base_cam is not None and d.xyz_camera is not None:
                d.xyz_base = self._camera_to_base(d.xyz_camera, T_base_cam)

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    # ── HSV segmentation ──────────────────────────────────────────────────────

    def _detect_hsv(self, bgr: np.ndarray) -> List[Detection]:
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self._min_area <= area <= self._max_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
            circularity = (4 * math.pi * area) / (cv2.arcLength(cnt, True) ** 2 + 1e-9)

            # Score by area and circularity
            conf = min(1.0, circularity * (area / self._max_area) ** 0.3)
            detections.append(Detection(
                bbox_xyxy=(x, y, x + w, y + h),
                centroid_px=(int(cx_f), int(cy_f)),
                radius_px=radius,
                confidence=conf,
            ))
        return detections

    # ── YOLOv8 detection ──────────────────────────────────────────────────────

    def _detect_yolo(self, bgr: np.ndarray) -> List[Detection]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._yolo.predict(rgb, classes=[49],   # COCO class 49 = orange
                                     conf=0.25, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = min(x2 - x1, y2 - y1) / 2.0
                detections.append(Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    centroid_px=(cx, cy),
                    radius_px=radius,
                    confidence=float(box.conf),
                ))
        return detections

    # ── 3-D estimation ────────────────────────────────────────────────────────

    def _estimate_xyz_camera(self, d: Detection,
                              K: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from apparent size using known real diameter.
        z = f * D_real / (2 * r_px)
        """
        if d.radius_px < 2:
            return None
        f = (K[0, 0] + K[1, 1]) / 2.0
        z = f * self.ORANGE_REAL_DIAMETER_M / (2.0 * d.radius_px)
        u, v = d.centroid_px
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.array([x, y, z])

    @staticmethod
    def _camera_to_base(xyz_cam: np.ndarray,
                        T_base_cam: np.ndarray) -> np.ndarray:
        p = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0])
        return (T_base_cam @ p)[:3]

    # ── Visualisation ─────────────────────────────────────────────────────────

    @staticmethod
    def draw_detections(bgr: np.ndarray,
                        detections: List[Detection]) -> np.ndarray:
        vis = bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.circle(vis, d.centroid_px, int(d.radius_px), (0, 200, 255), 2)
            label = f"orange {d.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            if d.xyz_base is not None:
                xyz_str = f"xyz={d.xyz_base[0]:.3f},{d.xyz_base[1]:.3f},{d.xyz_base[2]:.3f}"
                cv2.putText(vis, xyz_str, (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        return vis
