"""
Quick hardware smoke-test — verifies arm, gripper, and cameras are working.
Run this FIRST before any other script.

Usage: python scripts/test_hardware.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import yaml

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_hardware")


def test_cameras():
    logger.info("=== Camera Test ===")
    for dev in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            logger.info("  /dev/video%d : OK  shape=%s", dev, frame.shape)
        else:
            logger.warning("  /dev/video%d : FAILED", dev)


def test_arm_ping():
    logger.info("=== Arm Servo Ping Test ===")
    try:
        from hardware.feetech_bus import FeetechBus
        with open("config/arm_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        arm_port = cfg["arm"]["port"]
        baud = cfg["arm"]["baudrate"]
        arm_ids = [j["id"] for j in cfg["arm"]["joints"]]
        gripper_id = cfg["gripper"]["id"]

        bus = FeetechBus(arm_port, baud)
        for sid in sorted(set(arm_ids + [gripper_id])):
            ok = bus.ping(sid)
            logger.info("  Servo ID %d: %s", sid, "OK" if ok else "NOT FOUND")
        bus.close()
    except Exception as e:
        logger.error("  Arm bus error: %s", e)


def test_arm_motion():
    logger.info("=== Arm Motion Test (small nudge) ===")
    try:
        from hardware.arm_controller import ArmController
        arm = ArmController("config/arm_config.yaml")
        if not arm.connect():
            logger.error("  Arm connect failed")
            return
        angles = arm.get_joint_angles_deg()
        logger.info("  Current angles: %s", [f"{a:.1f}" for a in angles])
        logger.info("  Moving to home…")
        arm.go_home(speed=100, blocking=True)
        arm.disconnect()
        logger.info("  Arm motion: OK")
    except Exception as e:
        logger.error("  Arm motion error: %s", e)


def test_gripper():
    logger.info("=== Gripper Test ===")
    try:
        from hardware.gripper_controller import GripperController
        g = GripperController("config/arm_config.yaml")
        if not g.connect():
            logger.error("  Gripper not found")
            return
        logger.info("  Opening gripper…")
        g.open(blocking=True)
        time.sleep(0.5)
        logger.info("  Closing gripper…")
        g.close(blocking=True)
        time.sleep(0.5)
        g.open(blocking=True)
        g.disconnect()
        logger.info("  Gripper: OK")
    except Exception as e:
        logger.error("  Gripper error: %s", e)


def test_orange_detector():
    logger.info("=== Orange Detector Test ===")
    try:
        from perception.orange_detector import OrangeDetector
        det = OrangeDetector("config/camera_config.yaml")
        # Generate synthetic orange patch
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 60, (0, 128, 255), -1)   # BGR orange
        dets = det.detect(img)
        logger.info("  Synthetic orange detected: %d (expected ≥1)", len(dets))
    except Exception as e:
        logger.error("  Detector error: %s", e)


def main():
    test_cameras()
    test_arm_ping()
    test_arm_motion()
    test_gripper()
    test_orange_detector()
    logger.info("Hardware test complete.")


if __name__ == "__main__":
    main()
