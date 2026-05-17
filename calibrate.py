"""
Camera calibration (intrinsics + extrinsics) and arm-camera hand-eye calibration.

Usage
-----
# Calibrate wrist camera intrinsics (hold checkerboard in front of wrist cam)
python calibrate.py --target wrist_intrinsics

# Calibrate stand camera intrinsics
python calibrate.py --target stand_intrinsics

# Hand-eye calibration (AX=ZB method, calibrate.py moves arm to N poses)
python calibrate.py --target hand_eye
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibrate")

BOARD_SIZE  = (9, 6)      # interior corners (cols, rows)
SQUARE_M    = 0.025       # 25 mm squares


def capture_checkerboard_frames(cap: cv2.VideoCapture,
                                 n_frames: int = 20,
                                 name: str = "camera"
                                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Interactively capture n_frames checkerboard images. Press SPACE to capture."""
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_M

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    captured = 0

    while captured < n_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, BOARD_SIZE, corners, found)
        cv2.putText(vis, f"{name}: {captured}/{n_frames}  SPACE=capture  Q=quit",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        cv2.imshow("Calibrate", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            captured += 1
            logger.info("Captured %d/%d", captured, n_frames)

    cv2.destroyAllWindows()
    return obj_points, img_points


def calibrate_intrinsics(device_id: int, name: str,
                          config_key: str,
                          config_path: str = "config/camera_config.yaml") -> None:
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    obj_pts, img_pts = capture_checkerboard_frames(cap, n_frames=20, name=name)
    cap.release()

    if len(obj_pts) < 5:
        logger.error("Not enough frames (%d). Aborting.", len(obj_pts))
        return

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (640, 480), None, None
    )
    logger.info("Reprojection error: %.4f px", ret)
    logger.info("K=\n%s", K)
    logger.info("dist=%s", dist.ravel())

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg[config_key]["fx"] = float(K[0, 0])
    cfg[config_key]["fy"] = float(K[1, 1])
    cfg[config_key]["cx"] = float(K[0, 2])
    cfg[config_key]["cy"] = float(K[1, 2])
    cfg[config_key]["distortion"] = dist.ravel().tolist()
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info("Intrinsics saved to %s [%s]", config_path, config_key)


def hand_eye_calibration(arm_cfg: str = "config/arm_config.yaml",
                          cam_cfg: str = "config/camera_config.yaml") -> None:
    """
    AX = ZB hand-eye calibration.
    Moves arm to N random poses, captures board image at each,
    solves for T_base_standcam.
    """
    from hardware.arm_controller import ArmController
    from hardware.camera_manager import CameraManager

    arm     = ArmController(arm_cfg)
    cameras = CameraManager(cam_cfg)
    arm.connect()
    cameras.start()

    POSES = [
        [0,  -20, 50, -30, 0, 0],
        [20, -20, 50, -30, 0, 0],
        [-20,-20, 50, -30, 0, 0],
        [0,  -40, 70, -30, 0, 0],
        [0,   0,  40, -40, 0, 0],
        [30, -30, 60, -30, 0, 0],
    ]

    R_gripper2base, t_gripper2base = [], []
    R_target2cam,   t_target2cam   = [], []

    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_M
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    with open(cam_cfg) as f:
        cc = yaml.safe_load(f)
    sc = cc["stand_camera"]
    K  = np.array([[sc["fx"],0,sc["cx"]],[0,sc["fy"],sc["cy"]],[0,0,1]])
    dist = np.array(sc.get("distortion", [0]*5))

    for joints in POSES:
        arm.set_joint_angles_deg(joints, speed=150, blocking=True)
        time.sleep(0.5)

        T_ee = arm.forward_kinematics(joints)
        R_g2b = T_ee[:3,:3]
        t_g2b = T_ee[:3, 3]

        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        if not found:
            logger.warning("Board not found at pose %s", joints)
            continue
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        _, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
        R_t2c, _ = cv2.Rodrigues(rvec)

        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b.reshape(3,1))
        R_target2cam.append(R_t2c)
        t_target2cam.append(tvec)

    cap.release()
    cameras.stop()
    arm.disconnect()

    if len(R_gripper2base) < 3:
        logger.error("Too few valid poses for hand-eye calibration.")
        return

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_base_cam = np.eye(4)
    T_base_cam[:3,:3] = R_cam2base
    T_base_cam[:3, 3] = t_cam2base.ravel()
    logger.info("T_base_standcam =\n%s", T_base_cam)

    with open(cam_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["stand_camera"]["T_base_cam"] = T_base_cam.tolist()
    with open(cam_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info("Hand-eye result saved to %s", cam_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["wrist_intrinsics","stand_intrinsics","hand_eye"],
                        required=True)
    args = parser.parse_args()

    if args.target == "wrist_intrinsics":
        calibrate_intrinsics(0, "Wrist Camera", "wrist_camera")
    elif args.target == "stand_intrinsics":
        calibrate_intrinsics(2, "Stand Camera", "stand_camera")
    elif args.target == "hand_eye":
        hand_eye_calibration()


if __name__ == "__main__":
    main()
