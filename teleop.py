"""
Data collection via teleoperation.
Operator controls the arm with keyboard (or spacemouse if available).
Each episode is saved as a dict of numpy arrays under datasets/episode_XXXX.npz

Observation space per timestep:
  wrist_img  : (H, W, 3) uint8
  stand_img  : (H, W, 3) uint8
  joints_deg : (6,)  float32 — arm joint angles
  gripper_w  : ()    float32 — gripper width fraction [0,1]

Action space per timestep:
  joints_deg : (6,)  float32
  gripper_w  : ()    float32

Usage:
  python teleop.py --episodes 20 --save_dir datasets/

Controls:
  q/a  : joint 1 ± 5°    w/s  : joint 2 ± 5°
  e/d  : joint 3 ± 5°    r/f  : joint 4 ± 5°
  t/g  : joint 5 ± 5°    y/h  : joint 6 ± 5°
  o    : open gripper     c    : close gripper
  SPACE: save step        ENTER: end episode     ESC: quit
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List

import cv2
import numpy as np

from hardware.arm_controller import ArmController
from hardware.camera_manager import CameraManager
from hardware.gripper_controller import GripperController

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("teleop")

KEY_DELTA  = 5.0          # degrees per keypress
JOINT_KEYS = {
    ord("q"): (0, +KEY_DELTA), ord("a"): (0, -KEY_DELTA),
    ord("w"): (1, +KEY_DELTA), ord("s"): (1, -KEY_DELTA),
    ord("e"): (2, +KEY_DELTA), ord("d"): (2, -KEY_DELTA),
    ord("r"): (3, +KEY_DELTA), ord("f"): (3, -KEY_DELTA),
    ord("t"): (4, +KEY_DELTA), ord("g"): (4, -KEY_DELTA),
    ord("y"): (5, +KEY_DELTA), ord("h"): (5, -KEY_DELTA),
}


def collect_episode(arm: ArmController, gripper: GripperController,
                    cameras: CameraManager, ep_id: int,
                    save_dir: str) -> int:
    """Collect one episode. Returns number of steps saved."""
    arm.go_to_observe(blocking=True)
    gripper.open(blocking=True)

    steps: List[Dict[str, np.ndarray]] = []
    angles = arm.get_joint_angles_deg()
    gripper_w = gripper.get_width_fraction()

    logger.info("Episode %d started. Press SPACE to record, ENTER to save, ESC to abort.", ep_id)

    while True:
        raw = cameras.get_raw_frames()
        wrist_bgr = raw["wrist"]
        stand_bgr = raw["stand"]

        # Build display
        disp_w = cv2.resize(wrist_bgr, (320, 240)) if wrist_bgr is not None else np.zeros((240,320,3),np.uint8)
        disp_s = cv2.resize(stand_bgr, (320, 240)) if stand_bgr is not None else np.zeros((240,320,3),np.uint8)
        disp = np.hstack([disp_w, disp_s])
        cv2.putText(disp, f"Ep {ep_id} | Steps: {len(steps)} | SPACE=record ENTER=save ESC=abort",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        cv2.imshow("Teleop", disp)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:   # ESC
            logger.info("Episode aborted.")
            return 0

        if key == 13:   # ENTER — end episode
            break

        # Gripper
        if key == ord("o"):
            gripper_w = 1.0
            gripper.open(speed=150)
        elif key == ord("c"):
            gripper_w = 0.0
            gripper.close(speed=150)

        # Joints
        if key in JOINT_KEYS:
            jidx, delta = JOINT_KEYS[key]
            angles[jidx] += delta
            arm.set_joint_angles_deg(angles, speed=200)

        # Record step
        if key == ord(" "):
            angles = arm.get_joint_angles_deg()
            gripper_w = gripper.get_width_fraction()
            step = {
                "wrist_img":  wrist_bgr.copy() if wrist_bgr is not None else np.zeros((480,640,3),np.uint8),
                "stand_img":  stand_bgr.copy() if stand_bgr is not None else np.zeros((480,640,3),np.uint8),
                "joints_deg": np.array(angles, dtype=np.float32),
                "gripper_w":  np.float32(gripper_w),
            }
            steps.append(step)
            logger.info("  Recorded step %d", len(steps))

    if len(steps) < 2:
        logger.warning("Too few steps (%d). Episode discarded.", len(steps))
        return 0

    # Build action labels: action[t] = obs[t+1]
    # (last step has no next — drop it)
    n = len(steps) - 1
    save_dict = {
        "wrist_imgs":  np.stack([s["wrist_img"]  for s in steps[:n]]),
        "stand_imgs":  np.stack([s["stand_img"]  for s in steps[:n]]),
        "joints_deg":  np.stack([s["joints_deg"] for s in steps[:n]]),
        "gripper_w":   np.stack([s["gripper_w"]  for s in steps[:n]]),
        "act_joints":  np.stack([s["joints_deg"] for s in steps[1:n+1]]),
        "act_gripper": np.stack([s["gripper_w"]  for s in steps[1:n+1]]),
    }

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"episode_{ep_id:04d}.npz")
    np.savez_compressed(path, **save_dict)
    logger.info("Saved episode %d  (%d steps) → %s", ep_id, n, path)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=20)
    parser.add_argument("--save_dir",  type=str, default="datasets")
    parser.add_argument("--arm_cfg",   type=str, default="config/arm_config.yaml")
    parser.add_argument("--cam_cfg",   type=str, default="config/camera_config.yaml")
    args = parser.parse_args()

    arm     = ArmController(args.arm_cfg)
    gripper = GripperController(args.arm_cfg)
    cameras = CameraManager(args.cam_cfg)

    arm.connect()
    gripper.connect()
    cameras.start()

    ep_id     = 0
    ep_counts = []

    try:
        while ep_id < args.episodes:
            n = collect_episode(arm, gripper, cameras, ep_id, args.save_dir)
            ep_counts.append(n)
            ep_id += 1

        logger.info("Collection complete. %d episodes, %d total steps.",
                    len(ep_counts), sum(ep_counts))
    finally:
        cameras.stop()
        arm.disconnect()
        gripper.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
