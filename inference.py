"""
Inference / execution loop for SO-ARM101 orange pick task.

Modes
-----
1. policy   : Run trained diffusion policy or ACT end-to-end
2. scripted : Heuristic orange pick using detector + pre-programmed primitives
              (use this while collecting data / before policy is trained)

Usage
-----
# Scripted orange pick (no ML, good for testing hardware)
python inference.py --mode scripted

# Run Diffusion Policy
python inference.py --mode policy --policy diffusion --ckpt checkpoints/diffusion_best.pth

# Run ACT
python inference.py --mode policy --policy act --ckpt checkpoints/act_best.pth
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch

from hardware.arm_controller import ArmController
from hardware.camera_manager import CameraManager
from hardware.gripper_controller import GripperController
from perception.orange_detector import OrangeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference")

IMG_H, IMG_W = 96, 96


# ── Scripted pick primitive ───────────────────────────────────────────────────

def scripted_pick(arm: ArmController, gripper: Optional[GripperController],
                  cameras: CameraManager, detector: OrangeDetector,
                  drop_xyz: np.ndarray,
                  gripper_enabled: bool = False,
                  pick_z: float = 0.05,
                  grasp_width_fraction: float = 0.35,
                  grasp_speed: int = 70) -> bool:
    """
    Detect nearest orange in stand_cam, compute 3-D position,
    execute approach-grasp-lift sequence.
    Returns True if grasp succeeded.
    """
    raw = cameras.get_raw_frames()
    stand_bgr = raw["stand"]
    if stand_bgr is None:
        logger.warning("Stand camera returned None")
        return False

    import yaml
    with open("config/camera_config.yaml") as f:
        cam_cfg = yaml.safe_load(f)
    T_base_cam = np.array(cam_cfg["stand_camera"]["T_base_cam"])
    sc = cam_cfg["stand_camera"]
    K = np.array([[sc["fx"],0,sc["cx"]],[0,sc["fy"],sc["cy"]],[0,0,1]])

    dets = detector.detect(stand_bgr, camera_K=K, T_base_cam=T_base_cam)
    if not dets:
        logger.info("No oranges detected.")
        return False

    # Target: nearest orange by z
    best = min(dets, key=lambda d: d.xyz_base[2] if d.xyz_base is not None else 999)
    if best.xyz_base is None:
        return False

    # Monocular size-based Z can be noisy; use detected XY and a fixed table pick height.
    target_xyz = best.xyz_base.copy()
    target_xyz[2] = pick_z
    logger.info("Target orange XY from vision, using fixed pick_z=%.3f -> XYZ=%.3f, %.3f, %.3f",
                pick_z, target_xyz[0], target_xyz[1], target_xyz[2])

    # Validate target is inside configured workspace before any gripper action.
    ws = getattr(arm, "_workspace", {})
    if ws:
        x_ok = ws["x"][0] <= target_xyz[0] <= ws["x"][1]
        y_ok = ws["y"][0] <= target_xyz[1] <= ws["y"][1]
        z_ok = ws["z"][0] <= target_xyz[2] <= ws["z"][1]
        if not (x_ok and y_ok and z_ok):
            logger.warning(
                "Target outside workspace, skipping: XYZ=%.3f, %.3f, %.3f",
                target_xyz[0], target_xyz[1], target_xyz[2]
            )
            return False

    # ── Approach sequence ─────────────────────────────────────────────────────
    # 1. Pre-grasp: above target
    pre_grasp_xyz = target_xyz.copy()
    pre_grasp_xyz[2] += 0.08
    logger.info("ARM moving to pre-grasp XYZ=%.3f, %.3f, %.3f", *pre_grasp_xyz)
    if not arm.move_cartesian(pre_grasp_xyz):
        logger.warning("ARM failed to reach pre-grasp pose")
        return False
    time.sleep(0.5)

    # Open gripper only after arm reaches pre-grasp.
    if gripper_enabled and gripper is not None:
        gripper.open(blocking=True)

    # 2. Descend to grasp height
    grasp_xyz = target_xyz.copy()
    grasp_xyz[2] += 0.01   # slight offset above centre
    logger.info("ARM descending to grasp XYZ=%.3f, %.3f, %.3f", *grasp_xyz)
    if not arm.move_cartesian(grasp_xyz):
        logger.warning("ARM failed to reach grasp pose")
        return False
    time.sleep(0.3)

    # 3. Gentle grasp (limit closure to avoid squeezing soft fruit)
    grasp_width_fraction = max(0.05, min(0.90, grasp_width_fraction))
    if gripper_enabled and gripper is not None:
        gripper.set_width(grasp_width_fraction, speed=grasp_speed)
    time.sleep(0.8)
    grasped = True

    # 4. Lift
    lift_xyz = grasp_xyz.copy()
    lift_xyz[2] += 0.15
    arm.move_cartesian(lift_xyz)
    time.sleep(0.5)

    # 5. Move to drop location
    pre_drop = drop_xyz.copy()
    pre_drop[2] += 0.08
    if not arm.move_cartesian(pre_drop):
        logger.warning("Failed to reach pre-drop pose")
    time.sleep(0.3)

    if not arm.move_cartesian(drop_xyz):
        logger.warning("Failed to reach drop pose")
    time.sleep(0.3)

    # 6. Release orange
    if gripper_enabled and gripper is not None:
        gripper.open(speed=120, blocking=True)
    time.sleep(0.4)

    # 7. Retreat and return observe pose
    arm.move_cartesian(pre_drop)
    time.sleep(0.2)
    arm.go_to_observe(blocking=True)

    logger.info("Pick-and-place %s!", "SUCCESS" if grasped else "UNCERTAIN")
    return grasped


# ── Policy-based inference ────────────────────────────────────────────────────

class PolicyRunner:
    """
    Wraps a trained policy and handles observation building + action execution.
    """

    def __init__(self, policy_type: str, ckpt_path: str, device: str = "cuda"):
        self.policy_type = policy_type
        self.device      = device
        self._obs_buffer_wrist = []
        self._obs_buffer_stand = []
        self._obs_buffer_prop  = []

        if policy_type == "diffusion":
            from policy.diffusion_policy import DiffusionPolicy
            self.policy = DiffusionPolicy(
                action_dim=7, obs_horizon=2, pred_horizon=16,
                action_horizon=8, diffusion_steps=100, device=device,
            )
            self.obs_horizon    = 2
            self.action_horizon = 8
        else:
            from policy.act_policy import ACTPolicy
            self.policy = ACTPolicy(
                action_dim=7, chunk_size=50, d_model=512, device=device,
            )
            self.obs_horizon    = 1
            self.action_horizon = 1

        self.policy.load(ckpt_path)
        self.policy.eval()
        logger.info("Loaded %s policy from %s", policy_type, ckpt_path)

    def reset(self) -> None:
        self._obs_buffer_wrist.clear()
        self._obs_buffer_stand.clear()
        self._obs_buffer_prop.clear()
        if self.policy_type == "act":
            self.policy.reset_ensemble()

    def step(
        self,
        frames: dict,
        proprio_np: np.ndarray,    # (action_dim,)
    ) -> np.ndarray:
        """
        Update observation buffer and return next action (action_dim,).
        """
        # Convert frames → (3, H, W) float tensors
        def preproc(bgr):
            if bgr is None:
                return np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_W, IMG_H))
            return rgb.astype(np.float32).transpose(2, 0, 1) / 255.0

        self._obs_buffer_wrist.append(preproc(frames["wrist"]))
        self._obs_buffer_stand.append(preproc(frames["stand"]))
        self._obs_buffer_prop.append(proprio_np)

        # Keep only obs_horizon most recent
        if len(self._obs_buffer_wrist) > self.obs_horizon:
            self._obs_buffer_wrist.pop(0)
            self._obs_buffer_stand.pop(0)
            self._obs_buffer_prop.pop(0)

        # Pad if buffer not yet full
        while len(self._obs_buffer_wrist) < self.obs_horizon:
            self._obs_buffer_wrist.insert(0, self._obs_buffer_wrist[0])
            self._obs_buffer_stand.insert(0, self._obs_buffer_stand[0])
            self._obs_buffer_prop.insert(0, self._obs_buffer_prop[0])

        wrist_t = torch.from_numpy(
            np.stack(self._obs_buffer_wrist)
        ).unsqueeze(0).to(self.device)   # (1, obs_h, 3, H, W)
        stand_t = torch.from_numpy(
            np.stack(self._obs_buffer_stand)
        ).unsqueeze(0).to(self.device)
        prop_t  = torch.from_numpy(
            np.stack(self._obs_buffer_prop)
        ).unsqueeze(0).to(self.device)   # (1, obs_h, 7)

        if self.policy_type == "diffusion":
            actions = self.policy.predict_action(wrist_t, stand_t, prop_t)
            return actions[0]   # execute first action of chunk
        else:
            # ACT: squeeze obs_horizon dim
            return self.policy.predict_action(
                wrist_t.squeeze(1), stand_t.squeeze(1), prop_t.squeeze(1)
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_policy_loop(runner: PolicyRunner, arm: ArmController,
                    gripper: Optional[GripperController], cameras: CameraManager,
                    gripper_enabled: bool = False,
                    max_steps: int = 200) -> None:
    """Execute the policy loop for one pick episode."""
    runner.reset()
    arm.go_to_observe(blocking=True)
    if gripper_enabled and gripper is not None:
        gripper.open(blocking=True)

    logger.info("Policy execution started. Press ESC to stop.")

    for step in range(max_steps):
        t_start = time.monotonic()

        frames  = cameras.get_raw_frames()
        joints  = np.array(arm.get_joint_angles_deg(), dtype=np.float32)
        gripper_w = np.float32(gripper.get_width_fraction()) if gripper is not None else np.float32(0.0)
        # Keep policy interface stable at 6 arm joints + 1 gripper.
        if joints.shape[0] < 6:
            joints = np.pad(joints, (0, 6 - joints.shape[0]), mode="constant")
        elif joints.shape[0] > 6:
            joints = joints[:6]
        proprio = np.concatenate([joints, [gripper_w]])

        action = runner.step(frames, proprio)   # (7,)

        # Apply action
        target_joints  = action[:arm.dof].tolist()
        target_gripper = float(action[6])
        arm.set_joint_angles_deg(target_joints, speed=400)
        if gripper_enabled and gripper is not None:
            gripper.set_width(target_gripper, speed=200)

        # Display
        stand_bgr = frames["stand"]
        wrist_bgr = frames["wrist"]
        if stand_bgr is not None and wrist_bgr is not None:
            h_disp = np.hstack([
                cv2.resize(wrist_bgr, (320, 240)),
                cv2.resize(stand_bgr, (320, 240)),
            ])
            cv2.putText(h_disp, f"Step {step+1}/{max_steps}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Policy Execution", h_disp)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

        # Target ~10 Hz
        elapsed = time.monotonic() - t_start
        time.sleep(max(0, 0.1 - elapsed))

    cv2.destroyAllWindows()
    logger.info("Episode finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["scripted", "policy"], default="scripted")
    parser.add_argument("--policy",   choices=["diffusion", "act"],   default="diffusion")
    parser.add_argument("--ckpt",     type=str, default="checkpoints/diffusion_best.pth")
    parser.add_argument("--arm_cfg",  type=str, default="config/arm_config.yaml")
    parser.add_argument("--cam_cfg",  type=str, default="config/camera_config.yaml")
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--attempts", type=int, default=10,
                        help="Number of scripted pick-and-place cycles")
    parser.add_argument("--drop_x", type=float, default=0.22,
                        help="Drop location X in base frame (m)")
    parser.add_argument("--drop_y", type=float, default=-0.18,
                        help="Drop location Y in base frame (m)")
    parser.add_argument("--drop_z", type=float, default=0.08,
                        help="Drop location Z in base frame (m)")
    parser.add_argument("--grasp_width", type=float, default=0.35,
                        help="Gripper width fraction at grasp: 0=fully closed, 1=fully open")
    parser.add_argument("--grasp_speed", type=int, default=70,
                        help="Gripper closing speed for grasp")
    parser.add_argument("--pick_z", type=float, default=0.05,
                        help="Fixed pick height (m) used with vision XY")
    parser.add_argument("--enable_gripper", action="store_true",
                        help="Enable gripper motion commands (disabled by default for safety)")
    args = parser.parse_args()

    arm     = ArmController(args.arm_cfg)
    gripper = GripperController(args.arm_cfg) if args.enable_gripper else None
    cameras = CameraManager(args.cam_cfg)
    detector = OrangeDetector(args.cam_cfg)

    arm.connect()
    if args.enable_gripper and gripper is not None:
        gripper.connect()
    cameras.start()

    if not args.enable_gripper:
        logger.info("Gripper motion disabled. Gripper will not be connected or commanded.")
    else:
        logger.info("Gripper motion enabled. Gripper is an ACTUATOR in scripted/policy execution.")

    try:
        if args.mode == "scripted":
            drop_xyz = np.array([args.drop_x, args.drop_y, args.drop_z], dtype=np.float64)
            logger.info("Running scripted pick-and-place for %d attempts", args.attempts)
            logger.info("Drop location XYZ=%.3f, %.3f, %.3f", *drop_xyz)
            for attempt in range(args.attempts):
                success = scripted_pick(
                    arm, gripper, cameras, detector, drop_xyz,
                    gripper_enabled=args.enable_gripper,
                    pick_z=args.pick_z,
                    grasp_width_fraction=args.grasp_width,
                    grasp_speed=args.grasp_speed,
                )
                logger.info("Attempt %d/%d: %s", attempt + 1, args.attempts,
                            "SUCCESS" if success else "FAILED")
                arm.go_to_observe(blocking=True)
                if success and args.enable_gripper and gripper is not None:
                    gripper.open(blocking=True)
                time.sleep(1.0)

        else:
            device = args.device if torch.cuda.is_available() else "cpu"
            runner = PolicyRunner(args.policy, args.ckpt, device)
            run_policy_loop(runner, arm, gripper, cameras,
                            gripper_enabled=args.enable_gripper,
                            max_steps=args.max_steps)

    finally:
        cameras.stop()
        arm.go_home(blocking=True)
        arm.disconnect()
        if args.enable_gripper and gripper is not None:
            gripper.disconnect(park_open=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
