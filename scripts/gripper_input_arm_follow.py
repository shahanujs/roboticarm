"""
Strict gripper-input arm-follow mode (single output ID).

Design goal:
- Gripper is INPUT only (manual hand movement).
- Script NEVER commands gripper motor position.
- Script torques and commands only one selected arm ID (default: ID 2).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.feetech_bus import FeetechBus


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def deg_to_ticks(deg: float, joint_cfg: dict) -> int:
    dmin = float(joint_cfg["deg_min"])
    dmax = float(joint_cfg["deg_max"])
    deg = max(dmin, min(dmax, deg))
    min_pos = int(joint_cfg["min_pos"])
    max_pos = int(joint_cfg["max_pos"])
    home_pos = int(joint_cfg["home_pos"])
    ticks_per_deg = (max_pos - min_pos) / (dmax - dmin)
    ticks = int(round(home_pos + deg * ticks_per_deg))
    return max(20, min(4075, ticks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_cfg", type=str, default="config/arm_config.yaml")
    ap.add_argument("--rate_hz", type=float, default=10.0)
    ap.add_argument("--deadband", type=float, default=0.02)
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--output_id", type=int, default=2,
                    help="Single arm servo ID to command from gripper input")
    ap.add_argument("--angle_closed", type=float, default=-65.0,
                    help="Target angle for output_id when gripper is closed")
    ap.add_argument("--angle_open", type=float, default=-10.0,
                    help="Target angle for output_id when gripper is open")
    args = ap.parse_args()

    with open(args.arm_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    arm_cfg = cfg["arm"]
    arm_port = arm_cfg["port"]
    arm_baud = int(arm_cfg["baudrate"])
    gripper_id = int(cfg["gripper"]["id"])
    open_pos = int(cfg["gripper"]["open_pos"])
    close_pos = int(cfg["gripper"]["close_pos"])

    joint_map = {int(j["id"]): j for j in arm_cfg["joints"]}
    if args.output_id not in joint_map:
        raise ValueError(f"output_id {args.output_id} not in arm joints config")
    out_joint = joint_map[args.output_id]

    bus = FeetechBus(arm_port, arm_baud)

    try:
        # Input-only gripper: ensure torque OFF on gripper ID.
        bus.set_torque(gripper_id, False)

        # Safety: torque off all arm IDs first, then enable only selected output ID.
        arm_ids = [int(j["id"]) for j in arm_cfg["joints"]]
        for sid in arm_ids:
            bus.set_torque(sid, False)
        bus.set_torque(args.output_id, True)

        print("Gripper INPUT -> Arm FOLLOW mode started.")
        print("No gripper motor command is sent in this script.")
        print(f"Only output_id={args.output_id} is torqued/commanded.")
        print("Press Ctrl+C to stop.")

        dt = 1.0 / max(args.rate_hz, 1.0)

        def read_w() -> float:
            ticks = bus.get_present_position_unsigned(gripper_id)
            if ticks is None:
                return 0.0
            span = float(open_pos - close_pos)
            if abs(span) < 1e-6:
                return 0.0
            return clamp01((ticks - close_pos) / span)

        last_w = read_w()
        print(f"initial gripper_w={last_w:.3f}")
        last_print = time.monotonic()

        while True:
            w = read_w()
            if abs(w - last_w) >= args.deadband:
                target_deg = lerp(args.angle_closed, args.angle_open, w)
                target_ticks = deg_to_ticks(target_deg, out_joint)
                bus.set_goal_position(args.output_id, target_ticks, args.speed)
                print(f"cmd_ids=[{args.output_id}] gripper_w={w:.2f} -> target_deg={target_deg:.1f}")
                last_w = w

            now = time.monotonic()
            if now - last_print > 1.0:
                print(f"live gripper_w={w:.3f}")
                last_print = now

            time.sleep(dt)

    except KeyboardInterrupt:
        print("Stopping input-follow mode.")
    finally:
        try:
            bus.set_torque(args.output_id, False)
        except Exception:
            pass
        try:
            bus.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
