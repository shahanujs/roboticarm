"""
Safe arm-only motion demo.

This script never imports or commands the gripper.
It only commands arm joints from config/arm_config.yaml.

Usage:
  python scripts/arm_safe_demo.py
  python scripts/arm_safe_demo.py --cycles 3 --speed 120
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.arm_controller import ArmController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_cfg", default="config/arm_config.yaml")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--speed", type=int, default=120)
    args = ap.parse_args()

    arm = ArmController(args.arm_cfg)
    arm.connect()

    try:
        print("Arm-only safety demo started (no gripper commands).")
        for i in range(args.cycles):
            print(f"Cycle {i+1}/{args.cycles}: observe")
            arm.go_to_observe(speed=args.speed, blocking=True)
            time.sleep(0.5)
            print(f"Cycle {i+1}/{args.cycles}: home")
            arm.go_home(speed=args.speed, blocking=True)
            time.sleep(0.5)
        print("Arm-only safety demo complete.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
