"""
Identify which servo ID corresponds to the gripper by moving each ID slightly.

Safety:
- Small motion only (+/- 20 ticks)
- Slow speed
- Automatically returns to original position

Usage:
    python scripts/identify_gripper_id.py --port /dev/ttyACM0 --baud 1000000 --ids 1 2 3 4 5 6

Watch the robot and note which step visibly opens/closes the gripper.
"""

import argparse
import time
import sys

sys.path.insert(0, ".")
from hardware.feetech_bus import FeetechBus


def nudge_servo(bus: FeetechBus, sid: int, delta: int, speed: int) -> bool:
    pos = bus.get_present_position_unsigned(sid)
    if pos is None:
        print(f"ID {sid}: cannot read position")
        return False

    lo, hi = 20, 4075
    p1 = max(lo, min(hi, pos + delta))
    p2 = max(lo, min(hi, pos - delta))

    print(f"ID {sid}: pos={pos} -> {p1} -> {p2} -> {pos}")
    bus.set_goal_position(sid, p1, speed)
    time.sleep(0.5)
    bus.set_goal_position(sid, p2, speed)
    time.sleep(0.5)
    bus.set_goal_position(sid, pos, speed)
    time.sleep(0.6)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--delta", type=int, default=20)
    ap.add_argument("--speed", type=int, default=80)
    args = ap.parse_args()

    print("Connecting to", args.port, "@", args.baud)
    bus = FeetechBus(args.port, args.baud, timeout=0.02)

    try:
        print("\nStarting identification sweep.")
        print("Keep clear of the arm. Press Ctrl+C to stop.\n")

        for sid in args.ids:
            if bus.ping(sid):
                print("\n--- Testing ID", sid, "---")
                nudge_servo(bus, sid, args.delta, args.speed)
            else:
                print("ID", sid, "did not respond")

        print("\nDone. Note the ID that moved the gripper.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
