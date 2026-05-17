"""
No-motion servo ID diagnostic.

Use this to find which ID corresponds to the physical gripper handle movement.
It only reads servo positions; it never sends motion commands.

Procedure:
1) Keep arm still.
2) Move only gripper handle by hand during capture.
3) Script reports which ID changed the most.

Usage:
  python scripts/diagnose_gripper_manual.py
  python scripts/diagnose_gripper_manual.py --port /dev/ttyACM1 --ids 1 2 3 4 5 6 7 8 --seconds 12
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.feetech_bus import FeetechBus


def summarize(samples: List[int]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "span": 0.0, "stdev": 0.0}
    return {
        "count": float(len(samples)),
        "span": float(max(samples) - min(samples)),
        "stdev": float(statistics.pstdev(samples) if len(samples) > 1 else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/arm_config.yaml")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=None)
    ap.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--rate_hz", type=float, default=20.0)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    port = args.port or cfg["arm"]["port"]
    baud = args.baud or int(cfg["arm"]["baudrate"])

    print(f"Reading bus: {port} @ {baud}")
    print("Keep ARM still. Move only gripper handle now.")
    print(f"Capturing for {args.seconds:.1f}s at {args.rate_hz:.1f} Hz...")

    bus = FeetechBus(port, baud)
    dt = 1.0 / max(args.rate_hz, 1.0)

    data: Dict[int, List[int]] = {sid: [] for sid in args.ids}
    t_end = time.monotonic() + args.seconds

    try:
        while time.monotonic() < t_end:
            for sid in args.ids:
                pos = bus.get_present_position_unsigned(sid)
                if pos is not None:
                    data[sid].append(int(pos))
            time.sleep(dt)
    finally:
        bus.close()

    stats = []
    for sid in args.ids:
        s = summarize(data[sid])
        stats.append((sid, s["span"], s["stdev"], int(s["count"])))

    stats.sort(key=lambda x: (x[1], x[2]), reverse=True)

    print("\nResults (ranked by movement span):")
    for sid, span, stdev, count in stats:
        print(f"  ID {sid}: span={span:.1f} ticks, stdev={stdev:.1f}, samples={count}")

    best = stats[0]
    min_span_for_detection = 5.0
    if best[1] < min_span_for_detection:
        print("\nNo moving ID detected (all spans near zero).")
        print("Passive manual movement is not observable on this setup.")
        print("Use active probe instead: python scripts/identify_gripper_id.py --port /dev/ttyACM1 --ids 1 2 3 4 5 6 --delta 20 --speed 80")
    else:
        print("\nLikely gripper ID:", best[0])
        print("If this is not expected, repeat while moving only gripper and keeping arm fully still.")


if __name__ == "__main__":
    main()
