#!/usr/bin/env python3
"""
Leader-follower teleoperation for Hiwonder SO-ARM101.

Leader arm  (ACM0) : torque OFF — user moves freely
Follower arm (ACM1): torque ON  — mirrors leader positions

Joint mapping (both arms, ID 1-6):
  ID1 = Base rotation
  ID2 = Shoulder
  ID3 = Elbow
  ID4 = Wrist pitch
  ID5 = Wrist roll  (~360°)
  ID6 = Gripper

Usage:
  python scripts/teleop.py
  python scripts/teleop.py --rate_hz 20 --deadband 15
  python scripts/teleop.py --leader /dev/ttyACM0 --follower /dev/ttyACM1
"""

from __future__ import annotations

import argparse
import signal
import struct
import sys
import time

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

from hardware.feetech_bus import FeetechBus

# ── Per-joint safety limits (ticks, 0–4095) ─────────────────────────────────
# Clip follower commands to these ranges to protect hardware.
JOINT_LIMITS = {
    1: (600,  3400),   # Base rotation
    2: (200,  3900),   # Shoulder  (near-continuous, but arm has stops)
    3: (0,    2100),   # Elbow
    4: (100,  1900),   # Wrist pitch
    5: (0,    4095),   # Wrist roll (~360° — no hard limit)
    6: (1550, 2950),   # Gripper
}

SERVO_IDS = list(range(1, 7))


def clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def read_all(bus: FeetechBus) -> dict[int, int] | None:
    """Read positions for all 6 servos. Returns None if any read fails."""
    positions = {}
    for sid in SERVO_IDS:
        pos = bus.get_present_position_unsigned(sid)
        if pos is None:
            return None
        positions[sid] = pos
    return positions


def capture_initial_pose(bus: FeetechBus, samples: int = 7, interval_s: float = 0.02) -> dict[int, int] | None:
    """Capture a stable live pose from leader using per-joint median across samples."""
    collected: list[dict[int, int]] = []
    for _ in range(samples):
        pos = read_all(bus)
        if pos is not None:
            collected.append(pos)
        time.sleep(interval_s)

    if not collected:
        return None

    stable = {}
    for sid in SERVO_IDS:
        vals = sorted(sample[sid] for sample in collected)
        stable[sid] = vals[len(vals) // 2]
    return stable


def sync_write_positions(bus: FeetechBus, positions: dict[int, int], speed: int) -> None:
    """Send all 6 goal positions in a single sync_write packet."""
    pairs = []
    for sid in SERVO_IDS:
        pos = positions[sid]
        lo, hi = JOINT_LIMITS[sid]
        pos = clip(pos, lo, hi)
        data = struct.pack("<HH", pos & 0x0FFF, speed & 0x3FF)
        pairs.append((sid, data))
    bus.sync_write(address=0x2A, data_len=4, id_data_pairs=pairs)


def disable_all_torques(bus: FeetechBus, label: str) -> None:
    for sid in SERVO_IDS:
        bus.set_torque(sid, False)
    print(f"\n[teleop] All torques disabled on {label}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leader-follower teleoperation")
    parser.add_argument("--leader",    default="/dev/ttyACM0", help="Leader serial port")
    parser.add_argument("--follower",  default="/dev/ttyACM1", help="Follower serial port")
    parser.add_argument("--baud",      type=int, default=1_000_000)
    parser.add_argument("--rate_hz",   type=float, default=12.0,
                        help="Control loop rate (Hz)")
    parser.add_argument("--deadband",  type=int, default=0,
                        help="Minimum tick change to send new command (reduces jitter)")
    parser.add_argument("--speed",     type=int, default=50,
                        help="Follower move speed (0=max, 1-1023)")
    args = parser.parse_args()

    period = 1.0 / args.rate_hz

    print("[teleop] Opening serial ports...")
    leader_bus   = FeetechBus(port=args.leader,   baudrate=args.baud)
    follower_bus = FeetechBus(port=args.follower, baudrate=args.baud)

    # Graceful shutdown on Ctrl+C or SIGTERM
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[teleop] Disabling leader torques...")
    for sid in SERVO_IDS:
        leader_bus.set_torque(sid, False)

    print("[teleop] Enabling follower torques...")
    for sid in SERVO_IDS:
        follower_bus.set_torque(sid, True)

    # Read initial leader positions and send to follower as starting point
    print("[teleop] Reading initial leader positions...")
    init_pos = None
    for _ in range(10):
        init_pos = capture_initial_pose(leader_bus)
        if init_pos:
            break
        time.sleep(0.05)

    if init_pos is None:
        print("[teleop] ERROR: Could not read leader positions. Check /dev/ttyACM0 permissions.")
        disable_all_torques(follower_bus, "follower")
        leader_bus.close()
        follower_bus.close()
        sys.exit(1)

    last_sent = dict(init_pos)
    sync_write_positions(follower_bus, init_pos, speed=args.speed)
    print(
        "[teleop] Startup pose captured from leader: "
        + " ".join(f"ID{i}:{init_pos[i]}" for i in SERVO_IDS)
    )
    print(f"[teleop] Gripper default set from leader ID6: {init_pos[6]}")
    time.sleep(0.5)   # let follower reach start position

    print(f"[teleop] Running at {args.rate_hz} Hz | deadband={args.deadband} ticks | speed={args.speed}")
    print("[teleop] Press Ctrl+C to stop.\n")

    loop_count = 0
    err_count  = 0
    t_start    = time.monotonic()

    while running:
        t0 = time.monotonic()

        positions = read_all(leader_bus)
        if positions is None:
            err_count += 1
            if err_count % 20 == 0:
                print(f"[teleop] WARNING: {err_count} read failures so far")
            time.sleep(period)
            continue

        err_count = 0

        # Exact mirror by default (deadband=0). Deadband remains optional for jitter filtering.
        if args.deadband <= 0:
            sync_write_positions(follower_bus, positions, speed=args.speed)
            last_sent = dict(positions)
        else:
            to_send = dict(last_sent)
            moved = False
            for sid in SERVO_IDS:
                if abs(positions[sid] - last_sent[sid]) > args.deadband:
                    to_send[sid] = positions[sid]
                    moved = True

            if moved:
                sync_write_positions(follower_bus, to_send, speed=args.speed)
                last_sent = to_send

        loop_count += 1
        if loop_count % 120 == 0:
            elapsed = time.monotonic() - t_start
            actual_hz = loop_count / elapsed
            pos_str = " ".join(f"ID{i}:{positions[i]:4d}" for i in SERVO_IDS)
            print(f"[teleop] {actual_hz:.1f} Hz | {pos_str}")

        # Pace the loop
        elapsed = time.monotonic() - t0
        sleep_t = period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    disable_all_torques(follower_bus, "follower")
    leader_bus.close()
    follower_bus.close()
    print("[teleop] Done.")


if __name__ == "__main__":
    main()
