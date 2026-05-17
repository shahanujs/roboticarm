#!/usr/bin/env python3
"""
Ultra-simple test: manually move gripper (ID6), shoulder (ID2) follows.
No multi-joint complexity. Just raw servo mapping.
"""

import sys
import time
import argparse
import logging

sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test: gripper (ID6) input → shoulder (ID2) output")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--gripper_id", type=int, default=6, help="Gripper servo ID")
    parser.add_argument("--shoulder_id", type=int, default=2, help="Shoulder servo ID")
    parser.add_argument("--shoulder_closed", type=float, default=-70, help="Shoulder angle when gripper closed (deg)")
    parser.add_argument("--shoulder_open", type=float, default=-8, help="Shoulder angle when gripper open (deg)")
    parser.add_argument("--gripper_closed_pos", type=int, default=1600, help="Gripper servo pos when fully closed")
    parser.add_argument("--gripper_open_pos", type=int, default=2400, help="Gripper servo pos when fully open")
    parser.add_argument("--rate_hz", type=int, default=10, help="Update rate (Hz)")
    parser.add_argument("--duration_sec", type=int, default=30, help="Run duration (seconds)")
    args = parser.parse_args()

    # Shoulder (ID2) servo config from arm_config.yaml
    shoulder_home_pos = 2048
    shoulder_deg_min = -90.0
    shoulder_deg_max = 90.0
    shoulder_ticks_per_deg = (4095 - 0) / (shoulder_deg_max - shoulder_deg_min)  # ≈ 22.75 ticks/deg
    
    def deg_to_ticks(deg):
        """Convert shoulder angle (deg) to servo position (ticks)."""
        deg = max(shoulder_deg_min, min(shoulder_deg_max, deg))
        return int(round(shoulder_home_pos + deg * shoulder_ticks_per_deg))

    bus = FeetechBus(port=args.port)
    
    try:
        # Bus auto-opens in __init__, so just ping both servos
        logger.info("Bus connected.")
        
        logger.info(f"Pinging gripper (ID {args.gripper_id}) and shoulder (ID {args.shoulder_id})...")
        gripper_ok = bus.ping(args.gripper_id)
        shoulder_ok = bus.ping(args.shoulder_id)
        logger.info(f"Gripper ID{args.gripper_id}: {gripper_ok}, Shoulder ID{args.shoulder_id}: {shoulder_ok}")
        
        if not (gripper_ok and shoulder_ok):
            logger.error("One or both servos not responding. Abort.")
            return
        
        # Disable gripper torque so user can move it manually
        logger.info(f"Disabling gripper (ID {args.gripper_id}) torque for manual input...")
        bus.set_torque(args.gripper_id, False)
        
        # Enable shoulder torque
        logger.info(f"Enabling shoulder (ID {args.shoulder_id}) torque...")
        bus.set_torque(args.shoulder_id, True)
        
        # Give user time to prepare
        logger.info("Ready. Manually move the gripper. Shoulder will follow.")
        logger.info(f"Running for {args.duration_sec} seconds...")
        time.sleep(2)
        
        start_time = time.time()
        update_interval = 1.0 / args.rate_hz
        last_update = start_time
        
        while time.time() - start_time < args.duration_sec:
            now = time.time()
            if now - last_update < update_interval:
                time.sleep(0.01)
                continue
            
            # Read gripper current position
            gripper_pos = bus.get_present_position(args.gripper_id)
            if gripper_pos is None:
                logger.warning(f"Failed to read gripper ID{args.gripper_id} position")
                last_update = now
                continue
            
            # Map gripper position to shoulder angle
            # gripper_pos: [gripper_closed_pos, gripper_open_pos] → shoulder angle [shoulder_closed, shoulder_open]
            gripper_range = args.gripper_open_pos - args.gripper_closed_pos
            gripper_frac = (gripper_pos - args.gripper_closed_pos) / gripper_range if gripper_range > 0 else 0.0
            gripper_frac = max(0.0, min(1.0, gripper_frac))  # Clamp [0, 1]
            
            shoulder_angle = args.shoulder_closed + gripper_frac * (args.shoulder_open - args.shoulder_closed)
            
            # Convert shoulder angle to servo position (ticks)
            shoulder_ticks = deg_to_ticks(shoulder_angle)
            
            # Send shoulder command
            logger.info(
                f"Gripper ID{args.gripper_id}: pos={gripper_pos:4d} (frac={gripper_frac:.2f}) → "
                f"Shoulder ID{args.shoulder_id}: angle={shoulder_angle:6.1f}° (ticks={shoulder_ticks:4d})"
            )
            bus.set_goal_position(args.shoulder_id, shoulder_ticks, speed=60)
            
            last_update = now
        
        logger.info("Test duration complete.")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            logger.info("Disabling all torques...")
            bus.set_torque(args.gripper_id, False)
            bus.set_torque(args.shoulder_id, False)
            bus.close()
            logger.info("Bus closed.")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    main()
