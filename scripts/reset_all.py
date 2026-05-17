#!/usr/bin/env python3
"""
Reset: Disable ALL torque (arm and gripper).
Gripper goes back to unpowered state.
"""

import sys
sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

bus = FeetechBus(port="/dev/ttyACM0")

try:
    print("Disabling torque on all servos...")
    for servo_id in range(1, 7):
        bus.set_torque(servo_id, False)
        print(f"  ID{servo_id}: torque OFF")
    
    print("\n✓ All servos unpowered. Gripper is now in input-only state.")
    
finally:
    bus.close()
