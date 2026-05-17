#!/usr/bin/env python3
"""
Direct test: command ID2 to move and check if shoulder physically moves.
"""

import sys
import time

sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

def deg_to_ticks(deg, home_pos=2048, ticks_per_deg=22.75):
    """Convert shoulder angle (deg) to servo position (ticks)."""
    return int(round(home_pos + deg * ticks_per_deg))

bus = FeetechBus(port="/dev/ttyACM0")

try:
    print("Pinging ID2...")
    if not bus.ping(2):
        print("ERROR: ID2 not responding!")
        sys.exit(1)
    
    print("Enabling torque on ID2...")
    bus.set_torque(2, True)
    time.sleep(0.5)
    
    print("Reading current position of ID2...")
    current_pos = bus.get_present_position_unsigned(2)
    print(f"ID2 current position: {current_pos} ticks")
    
    # Move to -70 degrees
    target_angle = -70
    target_ticks = deg_to_ticks(target_angle)
    print(f"\n→ Moving ID2 to {target_angle}° ({target_ticks} ticks)...")
    bus.set_goal_position(2, target_ticks, speed=60)
    
    time.sleep(2)
    new_pos = bus.get_present_position_unsigned(2)
    print(f"ID2 new position: {new_pos} ticks (delta: {new_pos - current_pos})")
    
    # Move to -8 degrees
    target_angle = -8
    target_ticks = deg_to_ticks(target_angle)
    print(f"\n→ Moving ID2 to {target_angle}° ({target_ticks} ticks)...")
    bus.set_goal_position(2, target_ticks, speed=60)
    
    time.sleep(2)
    new_pos = bus.get_present_position_unsigned(2)
    print(f"ID2 new position: {new_pos} ticks (delta: {new_pos - current_pos})")
    
    print("\n✓ Test complete. Did you see the shoulder move?")
    
finally:
    bus.set_torque(2, False)
    bus.close()
