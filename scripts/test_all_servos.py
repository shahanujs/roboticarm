#!/usr/bin/env python3
"""
Test each servo ID (1-6) to see which ones respond to move commands.
"""

import sys
import time

sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

def deg_to_ticks(deg, home_pos=2048, ticks_per_deg=22.75):
    return int(round(home_pos + deg * ticks_per_deg))

bus = FeetechBus(port="/dev/ttyACM0")

try:
    for servo_id in range(1, 7):
        print(f"\n{'='*50}")
        print(f"Testing ID{servo_id}...")
        
        # Ping
        if not bus.ping(servo_id):
            print(f"  ✗ Does not respond to ping")
            continue
        print(f"  ✓ Ping OK")
        
        # Enable torque
        bus.set_torque(servo_id, True)
        print(f"  ✓ Torque enabled")
        
        time.sleep(0.3)
        
        # Read current position
        current_pos = bus.get_present_position_unsigned(servo_id)
        print(f"  Current pos: {current_pos} ticks")
        
        # Send move command
        target_ticks = deg_to_ticks(-50)
        print(f"  → Sending command to move to {target_ticks} ticks...")
        bus.set_goal_position(servo_id, target_ticks, speed=60)
        
        time.sleep(1)
        
        # Read new position
        new_pos = bus.get_present_position_unsigned(servo_id)
        delta = new_pos - current_pos
        
        print(f"  New pos: {new_pos} ticks")
        print(f"  Delta: {delta} ticks")
        
        if abs(delta) > 10:
            print(f"  ✓ MOVED!")
        else:
            print(f"  ✗ Did not move (or moved <10 ticks)")

finally:
    print(f"\n{'='*50}")
    print("Test complete.")
    bus.close()
