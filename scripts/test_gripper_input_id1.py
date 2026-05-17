#!/usr/bin/env python3
"""
Test: gripper (ID6) as INPUT ONLY (no torque), map to ID1 (arm).
User manually moves gripper. ID1 should follow.
"""

import sys
import time
sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

def deg_to_ticks(deg, home_pos=2048, ticks_per_deg=22.75):
    return int(round(home_pos + deg * ticks_per_deg))

bus = FeetechBus(port="/dev/ttyACM0")

try:
    gripper_id = 6
    arm_id = 1
    
    print(f"Setup: Gripper (ID{gripper_id}) = INPUT ONLY, Arm servo (ID{arm_id}) = OUTPUT")
    print()
    
    # Gripper: disable torque (read-only)
    bus.set_torque(gripper_id, False)
    print(f"✓ ID{gripper_id} torque OFF (gripper is input-only)")
    
    # Arm: enable torque
    bus.set_torque(arm_id, True)
    print(f"✓ ID{arm_id} torque ON (will respond to commands)")
    
    time.sleep(1)
    
    print(f"\nReady. Manually move the gripper finger.")
    print(f"ID{arm_id} will follow. Running for 20 seconds...\n")
    
    start_time = time.time()
    while time.time() - start_time < 20:
        # Read gripper position (input only)
        gripper_pos = bus.get_present_position(gripper_id)
        if gripper_pos is None:
            print(f"Failed to read gripper ID{gripper_id}")
            continue
        
        # Map gripper position to arm angle
        # gripper_pos: [1600, 2400] → arm_angle: [-70, -8]
        gripper_range = 2400 - 1600
        gripper_frac = (gripper_pos - 1600) / gripper_range
        gripper_frac = max(0.0, min(1.0, gripper_frac))
        
        arm_angle = -70 + gripper_frac * (-8 - (-70))  # -70 to -8
        arm_ticks = deg_to_ticks(arm_angle)
        
        # Command arm servo
        bus.set_goal_position(arm_id, arm_ticks, speed=60)
        
        print(f"Gripper pos: {gripper_pos:4d} (frac={gripper_frac:.2f}) → Arm ID{arm_id}: {arm_angle:6.1f}° ({arm_ticks:4d} ticks)")
        
        time.sleep(0.2)
    
    print(f"\n✓ Test complete.")
    print(f"Did you see the shoulder move when you moved the gripper?")
    
finally:
    bus.set_torque(1, False)
    bus.set_torque(6, False)
    bus.close()
