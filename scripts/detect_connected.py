#!/usr/bin/env python3
"""
Detect which servo IDs respond now that gripper is disconnected.
"""

import sys
sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

bus = FeetechBus(port="/dev/ttyACM0")

try:
    print("Scanning for connected servos...\n")
    
    connected = []
    for servo_id in range(1, 7):
        if bus.ping(servo_id):
            connected.append(servo_id)
            print(f"✓ ID{servo_id} responds")
        else:
            print(f"✗ ID{servo_id} no response")
    
    print(f"\n{'='*50}")
    print(f"Connected servos: {connected}")
    
    if 6 in connected:
        print(f"\n✓ ID6 (arm claw) is connected and responding.")
        print(f"Ready to test ID6 open/close 5 times.")
    else:
        print(f"\n✗ ID6 is NOT connected.")
    
finally:
    bus.close()
