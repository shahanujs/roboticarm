#!/usr/bin/env python3
"""
Check ID2 servo status: torque, errors, voltage, etc.
"""

import sys
sys.path.insert(0, "/home/anujnvidia/Documents/roboarm")

from hardware.feetech_bus import FeetechBus

bus = FeetechBus(port="/dev/ttyACM0")

try:
    print("Checking ID2 status...\n")
    
    # Ping
    ping = bus.ping(2)
    print(f"Ping ID2: {ping}")
    
    # Torque
    torque = bus.get_torque(2)
    print(f"Torque enabled: {torque}")
    
    # Position
    pos = bus.get_present_position_unsigned(2)
    print(f"Current position: {pos} ticks")
    
    # Voltage
    try:
        voltage = bus.read_register(2, 62, 2)  # Register 62, 2 bytes
        if voltage:
            v = voltage[0] | (voltage[1] << 8)
            print(f"Voltage: {v/1000:.1f}V")
    except:
        print("Could not read voltage")
    
    # Error code (register 63)
    try:
        error = bus.read_register(2, 63, 1)
        if error:
            print(f"Error byte: {bin(error[0])} (0x{error[0]:02x})")
            if error[0] & 0x01:
                print("  - Overload")
            if error[0] & 0x02:
                print("  - Electrical shock")
            if error[0] & 0x04:
                print("  - Motor stalled")
            if error[0] & 0x08:
                print("  - Over-temperature")
            if error[0] & 0x10:
                print("  - Inconsistent command")
            if error[0] & 0x20:
                print("  - Invalid instruction")
    except:
        print("Could not read error byte")
    
finally:
    bus.close()
