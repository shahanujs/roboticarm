"""
Feetech STS/SCS half-duplex UART protocol implementation.
Compatible with the servos used in Hiwonder SO-ARM101.

Protocol summary
----------------
TX packet  : 0xFF 0xFF  ID  LEN  INST  [params...]  CHECKSUM
RX packet  : 0xFF 0xFF  ID  LEN  ERR   [data...]    CHECKSUM
CHECKSUM   : (~(ID + LEN + INST_or_ERR + sum(params_or_data))) & 0xFF
"""

from __future__ import annotations

import struct
import time
import logging
from typing import List, Optional, Tuple

import serial

logger = logging.getLogger(__name__)

# ── Instruction set ───────────────────────────────────────────────────────────
INST_PING       = 0x01
INST_READ       = 0x02
INST_WRITE      = 0x03
INST_REG_WRITE  = 0x04
INST_ACTION     = 0x05
INST_SYNC_WRITE = 0x83

# ── Control table addresses ───────────────────────────────────────────────────
ADDR_ID            = 0x05
ADDR_BAUD          = 0x06
ADDR_TORQUE_ENABLE = 0x28
ADDR_GOAL_POS      = 0x2A   # 2 bytes, little-endian
ADDR_GOAL_SPEED    = 0x2E   # 2 bytes
ADDR_PRESENT_POS   = 0x38   # 2 bytes
ADDR_PRESENT_SPEED = 0x3A   # 2 bytes
ADDR_PRESENT_LOAD  = 0x3C   # 2 bytes
ADDR_PRESENT_TEMP  = 0x3F   # 1 byte
ADDR_MOVING        = 0x42   # 1 byte
ADDR_LOCK          = 0x30   # EEPROM lock

BROADCAST_ID = 0xFE


def _checksum(data: bytes) -> int:
    return (~sum(data)) & 0xFF


class FeetechBus:
    """
    Half-duplex serial bus driver for Feetech STS/SCS servos.
    Handles TX/RX switching automatically using pyserial.
    """

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02):
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        self._ser.flushInput()
        logger.info("FeetechBus opened: %s @ %d baud", port, baudrate)

    # ── Low-level I/O ─────────────────────────────────────────────────────────

    def _send(self, servo_id: int, instruction: int, params: bytes = b"") -> None:
        length = len(params) + 2   # INST + CHECKSUM
        header = bytes([servo_id, length, instruction]) + params
        pkt = b"\xff\xff" + header + bytes([_checksum(header)])
        self._ser.flushInput()
        self._ser.write(pkt)

    def _recv(self, expected_data_len: int = 0) -> Optional[Tuple[int, int, bytes]]:
        """Return (servo_id, error_byte, data) or None on timeout/error."""
        # Wait for 0xFF 0xFF header
        deadline = time.monotonic() + 0.05
        buf = b""
        while time.monotonic() < deadline:
            b = self._ser.read(1)
            if not b:
                continue
            buf += b
            if len(buf) >= 2 and buf[-2:] == b"\xff\xff":
                break
        else:
            return None

        # Read fixed header: ID LEN
        hdr = self._ser.read(2)
        if len(hdr) < 2:
            return None
        servo_id, length = hdr[0], hdr[1]

        # length = ERR + data + CHECKSUM  =>  data_len = length - 2
        payload = self._ser.read(length)
        if len(payload) < length:
            return None

        err  = payload[0]
        data = payload[1:-1]
        chk  = payload[-1]

        computed = _checksum(bytes([servo_id, length, err]) + data)
        if computed != chk:
            logger.warning("Checksum mismatch: expected %02x got %02x", computed, chk)
            return None

        return servo_id, err, data

    # ── Public API ────────────────────────────────────────────────────────────

    def ping(self, servo_id: int) -> bool:
        self._send(servo_id, INST_PING)
        r = self._recv()
        return r is not None and r[0] == servo_id

    def read(self, servo_id: int, address: int, length: int) -> Optional[bytes]:
        self._send(servo_id, INST_READ, bytes([address, length]))
        r = self._recv(length)
        if r is None:
            return None
        return r[2]

    def write(self, servo_id: int, address: int, data: bytes) -> bool:
        self._send(servo_id, INST_WRITE, bytes([address]) + data)
        r = self._recv()
        return r is not None and r[1] == 0

    def sync_write(self, address: int, data_len: int,
                   id_data_pairs: List[Tuple[int, bytes]]) -> None:
        """Write same-length data to multiple servos simultaneously (no reply)."""
        params = bytes([address, data_len])
        for sid, d in id_data_pairs:
            params += bytes([sid]) + d
        self._send(BROADCAST_ID, INST_SYNC_WRITE, params)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def set_torque(self, servo_id: int, enable: bool) -> None:
        self.write(servo_id, ADDR_TORQUE_ENABLE, bytes([int(enable)]))

    def set_goal_position(self, servo_id: int, position: int,
                          speed: int = 0) -> None:
        """position: 0–4095, speed: 0=max, 1–1023."""
        data = struct.pack("<HH", position & 0xFFF, speed & 0x3FF)
        self.write(servo_id, ADDR_GOAL_POS, data)

    def get_present_position(self, servo_id: int) -> Optional[int]:
        raw = self.read(servo_id, ADDR_PRESENT_POS, 2)
        if raw is None or len(raw) < 2:
            return None
        val = struct.unpack("<H", raw)[0]
        # STS sign extension: bit 11 is direction bit
        if val > 2047:
            val = val - 4096
        return val

    def get_present_position_unsigned(self, servo_id: int) -> Optional[int]:
        raw = self.read(servo_id, ADDR_PRESENT_POS, 2)
        if raw is None or len(raw) < 2:
            return None
        return struct.unpack("<H", raw)[0] & 0x0FFF

    def is_moving(self, servo_id: int) -> bool:
        raw = self.read(servo_id, ADDR_MOVING, 1)
        return bool(raw and raw[0])

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()
