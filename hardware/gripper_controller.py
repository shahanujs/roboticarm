"""
Gripper controller for SO-ARM101.
The gripper servo may live on the same Feetech bus as the arm joints or a
separate ttyACM device — controlled by config/arm_config.yaml.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import yaml

from hardware.feetech_bus import FeetechBus

logger = logging.getLogger(__name__)


class GripperController:
    """
    Controls the parallel-jaw gripper on SO-ARM101.

    Positions are in servo ticks (0–4095).
    open_pos  > close_pos  (wider opening = higher tick value).
    """

    GRASP_DETECT_LOAD_THRESHOLD = 200   # servo load units

    def __init__(self, config_path: str = "config/arm_config.yaml",
                 shared_bus: Optional[FeetechBus] = None):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        g = cfg["gripper"]
        self._id        = g["id"]
        self._open_pos  = g["open_pos"]
        self._close_pos = g["close_pos"]
        self._grasp_thr = g.get("grasp_force", 300)

        if shared_bus is not None:
            self._bus = shared_bus
            self._owns_bus = False
        else:
            self._bus = FeetechBus(g["port"], g["baudrate"])
            self._owns_bus = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        ok = self._bus.ping(self._id)
        if ok:
            self._bus.set_torque(self._id, True)
            logger.info("Gripper connected (id=%d)", self._id)
        else:
            logger.warning("Gripper servo not responding (id=%d)", self._id)
        return ok

    def disconnect(self, park_open: bool = False) -> None:
        try:
            if park_open:
                self.open(blocking=True)
            self._bus.set_torque(self._id, False)
        except Exception as e:
            logger.debug("Gripper disconnect torque-off skipped: %s", e)
        finally:
            if self._owns_bus:
                try:
                    self._bus.close()
                except Exception as e:
                    logger.debug("Gripper bus close skipped: %s", e)

    def set_torque(self, enable: bool) -> None:
        self._bus.set_torque(self._id, bool(enable))

    def disable_torque(self, close_bus: bool = False) -> None:
        """Force gripper torque off without motion commands.

        Set close_bus=False to keep reading gripper width afterward.
        """
        self._bus.set_torque(self._id, False)
        if close_bus and self._owns_bus:
            self._bus.close()

    # ── Commands ──────────────────────────────────────────────────────────────

    def open(self, speed: int = 200, blocking: bool = False) -> None:
        self._bus.set_goal_position(self._id, self._open_pos, speed)
        if blocking:
            self._wait()

    def close(self, speed: int = 200, blocking: bool = False) -> None:
        self._bus.set_goal_position(self._id, self._close_pos, speed)
        if blocking:
            self._wait()

    def set_width(self, fraction: float, speed: int = 200) -> None:
        """Set gripper width as fraction [0=closed, 1=open]."""
        fraction = max(0.0, min(1.0, fraction))
        ticks = int(self._close_pos + fraction * (self._open_pos - self._close_pos))
        self._bus.set_goal_position(self._id, ticks, speed)

    def get_width_fraction(self) -> float:
        """Return current opening as fraction [0=closed, 1=open]."""
        ticks = self._bus.get_present_position_unsigned(self._id)
        if ticks is None:
            return 0.0
        span = self._open_pos - self._close_pos
        return max(0.0, min(1.0, (ticks - self._close_pos) / span))

    def grasp_and_detect(self, speed: int = 150,
                          timeout: float = 2.0) -> bool:
        """
        Close gripper and return True if an object is detected (stall / load spike).
        """
        self.close(speed=speed, blocking=False)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not self._bus.is_moving(self._id):
                # Gripper stopped — check if it's stalled (object) or fully closed
                pos = self._bus.get_present_position_unsigned(self._id)
                if pos is not None and pos > self._close_pos + self._grasp_thr:
                    logger.info("Grasp detected at pos=%d", pos)
                    return True
                return False
            time.sleep(0.02)
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _wait(self, timeout: float = 3.0) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not self._bus.is_moving(self._id):
                break
            time.sleep(0.05)
