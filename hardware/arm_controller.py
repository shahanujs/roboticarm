"""
SO-ARM101 arm controller.
Wraps FeetechBus, exposes joint-space and (simple) task-space control.
"""

from __future__ import annotations

import math
import time
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

from hardware.feetech_bus import FeetechBus

logger = logging.getLogger(__name__)

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


class JointConfig:
    def __init__(self, cfg: dict):
        self.id        = cfg["id"]
        self.name      = cfg["name"]
        self.min_pos   = cfg["min_pos"]
        self.max_pos   = cfg["max_pos"]
        self.home_pos  = cfg["home_pos"]
        self.deg_min   = cfg["deg_min"]
        self.deg_max   = cfg["deg_max"]
        # ticks/deg
        self.ticks_per_deg = (self.max_pos - self.min_pos) / (self.deg_max - self.deg_min)

    def deg_to_ticks(self, degrees: float) -> int:
        degrees = max(self.deg_min, min(self.deg_max, degrees))
        return int(round(self.home_pos + degrees * self.ticks_per_deg))

    def ticks_to_deg(self, ticks: int) -> float:
        return (ticks - self.home_pos) / self.ticks_per_deg


class ArmController:
    """
    High-level controller for the SO-ARM101 arm.

    Joint ordering follows `config/arm_config.yaml`.
    All angles in **degrees** in the public API.
    """

    def __init__(self, config_path: str = "config/arm_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        arm_cfg = cfg["arm"]
        self._bus = FeetechBus(arm_cfg["port"], arm_cfg["baudrate"])
        self._joints: List[JointConfig] = [
            JointConfig(j) for j in arm_cfg["joints"]
        ]
        self.dof = len(self._joints)
        self.active_dof = int(arm_cfg.get("active_dof", self.dof))
        self.active_dof = max(1, min(self.active_dof, self.dof))
        self._joint_map: Dict[str, JointConfig] = {j.name: j for j in self._joints}
        self._workspace = cfg.get("workspace", {})
        self._home_angles = [0.0] * self.dof
        self._current_angles = [0.0] * self.dof

    # ── Setup / teardown ──────────────────────────────────────────────────────

    def connect(self) -> bool:
        found = []
        for j in self._joints:
            ok = self._bus.ping(j.id)
            if ok:
                self._bus.set_torque(j.id, True)
                found.append(j.name)
            else:
                logger.warning("Servo not responding: id=%d name=%s", j.id, j.name)
        logger.info("Connected joints: %s", found)
        if self.active_dof != self.dof:
            logger.info("Cartesian control limited to first %d/%d joints", self.active_dof, self.dof)
        return len(found) == self.dof

    def disconnect(self) -> None:
        self.go_home(blocking=True)
        for j in self._joints:
            self._bus.set_torque(j.id, False)
        self._bus.close()

    # ── Position read / write ─────────────────────────────────────────────────

    def get_joint_angles_deg(self) -> List[float]:
        angles = []
        for j in self._joints:
            ticks = self._bus.get_present_position_unsigned(j.id)
            angles.append(j.ticks_to_deg(ticks) if ticks is not None else 0.0)
        self._current_angles = angles
        return angles

    def get_joint_angles_rad(self) -> List[float]:
        return [a * DEG2RAD for a in self.get_joint_angles_deg()]

    def set_joint_angles_deg(self, angles: Sequence[float],
                              speed: int = 300, blocking: bool = False) -> None:
        assert len(angles) == self.dof
        pairs = []
        for j, deg in zip(self._joints, angles):
            ticks = j.deg_to_ticks(deg)
            import struct
            pairs.append((j.id, struct.pack("<HH", ticks & 0xFFF, speed & 0x3FF)))
        self._bus.sync_write(0x2A, 4, pairs)   # ADDR_GOAL_POS=0x2A, 4 bytes (pos+speed)
        self._current_angles = list(angles)
        if blocking:
            self._wait_for_motion()

    def set_joint_angles_by_index_deg(self, updates: Dict[int, float],
                                      speed: int = 300,
                                      blocking: bool = False) -> None:
        """Update only selected joints by index (0-based), leaving others untouched."""
        pairs = []
        for idx, deg in updates.items():
            if idx < 0 or idx >= self.dof:
                raise IndexError(f"Joint index out of range: {idx}")
            j = self._joints[idx]
            ticks = j.deg_to_ticks(deg)
            import struct
            pairs.append((j.id, struct.pack("<HH", ticks & 0xFFF, speed & 0x3FF)))

        if not pairs:
            return

        self._bus.sync_write(0x2A, 4, pairs)
        for idx, deg in updates.items():
            self._current_angles[idx] = float(deg)
        if blocking:
            self._wait_for_motion()

    def set_joint_angles_rad(self, angles: Sequence[float],
                              speed: int = 300, blocking: bool = False) -> None:
        self.set_joint_angles_deg([a * RAD2DEG for a in angles], speed, blocking)

    # ── Common poses ──────────────────────────────────────────────────────────

    def go_home(self, speed: int = 200, blocking: bool = True) -> None:
        self.set_joint_angles_deg(self._home_angles, speed, blocking)

    def go_to_observe(self, speed: int = 200, blocking: bool = True) -> None:
        """Safe observation pose — arm raised, looking forward."""
        presets = {
            5: [0.0, -30.0, 60.0, -30.0, 0.0],
            6: [0.0, -30.0, 60.0, -30.0, 0.0, 0.0],
        }
        observe = presets.get(self.dof, [0.0] * self.dof)
        self.set_joint_angles_deg(observe, speed, blocking)

    # ── FK (forward kinematics, DH convention) ────────────────────────────────

    @staticmethod
    def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,        sa,       ca,      d],
            [0,         0,        0,      1],
        ])

    def forward_kinematics(self, angles_deg: Optional[Sequence[float]] = None
                           ) -> np.ndarray:
        """
        Returns 4×4 homogeneous transform of end-effector in base frame.
        Uses approximate DH params — refine with calibrate.py.
        """
        if angles_deg is None:
            angles_deg = self._current_angles

        # Approximate SO-ARM101 DH parameters
        a     = [0.0,  0.1285, 0.124,  0.0,    0.0,   0.0]
        d     = [0.1025, 0.0,   0.0,  0.0,   0.0615, 0.0]
        alpha = [math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0]
        t_off = [0, -math.pi/2, 0, -math.pi/2, 0, 0]

        T = np.eye(4)
        n = min(self.dof, len(a), len(d), len(alpha), len(t_off), len(angles_deg))
        for i in range(n):
            theta = angles_deg[i] * DEG2RAD + t_off[i]
            T = T @ self.dh_matrix(theta, d[i], a[i], alpha[i])
        return T

    def get_end_effector_pose(self) -> np.ndarray:
        return self.forward_kinematics(self.get_joint_angles_deg())

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _wait_for_motion(self, timeout: float = 5.0) -> None:
        t0 = time.monotonic()
        time.sleep(0.1)
        while time.monotonic() - t0 < timeout:
            if not any(self._bus.is_moving(j.id) for j in self._joints):
                break
            time.sleep(0.05)

    def move_cartesian(self, target_xyz: Sequence[float],
                       target_rpy: Sequence[float] = (0, 0, 0),
                       speed: int = 300) -> bool:
        """
        Naive Jacobian pseudo-inverse IK step towards target_xyz.
        For production use replace with a full IK solver (ikpy, roboticstoolbox).
        """
        try:
            import ikpy.chain
        except ImportError:
            logger.error("ikpy not installed. Run: pip install ikpy")
            return False

        # Delegate to IK helper
        from hardware.ik_solver import IKSolver
        solver = IKSolver(self)
        angles = solver.solve(target_xyz, target_rpy)
        if angles is None:
            logger.warning("IK failed for target %s", target_xyz)
            return False
        # Freeze non-active joints to avoid unintended end-joint motion.
        if self.active_dof < self.dof:
            current = self.get_joint_angles_deg()
            angles = list(angles[:self.active_dof]) + list(current[self.active_dof:])
        self.set_joint_angles_deg(angles, speed, blocking=True)
        return True
