"""
IK Solver for SO-ARM101 using ikpy (or fallback Jacobian iteration).
Install: pip install ikpy
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hardware.arm_controller import ArmController

logger = logging.getLogger(__name__)

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class IKSolver:
    """
    Thin wrapper that provides IK via ikpy, falling back to Jacobian
    pseudo-inverse if ikpy is not available.
    """

    def __init__(self, arm: "ArmController"):
        self._arm = arm
        self._chain = None
        self._build_ikpy_chain()

    def _build_ikpy_chain(self) -> None:
        try:
            import ikpy.chain
            import ikpy.link
        except ImportError:
            logger.warning("ikpy not installed. Using Jacobian IK fallback.")
            return

        def make_link(name, bounds, tv, rot):
            # Support multiple ikpy API versions.
            try:
                return ikpy.link.URDFLink(
                    name=name,
                    bounds=bounds,
                    translation_vector=tv,
                    orientation=[0, 0, 0],
                    rotation=rot,
                )
            except TypeError:
                return ikpy.link.URDFLink(
                    name=name,
                    bounds=bounds,
                    origin_translation=tv,
                    origin_orientation=[0, 0, 0],
                    rotation=rot,
                )

        try:
            # SO-ARM101 DH parameters (approximate)
            links = [
                ikpy.link.OriginLink(),
                make_link("base",        (-2.62, 2.62), [0,      0,      0.1025], [0, 0, 1]),
                make_link("shoulder",    (-1.57, 1.57), [0.1285, 0,      0],      [0, 1, 0]),
                make_link("elbow",       (-2.36, 2.36), [0.124,  0,      0],      [0, 1, 0]),
                make_link("wrist_pitch", (-1.57, 1.57), [0,      0,      0],      [0, 1, 0]),
                make_link("wrist_roll",  (-3.14, 3.14), [0,      0,      0.0615], [1, 0, 0]),
                make_link("wrist_yaw",   (-1.57, 1.57), [0,      0,      0],      [0, 0, 1]),
            ]
            self._chain = ikpy.chain.Chain(links)
        except Exception as e:
            logger.warning("ikpy chain init failed (%s). Using Jacobian IK fallback.", e)
            self._chain = None

    def solve(self, target_xyz: Sequence[float],
              target_rpy: Sequence[float] = (0, 0, 0)) -> Optional[List[float]]:
        """
        Returns joint angles in degrees, or None if IK failed.
        """
        if self._chain is not None:
            try:
                return self._solve_ikpy(target_xyz, target_rpy)
            except Exception as e:
                logger.warning("ikpy solve failed (%s). Falling back to Jacobian IK.", e)
        return self._solve_jacobian(target_xyz)

    def _solve_ikpy(self, target_xyz, target_rpy) -> Optional[List[float]]:
        import ikpy.chain
        target_pos = np.asarray(target_xyz, dtype=np.float64)

        # Build initial position vector sized to the chain links.
        current_joint_rad = [a * DEG2RAD for a in self._arm.get_joint_angles_deg()]
        required_len = len(self._chain.links)
        current_rad = [0.0] + current_joint_rad
        if len(current_rad) < required_len:
            current_rad += [0.0] * (required_len - len(current_rad))
        elif len(current_rad) > required_len:
            current_rad = current_rad[:required_len]

        result = self._chain.inverse_kinematics(
            target_position=target_pos,
            initial_position=current_rad,
        )

        # Drop origin link and keep only active arm joints.
        active_dof = getattr(self._arm, "active_dof", self._arm.dof)
        angles_deg = [a * RAD2DEG for a in result[1:1 + active_dof]]
        return angles_deg

    def _solve_jacobian(self, target_xyz: Sequence[float],
                        max_iter: int = 200, tol: float = 1e-3) -> Optional[List[float]]:
        """Damped Jacobian pseudo-inverse IK fallback."""
        angles_rad = np.array(self._arm.get_joint_angles_rad())
        target = np.array(target_xyz)

        for _ in range(max_iter):
            J = self._numerical_jacobian(angles_rad)
            T = self._arm.forward_kinematics([a * RAD2DEG for a in angles_rad])
            current = T[:3, 3]
            err = target - current
            if np.linalg.norm(err) < tol:
                return [a * RAD2DEG for a in angles_rad]

            # Damped least-squares
            lam = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
            d_theta = J_pinv @ err
            angles_rad = angles_rad + 0.5 * d_theta

        logger.warning("Jacobian IK did not converge")
        return None

    def _numerical_jacobian(self, angles_rad: np.ndarray,
                             eps: float = 1e-4) -> np.ndarray:
        J = np.zeros((3, len(angles_rad)))
        T0 = self._arm.forward_kinematics([a * RAD2DEG for a in angles_rad])
        p0 = T0[:3, 3]
        for i in range(len(angles_rad)):
            dq = angles_rad.copy()
            dq[i] += eps
            T1 = self._arm.forward_kinematics([a * RAD2DEG for a in dq])
            J[:, i] = (T1[:3, 3] - p0) / eps
        return J

    @staticmethod
    def _rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        return Rz @ Ry @ Rx
