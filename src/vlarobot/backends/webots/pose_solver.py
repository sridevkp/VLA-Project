"""Optional VLA full-pose IK; the scripted baseline's solver and behavior stay unchanged."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vlarobot.backends.webots.config import ARM_JOINTS
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import Pose3D, RobotState
from vlarobot.vla.action_adapter import PoseSolution


def quaternion_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class BoundedPoseSolver:
    """One deterministic warm-start solve inside the existing joint trust-region bound."""

    def __init__(self, kinematics: Kinematics, *, max_joint_step_rad: float) -> None:
        self.kinematics = kinematics
        self.max_joint_step_rad = max_joint_step_rad

    def solve(self, pose: Pose3D, state: RobotState) -> PoseSolution:
        kin = self.kinematics
        seed = np.zeros(len(kin.chain.links))
        for name, value in state.joint_pos.items():
            if name in kin.idx:
                seed[kin.idx[name]] = value
        chain = kin._trust_region_chain(seed, self.max_joint_step_rad)
        target_rotation = quaternion_matrix(pose.orientation)
        solved = chain.inverse_kinematics(
            pose.position.as_array(),
            target_orientation=target_rotation,
            orientation_mode="all",
            initial_position=seed,
        )
        actual = kin.forward(solved)
        orientation_error = math.acos(
            float(np.clip((np.trace(target_rotation @ actual[:3, :3].T) - 1) / 2, -1, 1))
        )
        return PoseSolution(
            {name: float(solved[kin.idx[name]]) for name in ARM_JOINTS},
            float(np.linalg.norm(actual[:3, 3] - pose.position.as_array())),
            orientation_error,
        )


class GateNormalizedPoseSolver(BoundedPoseSolver):
    """V2 fallback when an otherwise bounded solve misses a pose residual gate."""

    def __init__(
        self,
        kinematics: Kinematics,
        *,
        max_joint_step_rad: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> None:
        super().__init__(kinematics, max_joint_step_rad=max_joint_step_rad)
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.fallback_solutions = 0

    def solve(self, pose: Pose3D, state: RobotState) -> PoseSolution:
        original = super().solve(pose, state)
        if (
            original.position_error_m <= self.position_tolerance_m
            and original.orientation_error_rad <= self.orientation_tolerance_rad
        ):
            return original

        kin = self.kinematics
        current = np.asarray([state.joint_pos[name] for name in ARM_JOINTS])
        bounds = np.asarray([kin.bounds(name) for name in ARM_JOINTS])
        lower = np.maximum(bounds[:, 0], current - self.max_joint_step_rad)
        upper = np.minimum(bounds[:, 1], current + self.max_joint_step_rad)
        if np.any(lower >= upper):
            return original
        initial = np.zeros(len(kin.chain.links))
        for name, value in state.joint_pos.items():
            if name in kin.idx:
                initial[kin.idx[name]] = value
        indices = [kin.idx[name] for name in ARM_JOINTS]
        position = pose.position.as_array()
        rotation = quaternion_matrix(pose.orientation)

        def residual(joints: np.ndarray) -> np.ndarray:
            full = initial.copy()
            full[indices] = joints
            actual = kin.forward(full)
            return np.r_[
                (actual[:3, 3] - position) / self.position_tolerance_m,
                Rotation.from_matrix(rotation @ actual[:3, :3].T).as_rotvec()
                / self.orientation_tolerance_rad,
            ]

        candidate = least_squares(residual, np.clip(current, lower, upper), bounds=(lower, upper))
        error = residual(candidate.x)
        pos = float(np.linalg.norm(error[:3]) * self.position_tolerance_m)
        rot = float(np.linalg.norm(error[3:]) * self.orientation_tolerance_rad)
        if not (
            np.all(candidate.x >= lower)
            and np.all(candidate.x <= upper)
            and pos <= self.position_tolerance_m
            and rot <= self.orientation_tolerance_rad
        ):
            return original
        self.fallback_solutions += 1
        return PoseSolution(dict(zip(ARM_JOINTS, map(float, candidate.x), strict=True)), pos, rot)
