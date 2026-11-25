"""Optional VLA full-pose IK; the scripted baseline's solver and behavior stay unchanged."""

from __future__ import annotations

import math

import numpy as np

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
