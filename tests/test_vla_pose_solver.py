"""Full-pose VLA IK must reproduce a reachable measured-configuration target in its trust region."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from vlarobot.backends.webots import kinematics
from vlarobot.backends.webots.backend import _rotation_to_quaternion
from vlarobot.backends.webots.config import ARM_JOINTS, SETTINGS
from vlarobot.backends.webots.pose_solver import BoundedPoseSolver
from vlarobot.core.types import Pose2D, Pose3D, RobotState, Vec3


def test_registered_full_pose_solver_uses_bounded_joint_changes(monkeypatch):
    calibration = Path(__file__).parents[1] / "configs/calibration/urdf-base-train-v1.yaml"
    monkeypatch.setattr(
        kinematics, "SETTINGS", replace(SETTINGS, urdf_base_registration_file=str(calibration))
    )
    kin = kinematics.Kinematics()
    q = np.zeros(len(kin.chain.links))
    for name in ARM_JOINTS:
        lo, hi = kin.bounds(name)
        q[kin.idx[name]] = (lo + hi) / 2
    q[kin.idx["torso_lift_joint"]] = 0.35
    before = kin.forward(q)
    joints = {name: float(q[kin.idx[name]]) for name in [*ARM_JOINTS, "torso_lift_joint"]}
    state = RobotState(
        joints,
        {},
        Pose2D(0, 0, 0),
        Pose3D(Vec3(*before[:3, 3]), _rotation_to_quaternion(before[:3, :3])),
        0.03,
    )
    target_q = q.copy()
    target_q[kin.idx["arm_3_joint"]] += 0.01
    target_q[kin.idx["arm_7_joint"]] += 0.01
    target = kin.forward(target_q)
    solver = BoundedPoseSolver(kin, max_joint_step_rad=0.15)
    solution = solver.solve(
        Pose3D(Vec3(*target[:3, 3]), _rotation_to_quaternion(target[:3, :3])), state
    )
    assert solution.position_error_m < 1e-5
    assert solution.orientation_error_rad < 1e-4
    assert max(abs(solution.joints[n] - joints[n]) for n in ARM_JOINTS) <= 0.15
