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


def test_v2_normalized_ik_recovers_recorded_target_without_expanding_bounds(monkeypatch):
    import json

    import pytest

    from vlarobot.execution.vla_campaign import make_vla_adapter
    from vlarobot.execution.vla_episode import project_recovery_chunk

    calibration = Path(__file__).parents[1] / "configs/calibration/urdf-base-train-v1.yaml"
    monkeypatch.setattr(
        kinematics, "SETTINGS", replace(SETTINGS, urdf_base_registration_file=str(calibration))
    )
    witness = json.loads(
        Path(__file__).with_name("fixtures").joinpath("v2_train_ik_rejection.json").read_text()
    )
    state = RobotState(
        witness["joints"],
        {},
        Pose2D(0, 0, 0),
        Pose3D(Vec3(*witness["tcp_xyz"]), tuple(witness["tcp_quat"])),
        witness["gripper"],
    )
    old = make_vla_adapter()
    values = project_recovery_chunk(
        np.asarray([witness["raw_action"]]), old.safety_monitor.limits.gripper_opening
    )[0]
    with pytest.raises(ValueError, match="IK pose residual"):
        old.decode(values, state)
    recovered = make_vla_adapter(recovery_ik=True).decode(values, state)
    assert recovered.position_error_m < old.position_tolerance_m
    assert recovered.orientation_error_rad < old.orientation_tolerance_rad
    assert max(abs(v - state.joint_pos[n]) for n, v in recovered.joint_target.q.items()) <= 0.15
    assert old.safety_monitor.validate(recovered.joint_target, state).accepted
