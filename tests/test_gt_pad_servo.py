from __future__ import annotations

import numpy as np

import vlarobot.execution.scripted_can as sc
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import (
    ApplyResult,
    CameraInfo,
    GroundTruth,
    Observation,
    Pose2D,
    Pose3D,
    RobotState,
    Vec3,
)
from vlarobot.execution.scripted_can import ScriptedCanPolicy


def _obs(pad_mid_world, ee_base=(0.5, 0.0, 0.9), t=0.0):
    cam = CameraInfo(
        K=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        T_base_cam=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        width=1,
        height=1,
    )
    pad = np.asarray(pad_mid_world, dtype=float)
    # ADR-0022: the gt_pad servo reference is the arm_7 Solid pose composed with the
    # measured offset (0, 0, +0.202825) in the sim arm_7 frame. The fake provides the
    # Solid position/orientation that composes to the intended pad midpoint (identity
    # base/world), and the backend-composed diagnostics key itself.
    arm7_pos = (pad - np.array([0.0, 0.0, 0.202825])).tolist()
    robot = RobotState(
        joint_pos={},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(*ee_base), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.04,
        base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        diagnostics={
            "pad_midpoint_arm7_solid_measured_world_xyz_m": pad.tolist(),
            "arm_7_joint_ground_truth_world_xyz_m": arm7_pos,
            "arm_7_joint_ground_truth_orientation_rows": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    )
    return Observation(
        t=t,
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        depth=None,
        cam_info=cam,
        robot=robot,
    )


class FakeServoBackend:
    """Minimal backend for gt_pad servo logic (Webots-free)."""

    def __init__(self, can_world=(0.7, 0.2, 0.8), pad_world=(0.65, 0.2, 0.92)):
        self.can_world = np.array(can_world, dtype=float)
        self.pad_world = np.array(pad_world, dtype=float)
        self.t = 0.0

    def get_ground_truth(self):
        return GroundTruth(
            object_poses={"CAN": Pose3D(Vec3(*self.can_world.tolist()), (0, 0, 0, 1))}
        )

    def step(self, n=1):
        self.t += 0.032 * n
        return _obs(self.pad_world, t=self.t)

    def apply(self, action):
        return ApplyResult(True)


def test_open_loop_fails_with_50mm_wrong_tool_axis_fk_bias():
    """Wrong tool-axis/FK bias (50 mm XY): open-loop (no servo) would be misaligned."""
    # Measured pad midpoint is 50 mm away from can XY projection at hover.
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.65, 0.2, 0.92))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="none")
    obs = _obs(backend.pad_world)
    meas = policy._gt_pad_xy_error_world(backend, obs)
    assert meas is not None
    _e_xy, e_norm, _can, _pad, _src = meas
    assert e_norm > 0.049  # ~50 mm wrong-tool-axis/FK bias present
    assert policy.grasp_servo == "none"


def test_gt_pad_servo_converges_with_50mm_wrong_tool_axis_fk_bias():
    """With same 50 mm wrong tool-axis/FK bias, gt_pad servo converges to <4 mm."""
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.65, 0.2, 0.92))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)

    def fake_move_to(_backend, current_q, target, _state, waypoint_records=None):
        # Commanded target comes from biased FK/planning, but measured pad moves by the
        # requested base-frame XY correction (identity base/world in this fake).
        prev = np.array(target, dtype=float)
        nonlocal_target = (
            policy._last_servo_target if hasattr(policy, "_last_servo_target") else prev
        )
        delta = np.array(target, dtype=float) - np.array(nonlocal_target, dtype=float)
        backend.pad_world = backend.pad_world + np.array([delta[0], delta[1], 0.0])
        policy._last_servo_target = np.array(target, dtype=float)
        ee_base = (float(target[0]), float(target[1]), float(target[2]))
        return np.zeros(len(kin.chain.links)), _obs(backend.pad_world, ee_base=ee_base)

    # Initial planned hover target (could be biased by wrong tool-axis FK); servo corrects it.
    policy._last_servo_target = np.array([0.7, 0.2, 0.92], dtype=float)
    policy._move_to = fake_move_to  # type: ignore[method-assign]
    q = np.zeros(len(kin.chain.links))
    out = policy._servo_correct_pregrasp_xy(backend, obs, q, np.array([0.7, 0.2, 0.92]))
    assert out is not None
    _q, obs2, _target = out
    err = policy._gt_pad_xy_error_world(backend, obs2)
    assert err is not None
    _e_xy, e_norm, _can, _pad, _src = err
    assert e_norm < 0.004


def test_gt_pad_descend_refuses_when_probe_offset_is_none(monkeypatch):
    backend = FakeServoBackend()
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)
    monkeypatch.setattr(sc, "PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M", None)
    res = policy._servo_descend_to_grasp(
        backend, obs, np.zeros(len(kin.chain.links)), np.array([0.7, 0.2, 0.92])
    )
    assert res is None
    assert policy._last_failure_reason is not None
    assert "probe_pad_reference_to_pad_bottom_m is null" in policy._last_failure_reason


def test_gt_pad_descend_stops_at_probe_lid_height():
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.7, 0.2, 0.88))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)

    def fake_move_to(_backend, current_q, target, _state, waypoint_records=None):
        backend.pad_world[2] = float(target[2])
        ee_base = (float(target[0]), float(target[1]), float(target[2]))
        return current_q, _obs(backend.pad_world, ee_base=ee_base)

    policy._move_to = fake_move_to  # type: ignore[method-assign]
    q = np.zeros(len(kin.chain.links))
    res = policy._servo_descend_to_grasp(backend, obs, q, np.array([0.7, 0.2, 0.88]))
    assert res is not None
    _q_out, _obs_out = res
    can_lid_z = 0.8 + sc.CAN_HALF_HEIGHT_M
    assert sc.PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M is not None
    pad_bottom_z = backend.pad_world[2] - sc.PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M
    assert pad_bottom_z <= can_lid_z + 1e-6


def test_gt_pad_close_guard_passes_when_aligned():
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.7, 0.2, 0.86))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)
    q = np.zeros(len(kin.chain.links))
    target = np.array([0.7, 0.2, 0.86])
    res = policy._guard_close_alignment(backend, obs, q, target)
    assert res is not None
    _q_res, _obs_res, refined_target = res
    assert np.allclose(refined_target, target)


def test_gt_pad_close_guard_realigns_when_misaligned():
    # 10 mm error in X (> CLOSE_ALIGN_TOL of 8 mm)
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.69, 0.2, 0.86))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)
    q = np.zeros(len(kin.chain.links))
    target = np.array([0.69, 0.2, 0.86])

    def fake_move_linear(_backend, _obs_in, refined_target, _state, target_provider=None):
        # Reposition pad to match refined target XY (aligned to can)
        backend.pad_world[0] = 0.7
        return q, _obs(backend.pad_world, ee_base=refined_target)

    policy._move_linear = fake_move_linear  # type: ignore[method-assign]
    res = policy._guard_close_alignment(backend, obs, q, target)
    assert res is not None
    _q_out, _obs_out, _refined = res
    assert policy._last_failure_reason is None


def test_gt_pad_close_guard_fails_when_residual_exceeds_tol():
    # 20 mm error in X (> CLOSE_ALIGN_TOL of 8 mm)
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.68, 0.2, 0.86))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)
    q = np.zeros(len(kin.chain.links))
    target = np.array([0.68, 0.2, 0.86])

    def fake_move_linear(_backend, _obs_in, refined_target, _state, target_provider=None):
        # Residual error remains 15 mm > 8 mm
        backend.pad_world[0] = 0.685
        return q, _obs(backend.pad_world, ee_base=refined_target)

    policy._move_linear = fake_move_linear  # type: ignore[method-assign]
    res = policy._guard_close_alignment(backend, obs, q, target)
    assert res is None
    assert policy._last_failure_reason == "misaligned at close"


def test_gt_pad_close_guard_fails_when_pad_midpoint_unavailable():
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.7, 0.2, 0.86))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)
    # Clear pad diagnostics
    obs.robot.diagnostics.clear()
    q = np.zeros(len(kin.chain.links))
    target = np.array([0.7, 0.2, 0.86])
    res = policy._guard_close_alignment(backend, obs, q, target)
    assert res is None
    assert policy._last_failure_reason == "gt_pad close guard failed: pad midpoint unavailable"
