from __future__ import annotations

import numpy as np

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
