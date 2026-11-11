from __future__ import annotations

import numpy as np

from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import (
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
    robot = RobotState(
        joint_pos={},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(*ee_base), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.04,
        base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        diagnostics={"finger_midpoint_world_xyz_m": list(pad_mid_world)},
    )
    return Observation(t=t, rgb=np.zeros((1, 1, 3), dtype=np.uint8), depth=None, cam_info=cam, robot=robot)


class FakeServoBackend:
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
        from vlarobot.core.types import ApplyResult

        return ApplyResult(True)


def test_gt_pad_pregrasp_servo_converges_with_biased_fk():
    # FK/tool bias: pad starts 50mm off in XY from can projection, but measured servo corrects.
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.65, 0.2, 0.92))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="gt_pad")
    obs = _obs(backend.pad_world)

    def fake_move_to(_backend, current_q, target, _state, waypoint_records=None):
        # Simulate that commanded base-frame XY correction moves measured pad by same world delta
        # (base at identity here), while preserving a wrong-tool-axis effect in Z irrelevant to XY.
        prev = np.array(target, dtype=float)
        # infer delta from previous hover target stored on policy via closure is complex; instead
        # move pad toward target's XY intent by applying target change approximately.
        nonlocal_target = policy._last_servo_target if hasattr(policy, "_last_servo_target") else prev
        delta = np.array(target, dtype=float) - np.array(nonlocal_target, dtype=float)
        backend.pad_world = backend.pad_world + np.array([delta[0], delta[1], 0.0])
        policy._last_servo_target = np.array(target, dtype=float)
        new_obs = _obs(backend.pad_world, ee_base=(float(target[0]), float(target[1]), float(target[2])))
        return np.zeros(len(kin.chain.links)), new_obs

    policy._last_servo_target = np.array([0.7, 0.2, 0.92], dtype=float)
    policy._move_to = fake_move_to  # type: ignore[method-assign]
    q = np.zeros(len(kin.chain.links))
    out = policy._servo_correct_pregrasp_xy(backend, obs, q, np.array([0.7, 0.2, 0.92]))
    assert out is not None
    _q, obs2, target = out
    err = backend.can_world[:2] - np.array(obs2.robot.diagnostics["finger_midpoint_world_xyz_m"])[:2]
    assert float(np.linalg.norm(err)) < 0.004


def test_gt_pad_pregrasp_open_loop_bias_would_fail_without_servo():
    # Demonstrates the biased condition: initial XY error > 4mm (open-loop would be misaligned).
    backend = FakeServoBackend(can_world=(0.7, 0.2, 0.8), pad_world=(0.65, 0.2, 0.92))
    kin = Kinematics()
    policy = ScriptedCanPolicy(kinematics=kin, grasp_servo="none")
    obs = _obs(backend.pad_world)
    meas = policy._gt_pad_xy_error_world(backend, obs)
    assert meas is not None
    _e_xy, e_norm, _can, _pad, _src = meas
    assert e_norm > 0.04  # 50mm bias present
    assert policy.grasp_servo == "none"
