import math
from typing import cast

import pytest

from vlarobot.backends.webots.config import SAFETY_SETTINGS
from vlarobot.core.types import (
    Action,
    BaseGoal,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    JointTarget,
    Pose2D,
    Pose3D,
    RobotState,
    Vec3,
)
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor


@pytest.fixture
def monitor_and_state() -> tuple[SafetyMonitor, RobotState]:
    monitor = SafetyMonitor(
        SafetyLimits(
            joint_position={"arm_1_joint": (-2.0, 2.0)},
            joint_velocity={"arm_1_joint": 1.0},
            workspace_min=(0.0, -1.0, 0.0),
            workspace_max=(1.5, 1.0, 1.8),
        )
    )
    state = RobotState(
        joint_pos={"arm_1_joint": 0.0},
        joint_vel={"arm_1_joint": 0.0},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.02,
    )
    return monitor, state


def test_joint_position_velocity_and_unknown_joint_are_rejected(monitor_and_state) -> None:
    monitor, state = monitor_and_state

    assert monitor.validate(JointTarget({"arm_1_joint": 0.5}), state).accepted
    assert not monitor.validate(JointTarget({"arm_1_joint": 2.1}), state).accepted
    assert not monitor.validate(JointTarget({"arm_1_joint": 1.1}), state).accepted
    assert not monitor.validate(JointTarget({"arm_2_joint": 0.1}), state).accepted
    assert not monitor.validate(JointTarget({"arm_1_joint": math.nan}), state).accepted
    assert not monitor.validate(JointTarget({}), state).accepted


def test_end_effector_pose_and_delta_are_checked(monitor_and_state) -> None:
    monitor, state = monitor_and_state
    safe_pose = EEPoseTarget(
        Pose3D(Vec3(0.6, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        gripper=None,
        duration=0.5,
    )

    assert monitor.validate(safe_pose, state).accepted
    assert not monitor.validate(
        EEPoseTarget(
            Pose3D(Vec3(2.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
            gripper=None,
            duration=0.5,
        ),
        state,
    ).accepted
    assert not monitor.validate(
        EEPoseTarget(
            Pose3D(Vec3(0.6, 0.0, 0.5), (0.0, 0.0, 0.0, 0.0)),
            gripper=None,
            duration=0.5,
        ),
        state,
    ).accepted
    assert not monitor.validate(
        EEPoseTarget(safe_pose.pose, gripper=None, duration=0.0), state
    ).accepted
    assert not monitor.validate(
        EEDelta(Vec3(2.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), 0.02),
        state,
    ).accepted


def test_base_and_gripper_commands_reject_malformed_values(monitor_and_state) -> None:
    monitor, state = monitor_and_state

    assert monitor.validate(BaseGoal(Pose2D(2.0, 0.0, 0.0)), state).accepted
    assert not monitor.validate(BaseGoal(Pose2D(math.nan, 0.0, 0.0)), state).accepted
    assert monitor.validate(GripperCmd(0.02, force=50.0), state).accepted
    assert not monitor.validate(GripperCmd(0.05), state).accepted
    assert not monitor.validate(GripperCmd(0.02, force=101.0), state).accepted
    assert not monitor.validate(GripperCmd(math.inf), state).accepted
    assert not monitor.validate(Action(), state).accepted


def test_malformed_runtime_action_values_are_rejected(monitor_and_state) -> None:
    monitor, state = monitor_and_state

    assert not monitor.validate(JointTarget({"arm_1_joint": cast(float, "bad")}), state).accepted
    assert not monitor.validate(GripperCmd(cast(float, None)), state).accepted
    assert not monitor.validate(
        EEPoseTarget(
            Pose3D(Vec3(cast(float, "bad"), 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
            gripper=None,
            duration=1.0,
        ),
        state,
    ).accepted


@pytest.mark.parametrize(
    "limits",
    [
        {"command_horizon": 0.0},
        {"gripper_force_max": math.inf},
        {"workspace_min": (0.0, 0.0, 0.0), "workspace_max": (0.0, 1.0, 1.0)},
        {"joint_velocity": {"arm_1_joint": 0.0}},
        {"joint_position": {"arm_1_joint": (1.0, -1.0)}},
    ],
)
def test_invalid_safety_limit_configuration_is_rejected(limits) -> None:
    with pytest.raises(ValueError):
        SafetyLimits(**limits)


def test_original_pick_place_waypoints_fit_original_envelope() -> None:
    waypoints = (
        (0.20367229767278827, -0.9854553087033255, 0.963048982572836),
        (0.6727366605590026, -0.21329506483472999, 0.7648214194162674),
        (0.6727366605590026, -0.21329506483472999, 0.8848214194162674),
        (-1.0773279823208584, -3.8420458694668804, 0.9362847553793852),
        (-1.0773279823208584, -3.8420458694668804, 0.8162847553793852),
        (-1.086885822749492, -3.8580864458899136, 0.9351191204798371),
        (-1.086885822749492, -3.8580864458899136, 0.8151191204798371),
    )
    profile = SAFETY_SETTINGS.profiles["original_envelope"]
    monitor = SafetyMonitor(
        SafetyLimits(
            workspace_min=profile.workspace_min,
            workspace_max=profile.workspace_max,
        )
    )
    state = RobotState(
        joint_pos={},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.0, 0.0, 0.8), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.02,
    )

    for point in waypoints:
        target = EEPoseTarget(
            Pose3D(Vec3(*point), (0.0, 0.0, 0.0, 1.0)),
            gripper=None,
            duration=0.5,
        )
        assert monitor.validate(target, state).accepted


@pytest.mark.parametrize("profile_name", ["original_envelope", "strict"])
def test_out_of_box_target_is_rejected_for_each_profile(profile_name: str) -> None:
    profile = SAFETY_SETTINGS.profiles[profile_name]
    monitor = SafetyMonitor(
        SafetyLimits(
            workspace_min=profile.workspace_min,
            workspace_max=profile.workspace_max,
        )
    )
    state = RobotState(
        joint_pos={},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.0, -1.0, 0.8), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.02,
    )
    target = EEPoseTarget(
        Pose3D(Vec3(0.0, 0.0, 0.8), (0.0, 0.0, 0.0, 1.0)),
        gripper=None,
        duration=0.5,
    )

    assert not monitor.validate(target, state).accepted


def test_safety_profiles_keep_strict_box_inside_original_envelope() -> None:
    envelope = SAFETY_SETTINGS.profiles["original_envelope"]
    strict = SAFETY_SETTINGS.profiles["strict"]

    assert all(
        outer_min < inner_min < inner_max < outer_max
        for outer_min, inner_min, inner_max, outer_max in zip(
            envelope.workspace_min,
            strict.workspace_min,
            strict.workspace_max,
            envelope.workspace_max,
            strict=True,
        )
    )


def test_workspace_rejection_is_logged(monitor_and_state, caplog) -> None:
    monitor, state = monitor_and_state
    target = EEPoseTarget(
        Pose3D(Vec3(2.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        gripper=None,
        duration=0.5,
    )

    assert not monitor.validate(target, state).accepted
    assert "position x=2.0 is outside workspace" in caplog.text
