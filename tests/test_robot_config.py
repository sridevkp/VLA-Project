from pathlib import Path

import pytest

from vlarobot.backends.webots.config import (
    ALIGN_TOL,
    APPROACH_H,
    ARM_JOINTS,
    CAN_DEF,
    CONTROLLER_DIR,
    FINGER_JOINTS,
    FINGER_SPEED,
    GRASP_Z,
    GRIP_FORCE,
    HEAD_JOINTS,
    IK_TOL,
    IK_TOL_GRASP,
    PLACE_BASE,
    PLACE_CLEAR,
    SAFETY_PROFILE,
    SAFETY_PROFILE_NAME,
    SAFETY_SETTINGS_PATH,
    SETTINGS,
    SETTINGS_PATH,
    TIME_STEP,
    TOOL_OFFSET,
    URDF_PATH,
    RobotSettings,
)


def test_yaml_backed_settings_preserve_original_controller_defaults() -> None:
    assert TIME_STEP == 32
    assert CAN_DEF == "CAN"
    assert PLACE_BASE == [-1.07, -3.9, 0.8]
    assert GRASP_Z == 0.02
    assert APPROACH_H == 0.12
    assert PLACE_CLEAR == 0.01
    assert IK_TOL == 0.015
    assert IK_TOL_GRASP == 0.005
    assert ALIGN_TOL == 0.005
    assert GRIP_FORCE is None
    assert FINGER_SPEED == 0.03
    assert ARM_JOINTS == [f"arm_{index}_joint" for index in range(1, 8)]
    assert FINGER_JOINTS == [
        "gripper_left_finger_joint",
        "gripper_right_finger_joint",
    ]
    assert HEAD_JOINTS == ["head_1_joint", "head_2_joint"]
    assert TOOL_OFFSET == [0.0, 0.0, -0.1741]
    assert CONTROLLER_DIR == (
        Path(__file__).parents[1] / "src" / "vlarobot" / "backends" / "webots"
    )
    assert URDF_PATH == CONTROLLER_DIR / "tiago.urdf"
    assert SETTINGS_PATH == Path(__file__).parents[1] / "configs" / "robot.yaml"
    assert SETTINGS.skip_align_check is True
    assert SETTINGS.max_tilt_deg == 15
    assert SETTINGS.min_grasp_open == 0.004
    assert SETTINGS.lift_min == 0.04
    assert SETTINGS.max_retries == 2
    assert SETTINGS.torso_height == 0.35
    assert SETTINGS.fingers_inverted is False
    assert SETTINGS.gripper_test is False
    assert SETTINGS.head_pan == 0.0
    assert SETTINGS.head_tilt == -0.77
    assert SETTINGS.gripper_opening_limits == (0.0, 0.045)
    assert SETTINGS.gripper_force_max == 100.0
    assert SETTINGS.safety_command_horizon == 1.0
    assert SAFETY_SETTINGS_PATH == Path(__file__).parents[1] / "configs" / "safety.yaml"
    assert SAFETY_PROFILE_NAME == "original_envelope"
    assert SAFETY_PROFILE.workspace_min == (
        -1.186885822749492,
        -3.9580864458899137,
        0.6648214194162648,
    )
    assert SAFETY_PROFILE.workspace_max == (
        0.7727366605590086,
        -0.11329506483472998,
        1.063048982572836,
    )
    assert SAFETY_PROFILE.margin_m == 0.1


def test_robot_settings_reject_missing_and_unknown_yaml_keys(tmp_path: Path) -> None:
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text("time_step: 32\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid robot settings keys"):
        RobotSettings.from_yaml(bad_settings)
