from pathlib import Path

import pytest

from vlarobot.backends.webots.config import (
    ALIGN_TOL,
    APPROACH_H,
    ARM_JOINTS,
    CAN_DEF,
    CAN_PLACE_Z_TOLERANCE_M,
    CAN_RADIUS_M,
    CAN_TILT_LIMIT_DEG,
    CLOSE_ALIGN_TOL,
    CLOSED_LOOP_WAYPOINTS,
    CONTROLLER_DIR,
    FINGER_JOINTS,
    FINGER_SPEED,
    FINGER_TIP_OFFSET_M,
    FK_GROUND_TRUTH_TOLERANCE_M,
    GRASP_Z,
    GRIP_FORCE,
    HEAD_JOINTS,
    HOME_SETTLE_TOLERANCE_RAD,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    JOINT_MOVE_TIMEOUT_STEPS,
    JOINT_PROGRESS_EPSILON_RAD,
    JOINT_STALL_STEPS,
    LINEAR_FINAL_SETTLE_TOLERANCE_RAD,
    LINEAR_MOVE_TIMEOUT_STEPS,
    LINEAR_SETTLE_TOLERANCE_RAD,
    LINEAR_WAYPOINT_SPACING_M,
    MAX_JOINT_STEP_RAD,
    PLACE_CLEAR,
    PLACE_TOLERANCE_M,
    PLACE_WORLD,
    PREGRASP_REALIGN,
    PREGRASP_SETTLE_TOLERANCE_RAD,
    SAFETY_PROFILE,
    SAFETY_PROFILE_NAME,
    SAFETY_SETTINGS_PATH,
    SETTINGS,
    SETTINGS_PATH,
    TABLE_CLEARANCE_REQUIRED_M,
    TABLE_FOOTPRINT_CENTRE_XY_M,
    TABLE_FOOTPRINT_SIZE_XY_M,
    TABLE_TOP_HEIGHT_M,
    TIME_STEP,
    TOOL_OFFSET,
    URDF_PATH,
    RobotSettings,
)


def test_yaml_backed_settings_preserve_original_controller_defaults() -> None:
    assert TIME_STEP == 32
    assert CAN_DEF == "CAN"
    assert PLACE_WORLD == [-1.0, -4.3, 0.821]
    assert GRASP_Z == 0.02
    assert APPROACH_H == 0.12
    assert PLACE_CLEAR == 0.01
    assert IK_TOL == 0.015
    assert IK_TOL_GRASP == 0.005
    assert IK_TOL_TRANSIT == 0.015
    assert MAX_JOINT_STEP_RAD == 0.15
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
    assert SETTINGS.reset_mode == "soft"
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
    assert SETTINGS.joint_stall_steps == 10
    assert JOINT_STALL_STEPS == 10
    assert JOINT_PROGRESS_EPSILON_RAD == 1e-6
    assert PREGRASP_SETTLE_TOLERANCE_RAD == 0.01
    assert HOME_SETTLE_TOLERANCE_RAD == 0.02
    assert LINEAR_SETTLE_TOLERANCE_RAD == 0.03
    assert LINEAR_FINAL_SETTLE_TOLERANCE_RAD == 0.01
    assert LINEAR_WAYPOINT_SPACING_M == 0.03
    assert JOINT_MOVE_TIMEOUT_STEPS == 800
    assert LINEAR_MOVE_TIMEOUT_STEPS == 600
    assert SETTINGS.base_drift_warn_position_m == 0.005
    assert SETTINGS.base_drift_warn_yaw_deg == 0.5
    assert SETTINGS.base_drift_fail_position_m == 0.015
    assert SETTINGS.base_drift_fail_yaw_deg == 0.5
    assert SETTINGS.base_drift_rate_limit_m_per_s == 0.010
    assert SETTINGS.base_drift_rate_window_s == 0.5
    assert SETTINGS.hold_base == "none"
    assert SETTINGS.pregrasp_route == "interpolated"
    assert SETTINGS.pregrasp_realign is True
    assert PREGRASP_REALIGN is True
    assert SETTINGS.close_align_tol == 0.008
    assert CLOSE_ALIGN_TOL == 0.008
    assert SETTINGS.place_tolerance_m == 0.10
    assert PLACE_TOLERANCE_M == 0.10
    assert SETTINGS.can_place_z_tolerance_m == 0.02
    assert CAN_PLACE_Z_TOLERANCE_M == 0.02
    assert SETTINGS.can_tilt_limit_deg == 15
    assert CAN_TILT_LIMIT_DEG == 15
    assert TABLE_TOP_HEIGHT_M == 0.75
    assert TABLE_FOOTPRINT_CENTRE_XY_M == (-0.23882837, -4.8402928)
    assert TABLE_FOOTPRINT_SIZE_XY_M == (4.5, 1.5)
    assert TABLE_CLEARANCE_REQUIRED_M == 0.05
    assert FK_GROUND_TRUTH_TOLERANCE_M == 0.01
    assert SETTINGS.closed_loop_waypoints is True
    assert CLOSED_LOOP_WAYPOINTS is True
    assert SETTINGS.finger_tip_offset_m == pytest.approx(0.1741)
    assert FINGER_TIP_OFFSET_M == pytest.approx(0.1741)
    assert SETTINGS.can_radius_m == pytest.approx(0.03175)
    assert CAN_RADIUS_M == pytest.approx(0.03175)
    assert SAFETY_SETTINGS_PATH == Path(__file__).parents[1] / "configs" / "safety.yaml"
    assert SAFETY_PROFILE_NAME == "original_envelope"
    assert SAFETY_PROFILE.workspace_min == (
        0.10367229767278827,
        -1.0854553087033255,
        0.652413946369596,
    )
    assert SAFETY_PROFILE.workspace_max == (
        0.7844717,
        -0.12312418,
        1.09241395,
    )
    assert SAFETY_PROFILE.margin_m == 0.1


def test_robot_settings_reject_missing_and_unknown_yaml_keys(tmp_path: Path) -> None:
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text("time_step: 32\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid robot settings keys"):
        RobotSettings.from_yaml(bad_settings)


def test_robot_settings_reject_unknown_reset_mode(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "reset_mode: soft", "reset_mode: restart_everything"
    )
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="reset_mode must be either"):
        RobotSettings.from_yaml(bad_settings)


def test_robot_settings_accept_reload_as_a_deferred_lifecycle_option(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "reset_mode: soft", "reset_mode: reload"
    )
    reload_settings = tmp_path / "robot.yaml"
    reload_settings.write_text(config, encoding="utf-8")

    assert RobotSettings.from_yaml(reload_settings).reset_mode == "reload"


def test_robot_settings_reject_non_boolean_pregrasp_realign(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "pregrasp_realign: true", "pregrasp_realign: 1"
    )
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="pregrasp_realign must be a boolean"):
        RobotSettings.from_yaml(bad_settings)


def test_robot_settings_reject_non_boolean_closed_loop_waypoints(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "closed_loop_waypoints: true", "closed_loop_waypoints: 0"
    )
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="closed_loop_waypoints must be a boolean"):
        RobotSettings.from_yaml(bad_settings)


@pytest.mark.parametrize(
    ("key", "value"), (("finger_tip_offset_m", "0"), ("can_radius_m", "-0.03175"))
)
def test_robot_settings_reject_non_positive_tool_geometry(
    tmp_path: Path, key: str, value: str
) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8")
    needle = f"{key}: "
    start = config.index(needle)
    end = config.index("\n", start)
    config = config[:start] + f"{key}: {value}" + config[end:]
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match=f"{key} must be positive"):
        RobotSettings.from_yaml(bad_settings)


def test_robot_settings_reject_non_positive_close_align_tol(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "close_align_tol: 0.008", "close_align_tol: 0.0"
    )
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="close_align_tol must be positive"):
        RobotSettings.from_yaml(bad_settings)


def test_robot_settings_reject_unknown_hold_base(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "hold_base: none", "hold_base: magnetic"
    )
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="hold_base must be"):
        RobotSettings.from_yaml(bad_settings)


def test_robot_settings_require_fail_tier_above_warn_tier(tmp_path: Path) -> None:
    config = SETTINGS_PATH.read_text(encoding="utf-8").replace(
        "base_drift_fail_position_m: 0.015", "base_drift_fail_position_m: 0.004"
    )
    bad_settings = tmp_path / "robot.yaml"
    bad_settings.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="must be greater than"):
        RobotSettings.from_yaml(bad_settings)
