"""Typed YAML-backed controller settings and legacy-compatible constants."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml


@dataclass(frozen=True)
class WorkspaceProfile:
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    margin_m: float


@dataclass(frozen=True)
class SafetySettings:
    profiles: dict[str, WorkspaceProfile]
    PROFILE_NAMES: ClassVar[frozenset[str]] = frozenset({"original_envelope", "strict"})

    @classmethod
    def from_yaml(cls, path: Path) -> SafetySettings:
        with path.open("r", encoding="utf-8") as file:
            values: Any = yaml.safe_load(file)
        if not isinstance(values, dict) or set(values) != {"profiles"}:
            raise ValueError(f"Safety settings must contain only a profiles mapping: {path}")
        profiles = values["profiles"]
        if not isinstance(profiles, dict) or set(profiles) != cls.PROFILE_NAMES:
            raise ValueError(
                f"Safety settings must define profiles {sorted(cls.PROFILE_NAMES)}: {path}"
            )
        parsed_profiles: dict[str, WorkspaceProfile] = {}
        for name, profile in profiles.items():
            if (
                not isinstance(profile, dict)
                or set(profile) != {"workspace_min", "workspace_max", "margin_m"}
                or not isinstance(profile["workspace_min"], list)
                or not isinstance(profile["workspace_max"], list)
                or len(profile["workspace_min"]) != 3
                or len(profile["workspace_max"]) != 3
                or not isinstance(profile["margin_m"], (int, float))
                or not math.isfinite(profile["margin_m"])
                or profile["margin_m"] < 0.0
            ):
                raise ValueError(f"Invalid workspace profile {name!r}: {path}")
            parsed_profiles[name] = WorkspaceProfile(
                workspace_min=tuple(profile["workspace_min"]),
                workspace_max=tuple(profile["workspace_max"]),
                margin_m=float(profile["margin_m"]),
            )
        return cls(profiles=parsed_profiles)


@dataclass(frozen=True)
class RobotSettings:
    time_step: int
    can_def: str
    reset_mode: Literal["soft", "reload"]
    place_world: tuple[float, float, float]
    grasp_z: float
    approach_h: float
    place_clear: float
    ik_tol: float
    ik_tol_grasp: float
    ik_tol_transit: float
    max_joint_step_rad: float
    align_tol: float
    close_align_tol: float
    skip_align_check: bool
    max_tilt_deg: float
    place_tolerance_m: float
    can_place_z_tolerance_m: float
    can_tilt_limit_deg: float
    grip_force: float | None
    min_grasp_open: float
    lift_min: float
    max_retries: int
    stop_after: str | None
    torso_height: float | None
    finger_speed: float
    gripper_open_timeout_s: float
    calibrate_tool: bool
    fingers_inverted: bool
    gripper_test: bool
    head_pan: float
    head_tilt: float
    arm_joints: tuple[str, ...]
    finger_joints: tuple[str, ...]
    torso_joint: str
    head_joints: tuple[str, ...]
    tool_offset: tuple[float, float, float]
    gripper_opening_limits: tuple[float, float]
    gripper_force_max: float
    safety_command_horizon: float
    joint_stall_steps: int
    joint_progress_epsilon_rad: float
    pregrasp_settle_tolerance_rad: float
    home_settle_tolerance_rad: float
    linear_settle_tolerance_rad: float
    linear_final_settle_tolerance_rad: float
    linear_waypoint_spacing_m: float
    joint_move_timeout_steps: int
    linear_move_timeout_steps: int
    base_drift_warn_position_m: float
    base_drift_warn_yaw_deg: float
    base_drift_fail_position_m: float
    base_drift_fail_yaw_deg: float
    base_drift_rate_limit_m_per_s: float
    base_drift_rate_window_s: float
    hold_base: Literal["none", "wheels_zero_velocity"]
    reset_torso_settle_tolerance_m: float
    reset_torso_timeout_steps: int
    reset_head_settle_ms: int
    pregrasp_route: Literal["interpolated", "safe_via"]
    pregrasp_realign: bool
    closed_loop_waypoints: bool
    finger_tip_offset_m: float
    can_radius_m: float
    table_top_height_m: float
    table_footprint_centre_xy_m: tuple[float, float]
    table_footprint_size_xy_m: tuple[float, float]
    table_clearance_required_m: float
    fk_ground_truth_tolerance_m: float
    urdf_filename: str
    export_sim_kinematics: bool
    kinematic_model: Literal["urdf", "sim"]
    sim_kinematics_filename: str
    observe_images: bool
    probe_stationary_delta_rad: float
    probe_stationarity_steps: int
    probe_max_steps_per_joint: int
    probe_total_step_cap: int
    zmq_ping_timeout_ms: int
    zmq_reset_timeout_ms: int
    zmq_step_timeout_ms: int
    zmq_apply_timeout_ms: int
    zmq_get_gt_timeout_ms: int
    zmq_shutdown_timeout_ms: int
    zmq_export_poll_timeout_ms: int
    zmq_port: int
    grasp_servo: Literal["none", "gt_pad"]
    allow_unverified_tool: bool
    debug_markers: bool
    reset_post_settle_enabled: bool
    reset_post_settle_velocity_threshold_m_s: float
    reset_post_settle_ang_velocity_threshold_rad_s: float
    reset_post_settle_tilt_delta_deg: float
    reset_post_settle_consecutive_steps: int
    reset_post_settle_max_steps: int

    @classmethod
    def from_yaml(cls, path: Path) -> RobotSettings:
        with path.open("r", encoding="utf-8") as file:
            values: Any = yaml.safe_load(file)
        if not isinstance(values, dict):
            raise ValueError(f"Robot settings must be a YAML mapping: {path}")

        expected = {item.name for item in fields(cls)}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Invalid robot settings keys; missing={missing}, extra={extra}")
        if values.get("reset_mode") not in {"soft", "reload"}:
            raise ValueError("reset_mode must be either 'soft' or 'reload'.")
        if not isinstance(values.get("joint_stall_steps"), int) or values["joint_stall_steps"] <= 0:
            raise ValueError("joint_stall_steps must be a positive integer.")
        for key in (
            "joint_move_timeout_steps",
            "linear_move_timeout_steps",
            "reset_torso_timeout_steps",
            "reset_head_settle_ms",
        ):
            if not isinstance(values.get(key), int) or values[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        for key in (
            "joint_progress_epsilon_rad",
            "pregrasp_settle_tolerance_rad",
            "home_settle_tolerance_rad",
            "linear_settle_tolerance_rad",
            "linear_final_settle_tolerance_rad",
            "linear_waypoint_spacing_m",
            "reset_torso_settle_tolerance_m",
            "gripper_open_timeout_s",
        ):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")

        if values.get("pregrasp_route") not in {"interpolated", "safe_via"}:
            raise ValueError("pregrasp_route must be 'interpolated' or 'safe_via'.")
        if not isinstance(values.get("pregrasp_realign"), bool):
            raise ValueError("pregrasp_realign must be a boolean.")
        if not isinstance(values.get("closed_loop_waypoints"), bool):
            raise ValueError("closed_loop_waypoints must be a boolean.")
        for key in ("finger_tip_offset_m", "can_radius_m"):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")
        for key in (
            "close_align_tol",
            "place_tolerance_m",
            "can_place_z_tolerance_m",
            "can_tilt_limit_deg",
        ):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")
        for key in (
            "base_drift_warn_position_m",
            "base_drift_warn_yaw_deg",
            "base_drift_fail_position_m",
            "base_drift_fail_yaw_deg",
            "base_drift_rate_limit_m_per_s",
            "base_drift_rate_window_s",
        ):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")
        if values["base_drift_fail_position_m"] <= values["base_drift_warn_position_m"]:
            raise ValueError(
                "base_drift_fail_position_m must be greater than base_drift_warn_position_m."
            )
        if values["base_drift_fail_yaw_deg"] < values["base_drift_warn_yaw_deg"]:
            raise ValueError(
                "base_drift_fail_yaw_deg must be greater than or equal to base_drift_warn_yaw_deg."
            )
        if values.get("hold_base") not in {"none", "wheels_zero_velocity"}:
            raise ValueError("hold_base must be 'none' or 'wheels_zero_velocity'.")
        if not isinstance(values.get("export_sim_kinematics"), bool):
            raise ValueError("export_sim_kinematics must be a boolean.")
        if values.get("kinematic_model") not in {"urdf", "sim"}:
            raise ValueError("kinematic_model must be 'urdf' or 'sim'.")
        if (
            not isinstance(values.get("sim_kinematics_filename"), str)
            or not values["sim_kinematics_filename"]
        ):
            raise ValueError("sim_kinematics_filename must be a non-empty string.")
        if not isinstance(values.get("observe_images"), bool):
            raise ValueError("observe_images must be a boolean.")
        if (
            not isinstance(values.get("probe_stationary_delta_rad"), (int, float))
            or values["probe_stationary_delta_rad"] <= 0.0
        ):
            raise ValueError("probe_stationary_delta_rad must be positive.")
        for key in (
            "probe_stationarity_steps",
            "probe_max_steps_per_joint",
            "probe_total_step_cap",
            "zmq_ping_timeout_ms",
            "zmq_reset_timeout_ms",
            "zmq_step_timeout_ms",
            "zmq_apply_timeout_ms",
            "zmq_get_gt_timeout_ms",
            "zmq_shutdown_timeout_ms",
            "zmq_export_poll_timeout_ms",
        ):
            if not isinstance(values.get(key), int) or values[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if not isinstance(values.get("zmq_port"), int) or not (1 <= values["zmq_port"] <= 65535):
            raise ValueError("zmq_port must be an integer port in [1, 65535].")
        if values.get("grasp_servo") not in {"none", "gt_pad"}:
            raise ValueError("grasp_servo must be 'none' or 'gt_pad'.")
        for key in ("allow_unverified_tool", "debug_markers", "reset_post_settle_enabled"):
            if not isinstance(values.get(key), bool):
                raise ValueError(f"{key} must be a boolean.")
        for key in (
            "reset_post_settle_velocity_threshold_m_s",
            "reset_post_settle_ang_velocity_threshold_rad_s",
            "reset_post_settle_tilt_delta_deg",
        ):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")
        for key in ("reset_post_settle_consecutive_steps", "reset_post_settle_max_steps"):
            if not isinstance(values.get(key), int) or values[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        for key in (
            "table_top_height_m",
            "table_clearance_required_m",
            "fk_ground_truth_tolerance_m",
        ):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")
        for key, length in (("table_footprint_centre_xy_m", 2), ("table_footprint_size_xy_m", 2)):
            value = values.get(key)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != length
                or any(not isinstance(item, (int, float)) for item in value)
            ):
                raise ValueError(f"{key} must be a list of {length} numbers.")
        if any(size <= 0.0 for size in values["table_footprint_size_xy_m"]):
            raise ValueError("table_footprint_size_xy_m entries must be positive.")

        for key in (
            "place_world",
            "tool_offset",
            "gripper_opening_limits",
            "arm_joints",
            "finger_joints",
            "head_joints",
            "table_footprint_centre_xy_m",
            "table_footprint_size_xy_m",
        ):
            values[key] = tuple(values[key])
        return cls(**values)


_SOURCE_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "configs" / "robot.yaml"
_INSTALLED_SETTINGS_PATH = Path(sys.prefix) / "configs" / "robot.yaml"
SETTINGS_PATH = (
    _SOURCE_SETTINGS_PATH if _SOURCE_SETTINGS_PATH.is_file() else _INSTALLED_SETTINGS_PATH
)
if not SETTINGS_PATH.is_file():
    raise FileNotFoundError(f"Robot settings file not found: {SETTINGS_PATH}")
SETTINGS = RobotSettings.from_yaml(SETTINGS_PATH)
_SOURCE_SAFETY_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "configs" / "safety.yaml"
_INSTALLED_SAFETY_SETTINGS_PATH = Path(sys.prefix) / "configs" / "safety.yaml"
SAFETY_SETTINGS_PATH = (
    _SOURCE_SAFETY_SETTINGS_PATH
    if _SOURCE_SAFETY_SETTINGS_PATH.is_file()
    else _INSTALLED_SAFETY_SETTINGS_PATH
)
if not SAFETY_SETTINGS_PATH.is_file():
    raise FileNotFoundError(f"Safety settings file not found: {SAFETY_SETTINGS_PATH}")
SAFETY_SETTINGS = SafetySettings.from_yaml(SAFETY_SETTINGS_PATH)
SAFETY_PROFILE_NAME = "original_envelope"
SAFETY_PROFILE = SAFETY_SETTINGS.profiles[SAFETY_PROFILE_NAME]
CONTROLLER_DIR = Path(__file__).resolve().parent
URDF_PATH = CONTROLLER_DIR / SETTINGS.urdf_filename
SIM_KINEMATICS_PATH = SETTINGS_PATH.parent / SETTINGS.sim_kinematics_filename
EXPORT_SIM_KINEMATICS = SETTINGS.export_sim_kinematics
KINEMATIC_MODEL = SETTINGS.kinematic_model

# These names preserve the original controller module imports and values.
TIME_STEP = SETTINGS.time_step
CAN_DEF = SETTINGS.can_def
PLACE_WORLD = list(SETTINGS.place_world)
GRASP_Z = SETTINGS.grasp_z
APPROACH_H = SETTINGS.approach_h
PLACE_CLEAR = SETTINGS.place_clear
IK_TOL = SETTINGS.ik_tol
IK_TOL_GRASP = SETTINGS.ik_tol_grasp
IK_TOL_TRANSIT = SETTINGS.ik_tol_transit
MAX_JOINT_STEP_RAD = SETTINGS.max_joint_step_rad
ALIGN_TOL = SETTINGS.align_tol
CLOSE_ALIGN_TOL = SETTINGS.close_align_tol
SKIP_ALIGN_CHECK = SETTINGS.skip_align_check
MAX_TILT_DEG = SETTINGS.max_tilt_deg
PLACE_TOLERANCE_M = SETTINGS.place_tolerance_m
CAN_PLACE_Z_TOLERANCE_M = SETTINGS.can_place_z_tolerance_m
CAN_TILT_LIMIT_DEG = SETTINGS.can_tilt_limit_deg
GRIP_FORCE = SETTINGS.grip_force
MIN_GRASP_OPEN = SETTINGS.min_grasp_open
LIFT_MIN = SETTINGS.lift_min
MAX_RETRIES = SETTINGS.max_retries
STOP_AFTER = SETTINGS.stop_after
TORSO_HEIGHT = SETTINGS.torso_height
FINGER_SPEED = SETTINGS.finger_speed
GRIPPER_OPEN = SETTINGS.gripper_opening_limits[1]
GRIPPER_OPEN_TIMEOUT_S = SETTINGS.gripper_open_timeout_s
CALIBRATE_TOOL = SETTINGS.calibrate_tool
FINGERS_INVERTED = SETTINGS.fingers_inverted
GRIPPER_TEST = SETTINGS.gripper_test
HEAD_PAN = SETTINGS.head_pan
HEAD_TILT = SETTINGS.head_tilt
ARM_JOINTS = list(SETTINGS.arm_joints)
FINGER_JOINTS = list(SETTINGS.finger_joints)
TORSO_JOINT = SETTINGS.torso_joint
HEAD_JOINTS = list(SETTINGS.head_joints)
TOOL_OFFSET = list(SETTINGS.tool_offset)
JOINT_STALL_STEPS = SETTINGS.joint_stall_steps
JOINT_PROGRESS_EPSILON_RAD = SETTINGS.joint_progress_epsilon_rad
PREGRASP_SETTLE_TOLERANCE_RAD = SETTINGS.pregrasp_settle_tolerance_rad
HOME_SETTLE_TOLERANCE_RAD = SETTINGS.home_settle_tolerance_rad
LINEAR_SETTLE_TOLERANCE_RAD = SETTINGS.linear_settle_tolerance_rad
LINEAR_FINAL_SETTLE_TOLERANCE_RAD = SETTINGS.linear_final_settle_tolerance_rad
LINEAR_WAYPOINT_SPACING_M = SETTINGS.linear_waypoint_spacing_m
JOINT_MOVE_TIMEOUT_STEPS = SETTINGS.joint_move_timeout_steps
LINEAR_MOVE_TIMEOUT_STEPS = SETTINGS.linear_move_timeout_steps
BASE_DRIFT_WARN_POSITION_M = SETTINGS.base_drift_warn_position_m
BASE_DRIFT_WARN_YAW_DEG = SETTINGS.base_drift_warn_yaw_deg
BASE_DRIFT_FAIL_POSITION_M = SETTINGS.base_drift_fail_position_m
BASE_DRIFT_FAIL_YAW_DEG = SETTINGS.base_drift_fail_yaw_deg
BASE_DRIFT_RATE_LIMIT_M_PER_S = SETTINGS.base_drift_rate_limit_m_per_s
BASE_DRIFT_RATE_WINDOW_S = SETTINGS.base_drift_rate_window_s
HOLD_BASE = SETTINGS.hold_base
RESET_TORSO_SETTLE_TOLERANCE_M = SETTINGS.reset_torso_settle_tolerance_m
RESET_TORSO_TIMEOUT_STEPS = SETTINGS.reset_torso_timeout_steps
RESET_HEAD_SETTLE_MS = SETTINGS.reset_head_settle_ms
PREGRASP_ROUTE = SETTINGS.pregrasp_route
PREGRASP_REALIGN = SETTINGS.pregrasp_realign
CLOSED_LOOP_WAYPOINTS = SETTINGS.closed_loop_waypoints
FINGER_TIP_OFFSET_M = SETTINGS.finger_tip_offset_m
CAN_RADIUS_M = SETTINGS.can_radius_m
TABLE_TOP_HEIGHT_M = SETTINGS.table_top_height_m
TABLE_FOOTPRINT_CENTRE_XY_M = SETTINGS.table_footprint_centre_xy_m
TABLE_FOOTPRINT_SIZE_XY_M = SETTINGS.table_footprint_size_xy_m
TABLE_CLEARANCE_REQUIRED_M = SETTINGS.table_clearance_required_m
FK_GROUND_TRUTH_TOLERANCE_M = SETTINGS.fk_ground_truth_tolerance_m
OBSERVE_IMAGES = SETTINGS.observe_images
PROBE_STATIONARY_DELTA_RAD = SETTINGS.probe_stationary_delta_rad
PROBE_STATIONARITY_STEPS = SETTINGS.probe_stationarity_steps
PROBE_MAX_STEPS_PER_JOINT = SETTINGS.probe_max_steps_per_joint
PROBE_TOTAL_STEP_CAP = SETTINGS.probe_total_step_cap
ZMQ_PING_TIMEOUT_MS = SETTINGS.zmq_ping_timeout_ms
ZMQ_RESET_TIMEOUT_MS = SETTINGS.zmq_reset_timeout_ms
ZMQ_STEP_TIMEOUT_MS = SETTINGS.zmq_step_timeout_ms
ZMQ_APPLY_TIMEOUT_MS = SETTINGS.zmq_apply_timeout_ms
ZMQ_GET_GT_TIMEOUT_MS = SETTINGS.zmq_get_gt_timeout_ms
ZMQ_SHUTDOWN_TIMEOUT_MS = SETTINGS.zmq_shutdown_timeout_ms
ZMQ_EXPORT_POLL_TIMEOUT_MS = SETTINGS.zmq_export_poll_timeout_ms
ZMQ_PORT = SETTINGS.zmq_port
GRASP_SERVO = SETTINGS.grasp_servo
ALLOW_UNVERIFIED_TOOL = SETTINGS.allow_unverified_tool
DEBUG_MARKERS = SETTINGS.debug_markers
RESET_POST_SETTLE_ENABLED = SETTINGS.reset_post_settle_enabled
RESET_POST_SETTLE_VELOCITY_THRESHOLD_M_S = SETTINGS.reset_post_settle_velocity_threshold_m_s
RESET_POST_SETTLE_ANG_VELOCITY_THRESHOLD_RAD_S = (
    SETTINGS.reset_post_settle_ang_velocity_threshold_rad_s
)
RESET_POST_SETTLE_TILT_DELTA_DEG = SETTINGS.reset_post_settle_tilt_delta_deg
RESET_POST_SETTLE_CONSECUTIVE_STEPS = SETTINGS.reset_post_settle_consecutive_steps
RESET_POST_SETTLE_MAX_STEPS = SETTINGS.reset_post_settle_max_steps
