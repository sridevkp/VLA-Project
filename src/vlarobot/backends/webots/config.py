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
    skip_align_check: bool
    max_tilt_deg: float
    grip_force: float | None
    min_grasp_open: float
    lift_min: float
    max_retries: int
    stop_after: str | None
    torso_height: float | None
    finger_speed: float
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
    base_drift_position_tolerance_m: float
    base_drift_yaw_tolerance_deg: float
    pregrasp_route: Literal["interpolated", "safe_via"]
    urdf_filename: str

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
        ):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")

        if values.get("pregrasp_route") not in {"interpolated", "safe_via"}:
            raise ValueError("pregrasp_route must be 'interpolated' or 'safe_via'.")
        for key in ("base_drift_position_tolerance_m", "base_drift_yaw_tolerance_deg"):
            if not isinstance(values.get(key), (int, float)) or values[key] <= 0.0:
                raise ValueError(f"{key} must be positive.")

        for key in (
            "place_world",
            "tool_offset",
            "gripper_opening_limits",
            "arm_joints",
            "finger_joints",
            "head_joints",
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
SKIP_ALIGN_CHECK = SETTINGS.skip_align_check
MAX_TILT_DEG = SETTINGS.max_tilt_deg
GRIP_FORCE = SETTINGS.grip_force
MIN_GRASP_OPEN = SETTINGS.min_grasp_open
LIFT_MIN = SETTINGS.lift_min
MAX_RETRIES = SETTINGS.max_retries
STOP_AFTER = SETTINGS.stop_after
TORSO_HEIGHT = SETTINGS.torso_height
FINGER_SPEED = SETTINGS.finger_speed
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
BASE_DRIFT_POSITION_TOLERANCE_M = SETTINGS.base_drift_position_tolerance_m
BASE_DRIFT_YAW_TOLERANCE_DEG = SETTINGS.base_drift_yaw_tolerance_deg
PREGRASP_ROUTE = SETTINGS.pregrasp_route
