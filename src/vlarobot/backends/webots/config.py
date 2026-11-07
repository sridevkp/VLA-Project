"""Typed YAML-backed controller settings and legacy-compatible constants."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

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
    place_base: tuple[float, float, float]
    grasp_z: float
    approach_h: float
    place_clear: float
    ik_tol: float
    ik_tol_grasp: float
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

        for key in (
            "place_base",
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
PLACE_BASE = list(SETTINGS.place_base)
GRASP_Z = SETTINGS.grasp_z
APPROACH_H = SETTINGS.approach_h
PLACE_CLEAR = SETTINGS.place_clear
IK_TOL = SETTINGS.ik_tol
IK_TOL_GRASP = SETTINGS.ik_tol_grasp
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
