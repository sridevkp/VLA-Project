"""Concrete, inert backend registration for the production WorldMind composition.

This module is the only place that imports the existing Webots-backed geometry and safety
objects for WorldMind. Registering them does not launch Webots, solve IK, run inference, train,
actuate or access the TEST split. Geometry is constructed lazily when a reviewed config actually
calls a factory.

Eq. 3.8 component formulas, weights, ranking direction and calibrated risk thresholds are
deliberately absent: no authoritative source supplies them. Their registry stays empty so
``validate_worldmind_bindings`` fails closed until owner-reviewed values are supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from vlarobot.backends.webots.config import (
    APPROACH_H,
    ARM_JOINTS,
    GRASP_Z,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    PLACE_CLEAR,
    SAFETY_PROFILE,
    SETTINGS,
    SETTINGS_PATH,
    URDF_PATH,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.pose_solver import BoundedPoseSolver
from vlarobot.backends.webots.proto_sources import load_proto_source_manifest
from vlarobot.backends.webots.urdf_registration import load_registration
from vlarobot.core.types import (
    GripperCmd,
    JointTarget,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_bindings import (
    ANALYTIC_SURROGATE_BINDING,
    CORRECTED_POSE_SOLVER_BINDING,
    CORRECTED_WAYPOINTS_BINDING,
    FINITE_CHUNK_CHECK,
    PLACEMENT_VERIFIER_BINDING,
    RUNTIME_SAFETY_BINDING,
)
from vlarobot.eval.worldmind_config import (
    WorldMindBindings,
    WorldMindConfig,
    WorldMindDiagnosticConfig,
)
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.planning.verification import PlacementVerifier
from vlarobot.vla.action_adapter import PoseSolution
from vlarobot.world_model.arm_geometry import Capsule, StaticSceneGeometry
from vlarobot.world_model.arm_sweep import (
    ArmSweepContract,
    IntermediateSweptArmCheck,
    SweepGeometryError,
    urdf_collision_capsules,
)
from vlarobot.world_model.kinematic import MemoizedForward, VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate, compose, rotate
from vlarobot.world_model.scene_geometry import load_scene_collision_catalog

CORRECTED_REGISTRATION_RELATIVE = Path("calibration") / "urdf-base-train-v1.yaml"
CORRECTED_REGISTRATION_SHA256 = "dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0"
ROBOT_CONFIG_SHA256 = "1a9f0c2b57d771e45b6908d05177eb64eb80634b7ab2d35c5dfbefb1f863ca1d"
ARM_SWEEP_CONFIG_PATH = (SETTINGS_PATH.parent / "worldmind_arm_sweep.yaml").resolve()
ARM_SWEEP_SCHEMA_VERSION = 1
ARM_SWEEP_DT_SOURCE = "robot.yaml:time_step_ms/1000"
ARM_SWEEP_SPEED_SOURCE = "urdf_joint_velocity_reach_bound_v1"

SURROGATE_INPUTS = ("forward", "inverse", "validate", "tracking")
SAFETY_INPUTS = ("monitor", "limits")
VERIFICATION_INPUTS: tuple[str, ...] = ()

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
# Ordered to mirror the frozen baseline's station sequence: INIT(open) -> PREGRASP -> DESCEND
# -> CLOSE -> LIFT -> TRANSIT -> PLACE_DESCEND -> RELEASE -> RETREAT -> HOME. These hold steps
# are an explicit scripted cadence, NOT a scientific threshold or a claim of settle-equivalence;
# the frozen baseline waits for measured joint settling and gripper stall. The sum is H=50.
_CORRECTED_HOLD_STEPS: Mapping[str, int] = MappingProxyType(
    {
        "INIT": 2,
        "PREGRASP": 10,
        "DESCEND": 6,
        "CLOSE": 3,
        "LIFT": 8,
        "TRANSIT": 8,
        "PLACE_DESCEND": 6,
        "RELEASE": 2,
        "RETREAT": 2,
        "HOME": 3,
    }
)


def corrected_registration_path() -> Path:
    """Return the frozen, hash-pinned corrected registration path."""
    path = (SETTINGS_PATH.parent / CORRECTED_REGISTRATION_RELATIVE).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CORRECTED_REGISTRATION_SHA256:
        raise ValueError("corrected URDF registration bytes differ from the frozen provenance")
    load_registration(path)
    return path


@lru_cache(maxsize=1)
def corrected_kinematics() -> Kinematics:
    """Build the existing URDF chain with the corrected registration; no IK solve is performed."""
    corrected_registration_path()
    previous = os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE")
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = CORRECTED_REGISTRATION_RELATIVE.as_posix()
    try:
        kinematics = Kinematics("urdf")
    finally:
        if previous is None:
            os.environ.pop("VLAROBOT_URDF_REGISTRATION_FILE", None)
        else:
            os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = previous
    if kinematics.registration_sha256 != CORRECTED_REGISTRATION_SHA256:
        raise ValueError("constructed kinematics did not use the corrected registration")
    return kinematics


@lru_cache(maxsize=1)
def _urdf_velocities() -> dict[str, float]:
    tree = ET.parse(URDF_PATH)
    velocities: dict[str, float] = {}
    for name in ARM_JOINTS:
        limit = tree.find(f'./joint[@name="{name}"]/limit')
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"URDF velocity limit unavailable for {name}")
        velocities[name] = float(limit.attrib["velocity"])
    if not all(math.isfinite(value) and value > 0 for value in velocities.values()):
        raise ValueError("URDF velocity limits must be finite and positive")
    return velocities


def runtime_safety_limits() -> SafetyLimits:
    """Use existing effective robot/URDF/safety settings, never default-only limits."""
    kin = corrected_kinematics()
    return SafetyLimits(
        joint_position={name: kin.bounds(name) for name in ARM_JOINTS},
        joint_velocity=dict(_urdf_velocities()),
        workspace_min=SAFETY_PROFILE.workspace_min,
        workspace_max=SAFETY_PROFILE.workspace_max,
        gripper_opening=SETTINGS.gripper_opening_limits,
        gripper_force_max=SETTINGS.gripper_force_max,
        command_horizon=SETTINGS.safety_command_horizon,
    )


def runtime_safety_factory(config: WorldMindConfig, inputs: Mapping[str, Any]) -> SafetyMonitor:
    """Bind the existing live SafetyMonitor to the reviewed effective limits."""
    del config, inputs
    return SafetyMonitor(runtime_safety_limits())


class _LazyCorrectedPoseSolver:
    """Defer geometry construction until a reviewed solve is actually requested."""

    def __init__(self) -> None:
        self._inner: BoundedPoseSolver | None = None

    @property
    def inner(self) -> BoundedPoseSolver:
        if self._inner is None:
            kin = corrected_kinematics()
            self._inner = BoundedPoseSolver(kin, max_joint_step_rad=SETTINGS.max_joint_step_rad)
        return self._inner

    def solve(self, pose: Pose3D, state: RobotState) -> PoseSolution:
        return self.inner.solve(pose, state)


class WorldStatePoseSolver:
    """Adapt the existing BoundedPoseSolver to the WorldState adapter protocol."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def solve(self, pose: Pose3D, state: WorldState) -> PoseSolution:
        if not isinstance(state, WorldState):
            raise TypeError("corrected solver requires a WorldState")
        return self.inner.solve(pose, state.robot)


def corrected_pose_solver() -> WorldStatePoseSolver:
    """Return the existing bounded IK solver with corrected URDF registration."""
    return WorldStatePoseSolver(_LazyCorrectedPoseSolver())


def _rotation_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a finite proper rotation matrix to an explicit xyzw quaternion."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("finite 3x3 rotation required for quaternion conversion")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation matrix must be orthonormal")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x, y, z = (
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            )
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x, y, z = (
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x, y, z = (
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            )
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))


def _full_joint_vector(joints: Mapping[str, float]) -> np.ndarray:
    kin = corrected_kinematics()
    vector = np.zeros(len(kin.chain.links), dtype=float)
    for name, value in joints.items():
        if name in kin.idx:
            vector[kin.idx[name]] = float(value)
    return vector


def _corrected_forward(joints: Mapping[str, float]) -> Pose3D:
    matrix = corrected_kinematics().forward(_full_joint_vector(joints))
    return Pose3D(Vec3(*map(float, matrix[:3, 3])), _rotation_to_quaternion(matrix[:3, :3]))


@dataclass(frozen=True)
class ArmSweepCertificate:
    """Owner evidence identifiers; presence is checked, never self-issued by this code."""

    certificate_id: str
    sha256: str
    approver: str
    approval_id: str
    approval_record: Path
    approval_record_sha256: str
    reviewed_cases: Path
    reviewed_cases_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class ArmSweepConfig:
    """Strict, pinned full-arm sweep contract with no implicit safety parameter."""

    source_path: Path
    status: str
    urdf_path: Path
    registration_path: Path
    robot_config_path: Path
    world_path: Path
    urdf_sha256: str
    registration_sha256: str
    robot_config_sha256: str
    world_sha256: str
    required_arm_links: tuple[str, ...]
    required_objects: frozenset[str]
    adjacent_pairs: frozenset[frozenset[str]]
    permitted_contact_objects: frozenset[str]
    dt_s: float
    margin_m: float
    substeps: int
    speed_bound_m_s: float
    certificate: ArmSweepCertificate | None
    scene_catalog_path: Path | None = None
    scene_catalog_sha256: str | None = None
    proto_manifest_path: Path | None = None
    proto_manifest_sha256: str | None = None
    adaptive_speed_bound_m_s: float | None = None
    max_adaptive_substeps: int = 4096
    max_joint_displacement_bound_rad: float | None = None
    joint_reach_bound_m: float | None = None
    geometry_error_bound_m: float | None = None
    registration_uncertainty_bound_m: float | None = None
    numerical_error_bound_m: float | None = None
    transform_error_bound_m: float | None = None


_SWEEP_TOP_KEYS = frozenset({"schema_version", "status", "geometry", "contract", "certificate"})
_SWEEP_FORBIDDEN_KEYS = frozenset(
    {
        "train",
        "test",
        "dataset",
        "scoring",
        "score",
        "calibration",
        "components",
        "weights",
        "evaluator",
        "motion_allowed",
        "apply",
        "step",
        "reset",
    }
)
_SWEEP_GEOMETRY_KEYS = frozenset(
    {
        "urdf",
        "registration",
        "robot_config",
        "world",
        "urdf_sha256",
        "registration_sha256",
        "robot_config_sha256",
        "world_sha256",
        "required_arm_links",
    }
)
_SWEEP_CONTRACT_KEYS = frozenset(
    {
        "margin_m",
        "substeps",
        "dt_source",
        "speed_source",
        "required_objects",
        "adjacent_pairs",
        "permitted_contact_objects",
    }
)
_SWEEP_OPTIONAL_GEOMETRY_KEYS = frozenset(
    {
        "scene_catalog",
        "scene_catalog_sha256",
        "proto_manifest",
        "proto_manifest_sha256",
    }
)
_SWEEP_OPTIONAL_CONTRACT_KEYS = frozenset(
    {
        "adaptive_speed_bound_m_s",
        "max_adaptive_substeps",
        "max_joint_displacement_bound_rad",
        "joint_reach_bound_m",
        "geometry_error_bound_m",
        "registration_uncertainty_bound_m",
        "numerical_error_bound_m",
        "transform_error_bound_m",
    }
)
_SWEEP_CERTIFICATE_KEYS = frozenset(
    {
        "id",
        "sha256",
        "approved",
        "approver",
        "approval_id",
        "approval_record",
        "approval_record_sha256",
        "reviewed_cases",
        "reviewed_cases_sha256",
        "contract_sha256",
    }
)


def _sweep_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SweepGeometryError(f"{name} must be a mapping")
    return value


def _sweep_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    allowed = expected | optional
    if not expected.issubset(value) or not set(value).issubset(allowed):
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - allowed)
        raise SweepGeometryError(f"{name} has missing={missing} and unexpected={unexpected}")


def _sweep_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SweepGeometryError(f"{name} must be a nonempty string")
    return value.strip()


def _sweep_sha(value: object, name: str) -> str:
    text = _sweep_text(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SweepGeometryError(f"{name} must be a 64-character SHA-256 hex digest")
    return text


def _sweep_path(value: object, name: str, source: Path) -> Path:
    candidate = Path(_sweep_text(value, name)).expanduser()
    return (
        (source.parent / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sweep_optional_sha(value: object, name: str) -> str | None:
    return None if value is None else _sweep_sha(value, name)


def _sweep_optional_path(value: object, name: str, source: Path) -> Path | None:
    return None if value is None else _sweep_path(value, name, source)


def _sweep_optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepGeometryError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SweepGeometryError(f"{name} must be a finite positive number")
    return number


def _sweep_optional_nonnegative_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepGeometryError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise SweepGeometryError(f"{name} must be a finite nonnegative number")
    return number


def _sweep_optional_int(value: object, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SweepGeometryError(f"{name} must be a positive integer")
    return value


def _sweep_string_set(value: object, name: str, *, required: bool = False) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SweepGeometryError(f"{name} must be a list of strings")
    result = frozenset(_sweep_text(item, name) for item in value)
    if len(result) != len(list(value)) or (required and not result):
        raise SweepGeometryError(
            f"{name} must contain unique strings" + (" and be nonempty" if required else "")
        )
    return result


def _urdf_joint_path(root: ET.Element, target: str) -> tuple[tuple[ET.Element, float], ...]:
    children: dict[str, list[str]] = {}
    joints: dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise SweepGeometryError("URDF joint has incomplete parent/child links")
        parent_name, child_name = parent.attrib["link"], child.attrib["link"]
        children.setdefault(parent_name, []).append(child_name)
        joints[child_name] = joint
    stack: list[tuple[str, tuple[tuple[ET.Element, float], ...]]] = [("base_link", ())]
    visited: set[str] = set()
    while stack:
        name, path = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        if name == target:
            return path
        for child in children.get(name, []):
            joint = joints[child]
            origin = joint.find("origin")
            xyz = (
                origin.attrib.get("xyz", "0 0 0").split() if origin is not None else ["0", "0", "0"]
            )
            if len(xyz) != 3:
                raise SweepGeometryError(f"invalid URDF joint origin for {child}")
            coordinates = np.asarray([float(value) for value in xyz], dtype=np.float64)
            if not np.isfinite(coordinates).all():
                raise SweepGeometryError(f"nonfinite URDF joint origin for {child}")
            stack.append((child, (*path, (joint, float(np.linalg.norm(coordinates))))))
    raise SweepGeometryError(f"no URDF joint path from base_link to {target}")


def _urdf_reach_speed_bound(urdf_path: Path, links: Sequence[str]) -> float:
    """Conservative speed bound from pinned URDF joint limits and full descendant reach."""
    root = ET.parse(urdf_path).getroot()
    provisional = urdf_collision_capsules(urdf_path, links, max_point_speed_m_s=1.0)
    reach = 0.0
    paths: list[tuple[tuple[ET.Element, float], ...]] = []
    for link in links:
        path = _urdf_joint_path(root, link)
        paths.append(path)
        chain_reach = sum(distance for _, distance in path)
        for primitive in provisional[link]:
            capsule = primitive.capsule
            extent = max(
                float(np.linalg.norm(capsule.start)) + capsule.radius_m,
                float(np.linalg.norm(capsule.end)) + capsule.radius_m,
            )
            reach = max(reach, chain_reach + extent)
    if not math.isfinite(reach) or reach <= 0.0:
        raise SweepGeometryError("URDF collision reach could not be established")

    bound = 0.0
    seen: set[str] = set()
    for path in paths:
        for joint, _ in path:
            name = joint.attrib.get("name", "")
            if name in seen:
                continue
            seen.add(name)
            kind = joint.attrib.get("type")
            limit = joint.find("limit")
            if kind == "fixed":
                continue
            if limit is None or "velocity" not in limit.attrib:
                raise SweepGeometryError(f"URDF velocity limit unavailable for {name}")
            velocity = float(limit.attrib["velocity"])
            if not math.isfinite(velocity) or velocity <= 0.0:
                raise SweepGeometryError(f"URDF velocity limit must be positive for {name}")
            if kind in {"revolute", "continuous"}:
                bound += velocity * reach
            elif kind == "prismatic":
                bound += velocity
            else:
                raise SweepGeometryError(f"unsupported URDF joint type for sweep: {kind}")
    if not math.isfinite(bound) or bound <= 0.0:
        raise SweepGeometryError("URDF point-speed bound could not be established")
    return bound


def _sweep_contract_sha256(config: ArmSweepConfig) -> str:
    payload = {
        "schema_version": ARM_SWEEP_SCHEMA_VERSION,
        "geometry": {
            "urdf_sha256": config.urdf_sha256,
            "registration_sha256": config.registration_sha256,
            "robot_config_sha256": config.robot_config_sha256,
            "world_sha256": config.world_sha256,
            "required_arm_links": list(config.required_arm_links),
        },
        "contract": {
            "dt_s": config.dt_s,
            "margin_m": config.margin_m,
            "substeps": config.substeps,
            "speed_bound_m_s": config.speed_bound_m_s,
            "required_objects": sorted(config.required_objects),
            "adjacent_pairs": sorted(sorted(pair) for pair in config.adjacent_pairs),
            "permitted_contact_objects": sorted(config.permitted_contact_objects),
        },
    }
    geometry_payload = payload["geometry"]
    contract_payload = payload["contract"]
    assert isinstance(geometry_payload, dict) and isinstance(contract_payload, dict)
    if config.scene_catalog_sha256 is not None:
        geometry_payload["scene_catalog_sha256"] = config.scene_catalog_sha256
    if config.proto_manifest_sha256 is not None:
        geometry_payload["proto_manifest_sha256"] = config.proto_manifest_sha256
    if config.adaptive_speed_bound_m_s is not None:
        contract_payload["adaptive_speed_bound_m_s"] = config.adaptive_speed_bound_m_s
    contract_payload["max_adaptive_substeps"] = config.max_adaptive_substeps
    for name in (
        "max_joint_displacement_bound_rad",
        "joint_reach_bound_m",
        "geometry_error_bound_m",
        "registration_uncertainty_bound_m",
        "numerical_error_bound_m",
        "transform_error_bound_m",
    ):
        value = getattr(config, name)
        if value is not None:
            contract_payload[name] = value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json_mapping(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepGeometryError(f"unreadable {name}: {error}") from error
    return _sweep_mapping(value, name)


def _load_sweep_certificate(
    raw: Mapping[str, object],
    source: Path,
    expected_contract_sha256: str,
    geometry: Mapping[str, str],
) -> ArmSweepCertificate | None:
    _sweep_keys(raw, _SWEEP_CERTIFICATE_KEYS, "certificate")
    if all(value in (None, False) for value in raw.values()):
        return None
    if raw["approved"] is not True:
        raise SweepGeometryError("swept-arm certificate is not owner-approved")
    certificate_id = _sweep_text(raw["id"], "certificate.id")
    digest = _sweep_sha(raw["sha256"], "certificate.sha256")
    approver = _sweep_text(raw["approver"], "certificate.approver")
    approval_id = _sweep_text(raw["approval_id"], "certificate.approval_id")
    record_path = _sweep_path(raw["approval_record"], "certificate.approval_record", source)
    record_sha = _sweep_sha(raw["approval_record_sha256"], "certificate.approval_record_sha256")
    cases_path = _sweep_path(raw["reviewed_cases"], "certificate.reviewed_cases", source)
    cases_sha = _sweep_sha(raw["reviewed_cases_sha256"], "certificate.reviewed_cases_sha256")
    contract_sha = _sweep_sha(raw["contract_sha256"], "certificate.contract_sha256")
    for path, expected in ((record_path, record_sha), (cases_path, cases_sha)):
        if not path.is_file() or _sha256_file(path) != expected:
            raise SweepGeometryError(f"review evidence hash mismatch or missing file: {path}")

    certificate = ArmSweepCertificate(
        certificate_id,
        digest,
        approver,
        approval_id,
        record_path,
        record_sha,
        cases_path,
        cases_sha,
        contract_sha,
    )
    certificate_path = source.parent / f"{certificate_id}.json"
    if not certificate_path.is_file() or _sha256_file(certificate_path) != digest:
        raise SweepGeometryError("owner certificate file/hash is unavailable")
    payload = _load_json_mapping(certificate_path, "swept-arm certificate")
    if (
        payload.get("certificate_id") != certificate_id
        or payload.get("approved") is not True
        or payload.get("contract_sha256") != contract_sha
        or payload.get("approval_id") != approval_id
        or payload.get("approver") != approver
    ):
        raise SweepGeometryError("owner certificate content does not match the reviewed contract")
    if payload.get("geometry_sha256") != dict(geometry):
        raise SweepGeometryError(
            "owner certificate geometry hashes do not match the pinned contract"
        )

    record = _load_json_mapping(record_path, "approval record")
    if (
        record.get("approved") is not True
        or record.get("role") != "independent_safety_owner"
        or record.get("approval_id") != approval_id
        or record.get("approver") != approver
        or record.get("certificate_id") != certificate_id
        or record.get("certificate_sha256") != digest
    ):
        raise SweepGeometryError("approval record does not match the certificate identity")
    cases = _load_json_mapping(cases_path, "reviewed reference cases")
    if (
        cases.get("schema_version") != ARM_SWEEP_SCHEMA_VERSION
        or cases.get("certificate_id") != certificate_id
        or not isinstance(cases.get("cases"), Sequence)
        or isinstance(cases.get("cases"), (str, bytes))
        or not cases["cases"]
    ):
        raise SweepGeometryError("reviewed reference cases are missing or malformed")
    if contract_sha != expected_contract_sha256:
        raise SweepGeometryError("certificate contract hash does not match the loaded contract")
    return certificate


def load_arm_sweep_config(path: Path) -> ArmSweepConfig:
    """Load the strict review-gated contract; no certificate means motion stays vetoed."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"arm sweep config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _sweep_mapping(raw, "arm sweep config")
    forbidden = sorted(set(root) & _SWEEP_FORBIDDEN_KEYS)
    if forbidden:
        raise SweepGeometryError(f"arm sweep config forbids keys: {forbidden}")
    _sweep_keys(root, _SWEEP_TOP_KEYS, "arm sweep config")
    if root["schema_version"] != ARM_SWEEP_SCHEMA_VERSION:
        raise SweepGeometryError("unsupported arm sweep schema_version")
    status = _sweep_text(root["status"], "status")
    if status not in {"blocked", "owner_approved"}:
        raise SweepGeometryError("arm sweep status must be blocked or owner_approved")

    geometry = _sweep_mapping(root["geometry"], "geometry")
    _sweep_keys(
        geometry,
        _SWEEP_GEOMETRY_KEYS,
        "geometry",
        optional=_SWEEP_OPTIONAL_GEOMETRY_KEYS,
    )
    urdf_path = _sweep_path(geometry["urdf"], "geometry.urdf", source)
    registration_path = _sweep_path(geometry["registration"], "geometry.registration", source)
    robot_config_path = _sweep_path(geometry["robot_config"], "geometry.robot_config", source)
    world_path = _sweep_path(geometry["world"], "geometry.world", source)
    urdf_sha = _sweep_sha(geometry["urdf_sha256"], "geometry.urdf_sha256")
    registration_sha = _sweep_sha(geometry["registration_sha256"], "geometry.registration_sha256")
    robot_sha = _sweep_sha(geometry["robot_config_sha256"], "geometry.robot_config_sha256")
    world_sha = _sweep_sha(geometry["world_sha256"], "geometry.world_sha256")
    links = tuple(
        _sweep_text(value, "geometry.required_arm_links")
        for value in cast_sequence(geometry["required_arm_links"])
    )
    if not links or len(set(links)) != len(links):
        raise SweepGeometryError("geometry.required_arm_links must be unique and nonempty")
    for name, path, expected in (
        ("URDF", urdf_path, urdf_sha),
        ("registration", registration_path, registration_sha),
        ("robot config", robot_config_path, robot_sha),
        ("world", world_path, world_sha),
    ):
        if not path.is_file() or _sha256_file(path) != expected:
            raise SweepGeometryError(f"{name} hash mismatch or missing file: {path}")
    if urdf_path != URDF_PATH.resolve() or robot_config_path != SETTINGS_PATH.resolve():
        raise SweepGeometryError("sweep contract must use the pinned URDF and robot config")
    if registration_sha != CORRECTED_REGISTRATION_SHA256:
        raise SweepGeometryError("sweep contract must use the corrected URDF registration")

    scene_path = _sweep_optional_path(
        geometry.get("scene_catalog"), "geometry.scene_catalog", source
    )
    scene_sha = _sweep_optional_sha(
        geometry.get("scene_catalog_sha256"), "geometry.scene_catalog_sha256"
    )
    proto_path = _sweep_optional_path(
        geometry.get("proto_manifest"), "geometry.proto_manifest", source
    )
    proto_sha = _sweep_optional_sha(
        geometry.get("proto_manifest_sha256"), "geometry.proto_manifest_sha256"
    )
    if (scene_path is None) != (scene_sha is None):
        raise SweepGeometryError("scene catalog path and hash must be supplied together")
    if (proto_path is None) != (proto_sha is None):
        raise SweepGeometryError("PROTO manifest path and hash must be supplied together")
    if scene_path is not None and scene_sha is not None:
        if not scene_path.is_file() or _sha256_file(scene_path) != scene_sha:
            raise SweepGeometryError(f"scene catalog hash mismatch or missing file: {scene_path}")
        catalog = load_scene_collision_catalog(scene_path)
        if (
            catalog.hashes["world"] != world_sha
            or catalog.hashes["registration"] != registration_sha
            or catalog.geometry.catalog_sha256 != scene_sha
        ):
            raise SweepGeometryError("scene catalog is not pinned to the sweep world/registration")
    if proto_path is not None and proto_sha is not None:
        if not proto_path.is_file() or _sha256_file(proto_path) != proto_sha:
            raise SweepGeometryError(f"PROTO manifest hash mismatch or missing file: {proto_path}")
        load_proto_source_manifest(
            proto_path,
            world_path=world_path,
            expected_world_sha256=world_sha,
        )

    contract = _sweep_mapping(root["contract"], "contract")
    _sweep_keys(
        contract,
        _SWEEP_CONTRACT_KEYS,
        "contract",
        optional=_SWEEP_OPTIONAL_CONTRACT_KEYS,
    )
    if contract["dt_source"] != ARM_SWEEP_DT_SOURCE:
        raise SweepGeometryError("sweep contract dt_source is not the pinned robot time step")
    if contract["speed_source"] != ARM_SWEEP_SPEED_SOURCE:
        raise SweepGeometryError("sweep contract speed_source is unsupported")
    raw_margin = contract["margin_m"]
    substeps = contract["substeps"]
    if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)):
        raise SweepGeometryError("sweep contract margin must be a finite positive number")
    margin = float(raw_margin)
    if (
        not math.isfinite(margin)
        or margin <= 0.0
        or isinstance(substeps, bool)
        or not isinstance(substeps, int)
        or substeps < 1
    ):
        raise SweepGeometryError("sweep contract requires a positive margin and integer substeps")
    required_objects = _sweep_string_set(
        contract["required_objects"], "contract.required_objects", required=True
    )
    permitted = _sweep_string_set(
        contract["permitted_contact_objects"], "contract.permitted_contact_objects"
    )
    raw_pairs = contract["adjacent_pairs"]
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise SweepGeometryError("contract.adjacent_pairs must be a list of capsule-name pairs")
    pairs = frozenset(
        frozenset(_sweep_string_set(pair, "contract.adjacent_pairs", required=True))
        for pair in raw_pairs
    )
    if any(len(pair) != 2 for pair in pairs):
        raise SweepGeometryError("each adjacent pair must name exactly two collision capsules")
    adaptive_speed = _sweep_optional_float(
        contract.get("adaptive_speed_bound_m_s"), "contract.adaptive_speed_bound_m_s"
    )
    max_adaptive = _sweep_optional_int(
        contract.get("max_adaptive_substeps"), "contract.max_adaptive_substeps", 4096
    )
    joint_displacement = _sweep_optional_nonnegative_float(
        contract.get("max_joint_displacement_bound_rad"),
        "contract.max_joint_displacement_bound_rad",
    )
    joint_reach = _sweep_optional_nonnegative_float(
        contract.get("joint_reach_bound_m"), "contract.joint_reach_bound_m"
    )
    geometry_error = _sweep_optional_nonnegative_float(
        contract.get("geometry_error_bound_m"), "contract.geometry_error_bound_m"
    )
    registration_error = _sweep_optional_nonnegative_float(
        contract.get("registration_uncertainty_bound_m"),
        "contract.registration_uncertainty_bound_m",
    )
    numerical_error = _sweep_optional_nonnegative_float(
        contract.get("numerical_error_bound_m"), "contract.numerical_error_bound_m"
    )
    transform_error = _sweep_optional_nonnegative_float(
        contract.get("transform_error_bound_m"), "contract.transform_error_bound_m"
    )
    speed = _urdf_reach_speed_bound(urdf_path, links)
    if adaptive_speed is not None and not math.isclose(
        adaptive_speed, speed, rel_tol=1e-12, abs_tol=0.0
    ):
        raise SweepGeometryError(
            "adaptive speed bound must equal the pinned URDF point-speed certificate"
        )
    base = ArmSweepConfig(
        source_path=source,
        status=status,
        urdf_path=urdf_path,
        registration_path=registration_path,
        robot_config_path=robot_config_path,
        world_path=world_path,
        urdf_sha256=urdf_sha,
        registration_sha256=registration_sha,
        robot_config_sha256=robot_sha,
        world_sha256=world_sha,
        required_arm_links=links,
        required_objects=required_objects,
        adjacent_pairs=pairs,
        permitted_contact_objects=permitted,
        dt_s=float(SETTINGS.time_step) / 1000.0,
        margin_m=margin,
        substeps=substeps,
        speed_bound_m_s=speed,
        certificate=None,
        scene_catalog_path=scene_path,
        scene_catalog_sha256=scene_sha,
        proto_manifest_path=proto_path,
        proto_manifest_sha256=proto_sha,
        adaptive_speed_bound_m_s=adaptive_speed,
        max_adaptive_substeps=max_adaptive,
        max_joint_displacement_bound_rad=joint_displacement,
        joint_reach_bound_m=joint_reach,
        geometry_error_bound_m=geometry_error,
        registration_uncertainty_bound_m=registration_error,
        numerical_error_bound_m=numerical_error,
        transform_error_bound_m=transform_error,
    )
    certificate = _load_sweep_certificate(
        _sweep_mapping(root["certificate"], "certificate"),
        source,
        _sweep_contract_sha256(base),
        {
            "urdf": urdf_sha,
            "registration": registration_sha,
            "robot_config": robot_sha,
            "world": world_sha,
            **({"scene_catalog": scene_sha} if scene_sha is not None else {}),
            **({"proto_manifest": proto_sha} if proto_sha is not None else {}),
        },
    )
    if status == "owner_approved" and certificate is None:
        raise SweepGeometryError("owner_approved status requires a complete certificate")
    if status == "blocked" and certificate is not None:
        raise SweepGeometryError("blocked status cannot carry an owner certificate")
    return replace(base, certificate=certificate)


def cast_sequence(value: object) -> Sequence[object]:
    """Narrow a YAML value to a sequence without accepting strings or bytes."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SweepGeometryError("expected a non-string sequence")
    return value


def _build_configured_sweep_check(config: ArmSweepConfig) -> IntermediateSweptArmCheck:
    if config.status != "owner_approved":
        raise SweepGeometryError("owner_approved status is required for the arm-sweep callback")
    if config.certificate is None:
        raise SweepGeometryError("owner-reviewed swept-arm certificate is absent")
    scene: StaticSceneGeometry | None = None
    expected_base_pose_world: Pose3D | None = None
    if config.scene_catalog_path is not None:
        catalog = load_scene_collision_catalog(config.scene_catalog_path)
        if catalog.geometry.catalog_sha256 != config.scene_catalog_sha256:
            raise SweepGeometryError("scene catalog hash does not match the sweep contract")
        if (
            catalog.hashes["world"] != config.world_sha256
            or catalog.hashes["registration"] != config.registration_sha256
        ):
            raise SweepGeometryError("scene catalog is not pinned to the sweep contract")
        scene = catalog.geometry
        expected_base_pose_world = catalog.expected_base_pose_world
    kinematics = corrected_kinematics()
    frames: dict[str, int] = {}
    for link in config.required_arm_links:
        frame = UNVERIFIED_LINK_FRAMES.get(link)
        if frame is None or frame not in kinematics.idx:
            raise SweepGeometryError(f"missing IKPy frame for {link}")
        frames[link] = kinematics.idx[frame]
    primitives = urdf_collision_capsules(
        config.urdf_path,
        config.required_arm_links,
        max_point_speed_m_s=config.speed_bound_m_s,
    )
    required_links = tuple(
        sorted(
            primitive.capsule.name
            for link in config.required_arm_links
            for primitive in primitives[link]
        )
    )
    if scene is not None and scene.gripper is not None:
        if scene.gripper.name in required_links:
            raise SweepGeometryError("gripper envelope name collides with a URDF capsule")
        required_links = tuple(sorted((*required_links, scene.gripper.name)))
    if any(not pair.issubset(required_links) for pair in config.adjacent_pairs):
        raise SweepGeometryError("adjacent pair names must be present collision capsules")
    if not set(UNVERIFIED_ARM_LINKS).issubset(config.required_arm_links):
        raise SweepGeometryError("sweep contract does not cover the full relevant arm")

    def geometry(state: Any) -> tuple[Capsule, ...]:
        transforms = kinematics.forward(_full_joint_vector(state.robot.joint_pos))
        if len(transforms) <= max(frames.values()):
            raise SweepGeometryError("FK returned incomplete link transforms")
        result: list[Capsule] = []
        for link in config.required_arm_links:
            matrix = transforms[frames[link]]
            for primitive in primitives[link]:
                result.append(_transformed_capsule(primitive.capsule, matrix))
        return tuple(result)

    contract = ArmSweepContract(
        required_links=required_links,
        required_objects=config.required_objects,
        adjacent_pairs=config.adjacent_pairs,
        dt_s=config.dt_s,
        margin_m=config.margin_m,
        substeps=config.substeps,
        permitted_contact_objects=config.permitted_contact_objects,
        adaptive_speed_bound_m_s=config.adaptive_speed_bound_m_s,
        max_adaptive_substeps=config.max_adaptive_substeps,
        max_joint_displacement_bound_rad=config.max_joint_displacement_bound_rad,
        joint_reach_bound_m=config.joint_reach_bound_m,
        geometry_error_bound_m=config.geometry_error_bound_m,
        registration_uncertainty_bound_m=config.registration_uncertainty_bound_m,
        numerical_error_bound_m=config.numerical_error_bound_m,
        transform_error_bound_m=config.transform_error_bound_m,
    )
    return IntermediateSweptArmCheck(
        geometry,
        contract,
        scene=scene,
        expected_base_pose_world=expected_base_pose_world,
    )


def build_reviewed_arm_sweep(config_path: Path) -> IntermediateSweptArmCheck:
    """Build the production callback only after every owner certificate gate passes."""
    return _build_configured_sweep_check(load_arm_sweep_config(config_path))


def arm_sweep_transition_check(config_path: Path) -> IntermediateSweptArmCheck | None:
    """Return a reviewed callback, or ``None`` so `unverified-arm-sweep` stays hard."""
    config = load_arm_sweep_config(config_path)
    if config.status != "owner_approved" or config.certificate is None:
        return None
    return _build_configured_sweep_check(config)


def analytic_surrogate_factory(
    config: WorldMindConfig, inputs: Mapping[str, Any]
) -> AnalyticSurrogate:
    """Bind AnalyticSurrogate to corrected FK/IK and the real safety veto.

    The shipped blocked contract leaves ``check_transition=None``. An explicit reviewed
    ``arm_sweep_config`` input is the only path that can attach the full-arm callback; an invalid
    or unapproved contract raises instead of weakening the ``unverified-arm-sweep`` veto.
    """
    forward = MemoizedForward(
        _corrected_forward, geometry_id=f"corrected-urdf:{CORRECTED_REGISTRATION_SHA256}"
    )
    solver = corrected_pose_solver()
    monitor = runtime_safety_factory(config, {})
    tracking = VelocityLimitedKinematics(dict(_urdf_velocities()), forward)
    sweep_path = inputs.get("arm_sweep_config")
    transition_check = (
        arm_sweep_transition_check(Path(sweep_path))
        if isinstance(sweep_path, (str, Path))
        else None
    )
    return AnalyticSurrogate(
        tracking,
        inverse=lambda pose, robot: JointTarget(
            solver.solve(pose, WorldState(SceneState([]), robot)).joints
        ),
        gripper_speed_m_s=float(SETTINGS.finger_speed),
        gravity_m_s2=9.81,
        support_z_m={config.task.target: config.verification.destination_world[2]},
        validate=lambda action, robot: monitor.validate(action, robot).accepted,
        check_transition=transition_check,
    )


def placement_verifier_factory(
    config: WorldMindConfig,
    destination_base: Pose3D,
    rest_center_base: Pose3D,
    inputs: Mapping[str, Any],
) -> PlacementVerifier:
    """Bind the existing observed-only verifier with frozen tolerances from config."""
    del inputs
    return PlacementVerifier(
        destination_base=destination_base,
        rest_center_base=rest_center_base,
        up_axis_base=config.verification.up_axis_base,
        home_joints=config.verification.home_joints,
        distance_tolerance_m=config.verification.distance_tolerance_m,
        height_tolerance_m=config.verification.height_tolerance_m,
        tilt_limit_rad=config.verification.tilt_limit_rad,
        home_tolerance_rad=config.verification.home_tolerance_rad,
    )


def finite_chunk_check(state: WorldState, chunk: Any, trajectory: Any) -> bool:
    """Mechanical validity check; it claims no model, safety margin or scientific result."""
    del state, trajectory
    actions = getattr(chunk, "actions", None)
    if not isinstance(actions, Sequence) or not actions:
        return False
    for action in actions:
        if isinstance(action, JointTarget):
            if not all(isinstance(name, str) and name for name in action.q) or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in action.q.values()
            ):
                return False
        elif isinstance(action, GripperCmd):
            if not isinstance(action.opening, (int, float)) or not math.isfinite(action.opening):
                return False
        else:
            return False
    return True


def _inverse_pose(pose: Pose3D) -> Pose3D:
    quaternion = np.asarray(pose.orientation, dtype=np.float64)
    position = pose.position.as_array()
    if quaternion.shape != (4,) or not np.isfinite(np.r_[quaternion, position]).all():
        raise ValueError("finite xyzw pose required for frame conversion")
    if not math.isclose(float(quaternion @ quaternion), 1.0, abs_tol=1e-8):
        raise ValueError("unit xyzw pose required for frame conversion")
    inverse = np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]))
    translated = rotate(inverse, -position)
    return Pose3D(
        Vec3(*map(float, translated)),
        (float(inverse[0]), float(inverse[1]), float(inverse[2]), float(inverse[3])),
    )


def corrected_scripted_waypoints(
    config: WorldMindConfig, state: WorldState, task: Any
) -> tuple[ScriptedWaypoint, ...]:
    """Derive baseline geometry from the oracle-initialized target and world-anchored destination.

    The target pose is supplied by the declared oracle state adapter; this function only converts
    it and the frozen destination into the base frame. It never calls the backend.
    """
    if state.robot.base_pose_world is None:
        raise ValueError("corrected waypoints require an explicit world-to-base transform")
    target = next((obj for obj in state.scene.objects if obj.object_id == task.target), None)
    if target is None:
        raise ValueError(f"corrected waypoints require {task.target} in the observed scene")
    base_inverse = _inverse_pose(state.robot.base_pose_world)
    destination_base = compose(
        base_inverse,
        Pose3D(Vec3(*config.verification.destination_world), _IDENTITY_QUAT),
    )
    anchors = {"target": target.pose.position, "destination": destination_base.position}
    closed, opened = (float(value) for value in SETTINGS.gripper_opening_limits)

    def pose_station(
        label: str, anchor: str, z_offset: float, position_tolerance_m: float
    ) -> ScriptedWaypoint:
        position = anchors[anchor]
        return ScriptedWaypoint(
            label,
            _CORRECTED_HOLD_STEPS[label],
            pose=Pose3D(
                Vec3(
                    float(position.x),
                    float(position.y),
                    float(position.z + z_offset),
                ),
                _IDENTITY_QUAT,
            ),
            position_tolerance_m=float(position_tolerance_m),
            orientation_tolerance_rad=float(config.adapter.ik_orientation_tolerance_rad),
        )

    waypoints: list[ScriptedWaypoint] = [
        ScriptedWaypoint("INIT", _CORRECTED_HOLD_STEPS["INIT"], gripper_opening=opened),
        pose_station("PREGRASP", "target", GRASP_Z + APPROACH_H, IK_TOL),
        pose_station("DESCEND", "target", GRASP_Z, IK_TOL_GRASP),
        ScriptedWaypoint("CLOSE", _CORRECTED_HOLD_STEPS["CLOSE"], gripper_opening=closed),
        pose_station("LIFT", "target", GRASP_Z + APPROACH_H, IK_TOL_TRANSIT),
        pose_station("TRANSIT", "destination", GRASP_Z + APPROACH_H, IK_TOL_TRANSIT),
        pose_station("PLACE_DESCEND", "destination", GRASP_Z + PLACE_CLEAR, IK_TOL_GRASP),
        ScriptedWaypoint("RELEASE", _CORRECTED_HOLD_STEPS["RELEASE"], gripper_opening=opened),
        pose_station("RETREAT", "destination", GRASP_Z + APPROACH_H, IK_TOL_TRANSIT),
        ScriptedWaypoint(
            "HOME",
            _CORRECTED_HOLD_STEPS["HOME"],
            joints=dict(config.verification.home_joints),
        ),
    ]
    if sum(item.hold_steps for item in waypoints) != config.pipeline.horizon_steps:
        raise ValueError("corrected waypoint cadence does not sum to the frozen horizon")
    return tuple(waypoints)


def runtime_preflight_probe(
    config: WorldMindConfig | WorldMindDiagnosticConfig, bindings: WorldMindBindings
) -> dict[str, str]:
    """Report inert backend-boundary facts; no Webots process or IK solve is started."""

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def sweep_status() -> str:
        configured = bindings.inputs.get("surrogate", {}).get("arm_sweep_config")
        if configured is None:
            return "not_configured"
        certificate = load_arm_sweep_config(Path(configured)).certificate
        return "absent" if certificate is None else certificate.certificate_id

    limits = runtime_safety_limits()
    solver = bindings.pose_solvers.get(config.adapter.pose_solver_binding)
    waypoint_factory = bindings.waypoint_factories.get(config.adapter.waypoints_binding)
    if not callable(getattr(solver, "solve", None)):
        raise ValueError("registered pose solver cannot solve")
    if not callable(waypoint_factory):
        raise ValueError("registered waypoint factory is not callable")
    return {
        "robot_config_sha256": digest(SETTINGS_PATH),
        "safety_config_sha256": digest(config.safety.config_path),
        "urdf_sha256": digest(URDF_PATH),
        "corrected_registration_sha256": digest(corrected_registration_path()),
        "pose_solver": type(solver).__name__,
        "safety_limits": (
            f"joints={len(limits.joint_position)} "
            f"workspace_min={limits.workspace_min} workspace_max={limits.workspace_max}"
        ),
        "waypoint_factory": getattr(waypoint_factory, "__name__", type(waypoint_factory).__name__),
        "sweep_certificate": sweep_status(),
    }


# Structural coverage only. The IKPy frame key for each URDF link is explicit; any missing key
# or transform fails closed. This is NOT a certified collision model (ADR-0049).
UNVERIFIED_ARM_LINKS: tuple[str, ...] = (
    "TIAGo front arm",
    "TIAGo front arm_3",
    "arm_2_link",
    "arm_3_link",
    "arm_4_link",
    "arm_5_link",
    "arm_6_link",
    "arm_7_link",
    "wrist_ft_tool_link",
)
UNVERIFIED_LINK_FRAMES: Mapping[str, str] = {
    "TIAGo front arm": "torso_lift_link_TIAGo front arm_joint",
    "TIAGo front arm_3": "arm_1_joint",
    "arm_2_link": "arm_2_joint",
    "arm_3_link": "arm_3_joint",
    "arm_4_link": "arm_4_joint",
    "arm_5_link": "arm_5_joint",
    "arm_6_link": "arm_6_joint",
    "arm_7_link": "arm_7_joint",
    "wrist_ft_tool_link": "wrist_ft_tool_link_front_joint",
}


def _transformed_capsule(capsule: Capsule, matrix: Any) -> Capsule:
    """Apply a rigid FK frame to a link-local capsule; reject invalid transforms."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (4, 4) or not np.isfinite(rotation).all():
        raise SweepGeometryError("invalid FK transform for arm collision geometry")
    if not np.allclose(rotation[:3, :3] @ rotation[:3, :3].T, np.eye(3), atol=1e-6):
        raise SweepGeometryError("non-rigid FK transform for arm collision geometry")
    start = rotation[:3, :3] @ np.asarray(capsule.start) + rotation[:3, 3]
    end = rotation[:3, :3] @ np.asarray(capsule.end) + rotation[:3, 3]
    if not np.isfinite(np.r_[start, end]).all():
        raise SweepGeometryError("nonfinite transformed arm capsule")
    start_xyz = (float(start[0]), float(start[1]), float(start[2]))
    end_xyz = (float(end[0]), float(end[1]), float(end[2]))
    return Capsule(
        capsule.name,
        start_xyz,
        end_xyz,
        capsule.radius_m,
        capsule.max_point_speed_m_s,
    )


def unverified_urdf_sweep_check(
    *,
    margin_m: float,
    substeps: int,
    max_point_speed_m_s: float,
    required_objects: frozenset[str],
    adjacent_pairs: frozenset[frozenset[str]],
    permitted_contact_objects: frozenset[str] = frozenset(),
) -> IntermediateSweptArmCheck:
    """Build the opt-in, UNCERTIFIED URDF full-arm sweep; never call from scored production.

    All physical margins, speed certificates and scene coverage must be supplied explicitly by an
    owner review. The returned checker uses the corrected registration and every supported
    collision primitive on each relevant arm link; it is intentionally absent from
    ``production_worldmind_bindings()`` so the ``unverified-arm-sweep`` veto stays hard.
    """
    if set(UNVERIFIED_LINK_FRAMES) != set(UNVERIFIED_ARM_LINKS):
        raise SweepGeometryError("arm link/frame coverage is incomplete")
    kinematics = corrected_kinematics()
    frames: dict[str, int] = {}
    for link, joint in UNVERIFIED_LINK_FRAMES.items():
        if joint not in kinematics.idx:
            raise SweepGeometryError(f"missing IKPy frame for {link}")
        frames[link] = kinematics.idx[joint]
    primitives = urdf_collision_capsules(
        URDF_PATH,
        UNVERIFIED_ARM_LINKS,
        max_point_speed_m_s=max_point_speed_m_s,
    )
    required_links = tuple(
        sorted(
            primitive.capsule.name
            for link in UNVERIFIED_ARM_LINKS
            for primitive in primitives[link]
        )
    )

    def geometry(state: Any) -> tuple[Capsule, ...]:
        transforms = kinematics.forward(_full_joint_vector(state.robot.joint_pos))
        if len(transforms) <= max(frames.values()):
            raise SweepGeometryError("FK returned incomplete link transforms")
        result: list[Capsule] = []
        for link in UNVERIFIED_ARM_LINKS:
            matrix = transforms[frames[link]]
            for primitive in primitives[link]:
                result.append(_transformed_capsule(primitive.capsule, matrix))
        return tuple(result)

    return IntermediateSweptArmCheck(
        geometry,
        ArmSweepContract(
            required_links=required_links,
            required_objects=frozenset(required_objects),
            adjacent_pairs=frozenset(adjacent_pairs),
            dt_s=float(SETTINGS.time_step) / 1000.0,
            margin_m=float(margin_m),
            substeps=substeps,
            permitted_contact_objects=frozenset(permitted_contact_objects),
        ),
    )


def production_worldmind_bindings() -> WorldMindBindings:
    """Return every concrete binding that exists without inventing a scientific component.

    Eq. 3.8 component registrations, weights, ranking direction, risk thresholds and calibration
    evaluators are absent. A ready config referencing them therefore still fails closed.
    """
    return WorldMindBindings(
        components={},
        hard_checks={FINITE_CHUNK_CHECK: finite_chunk_check},
        surrogates={ANALYTIC_SURROGATE_BINDING: analytic_surrogate_factory},
        safety={RUNTIME_SAFETY_BINDING: runtime_safety_factory},
        verification={PLACEMENT_VERIFIER_BINDING: placement_verifier_factory},
        pose_solvers={CORRECTED_POSE_SOLVER_BINDING: corrected_pose_solver()},
        waypoint_sets={},
        waypoint_factories={CORRECTED_WAYPOINTS_BINDING: corrected_scripted_waypoints},
        calibration_evaluators={},
        inputs={
            "surrogate": {
                **{name: name for name in SURROGATE_INPUTS},
                "arm_sweep_config": ARM_SWEEP_CONFIG_PATH,
            },
            "safety": {name: name for name in SAFETY_INPUTS},
            "verification": {name: name for name in VERIFICATION_INPUTS},
        },
    )
