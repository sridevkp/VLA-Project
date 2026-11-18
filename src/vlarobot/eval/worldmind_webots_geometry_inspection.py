"""Pure evidence rules for the dedicated read-only Webots geometry inspection.

This module imports no simulator, backend, ROS or MuJoCo package. The live export is collected by
``vlarobot.backends.worldmind_webots_geometry_inspection`` and injected as a snapshot. Session,
identity and launch-attestation checks are reused from the existing no-actuation diagnostic.
Nothing here launches Webots, solves IK, applies a command, scores an episode or reads TEST/TRAIN.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vlarobot.eval.worldmind_webots_diagnostic import (
    AUTHORITATIVE_REPORT_SHA256,
    DiagnosticSnapshot,
    WebotsDiagnosticConfig,
    _keys,
    _launch_version_attestation,
    _mapping,
    _observation_blockers,
    _path,
    _pose_payload,
    _sha256_file,
    _sha256_text,
    _snapshot_blockers,
    _text,
    load_webots_diagnostic_config,
    write_diagnostic_evidence,
)

GEOMETRY_INSPECTION_SCHEMA_VERSION = 1
GEOMETRY_INSPECTION_STATUS = "diagnostic_uncalibrated"
GEOMETRY_INSPECTION_MODE = "live_read_only_bounding_object_export"
_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics", "inspect", "get_ground_truth")
_ALLOWED_CALLS = ("ping", "inspect_geometry")
_GEOMETRY_INSPECTION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "mode",
        "scene_catalog",
        "scene_catalog_sha256",
        "proto_manifest",
        "proto_manifest_sha256",
        "arm_sweep_contract",
        "arm_sweep_contract_sha256",
    }
)
_GEOMETRY_OBJECT_KEYS = frozenset(
    {"id", "name", "def", "type", "T_base_owner", "T_world_owner", "shapes"}
)
_GEOMETRY_STATUS_MARKER = "MEASURED live expanded boundingObject export"
_NO_SPLIT_MARKER = "no TRAIN/VALIDATION/TEST task rollout"


def normalize_webots_sfbool(value: object, *, field_name: str = "SFBool") -> bool:
    """Normalize the documented Webots ``SFBool`` representations strictly.

    Webots R2025a may expose ``SFBool`` as a Python ``bool`` or as the documented integer
    ``0``/``1``. Only ``False``, ``True``, ``0`` and ``1`` are accepted; every other integer,
    numeric type, string, ``None`` or container is an ambiguous observation and raises. The raw
    value remains available in the exported evidence, so normalization never erases provenance.
    """
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(
        f"ambiguous Webots {field_name} value {value!r} of type {type(value).__name__}"
    )


@dataclass(frozen=True)
class GeometryInspectionConfig:
    """Reviewed read-only geometry contract nested inside the Webots diagnostic contract."""

    diagnostic: WebotsDiagnosticConfig
    inspection_mode: str
    scene_catalog_path: Path
    scene_catalog_sha256: str
    proto_manifest_path: Path
    proto_manifest_sha256: str
    arm_sweep_contract_path: Path
    arm_sweep_contract_sha256: str


@dataclass(frozen=True)
class GeometryInspectionSnapshot(DiagnosticSnapshot):
    """Read-only backend facts plus the supervisor-side boundingObject export."""

    geometry: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryInspectionPreflight:
    blockers: tuple[str, ...]
    hashes: Mapping[str, str | None]
    motion_blockers: tuple[str, ...]


def load_geometry_inspection_config(path: Path) -> GeometryInspectionConfig:
    """Load the strict geometry-inspection config; forbid scoring and scientific parameters."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Webots geometry inspection config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "Webots geometry inspection config")
    section = _mapping(root["geometry_inspection"], "geometry_inspection")
    _keys(section, _GEOMETRY_INSPECTION_KEYS, "geometry_inspection")
    if section["schema_version"] != GEOMETRY_INSPECTION_SCHEMA_VERSION:
        raise ValueError("unsupported geometry_inspection schema_version")
    status = _text(section["status"], "geometry_inspection.status")
    if status != GEOMETRY_INSPECTION_STATUS:
        raise ValueError(f"geometry_inspection.status must be '{GEOMETRY_INSPECTION_STATUS}'")
    mode = _text(section["mode"], "geometry_inspection.mode")
    if mode != GEOMETRY_INSPECTION_MODE:
        raise ValueError(f"geometry_inspection.mode must be '{GEOMETRY_INSPECTION_MODE}'")

    diagnostic = load_webots_diagnostic_config(
        source, extra_top_level_keys=frozenset({"geometry_inspection"})
    )
    if diagnostic.swept_arm_certificate is not None:
        raise ValueError("read-only geometry inspection forbids a swept-arm certificate")
    return GeometryInspectionConfig(
        diagnostic=diagnostic,
        inspection_mode=mode,
        scene_catalog_path=_path(
            section["scene_catalog"], "geometry_inspection.scene_catalog", source
        ),
        scene_catalog_sha256=_sha256_text(
            section["scene_catalog_sha256"], "geometry_inspection.scene_catalog_sha256"
        ),
        proto_manifest_path=_path(
            section["proto_manifest"], "geometry_inspection.proto_manifest", source
        ),
        proto_manifest_sha256=_sha256_text(
            section["proto_manifest_sha256"], "geometry_inspection.proto_manifest_sha256"
        ),
        arm_sweep_contract_path=_path(
            section["arm_sweep_contract"], "geometry_inspection.arm_sweep_contract", source
        ),
        arm_sweep_contract_sha256=_sha256_text(
            section["arm_sweep_contract_sha256"],
            "geometry_inspection.arm_sweep_contract_sha256",
        ),
    )


def preflight_geometry_inspection(config: GeometryInspectionConfig) -> GeometryInspectionPreflight:
    """Verify pinned local files before any Webots process or socket is created."""
    diagnostic = config.diagnostic
    paths = {
        "config": diagnostic.source_path,
        "world": diagnostic.world,
        "urdf": diagnostic.urdf_path,
        "registration": diagnostic.registration_path,
        "robot_config": diagnostic.robot_config_path,
        "scene_catalog": config.scene_catalog_path,
        "proto_manifest": config.proto_manifest_path,
        "arm_sweep_contract": config.arm_sweep_contract_path,
    }
    hashes: dict[str, str | None] = {}
    blockers: list[str] = []
    for name, path in paths.items():
        digest = _sha256_file(path)
        hashes[name] = digest
        if digest is None:
            blockers.append(f"missing geometry/config file: {name}={path}")
    expected = {
        "world": diagnostic.world_sha256,
        "scene_catalog": config.scene_catalog_sha256,
        "proto_manifest": config.proto_manifest_sha256,
        "arm_sweep_contract": config.arm_sweep_contract_sha256,
    }
    for name, digest_expected in expected.items():
        if hashes.get(name) is not None and hashes[name] != digest_expected:
            blockers.append(f"{name} SHA-256 does not match the reviewed geometry config")
    return GeometryInspectionPreflight(
        tuple(blockers),
        hashes,
        (
            "diagnostic_no_actuation_mode",
            "unverified-arm-sweep",
            "unresolved_forwarded_geometry",
        ),
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _geometry_blockers(
    config: GeometryInspectionConfig, snapshot: GeometryInspectionSnapshot
) -> list[str]:
    geometry = snapshot.geometry
    required = {
        "status",
        "robot_id",
        "self_collision_enabled",
        "sim_time_s",
        "objects",
        "devices",
        "joint_positions",
        "motor_velocity_settings",
        "scene",
        "unsupported_shapes",
        "motor_commands_sent",
    }
    blockers: list[str] = []
    if not set(geometry).issuperset(required):
        blockers.append("geometry export omitted required fields")
    status = geometry.get("status")
    if not isinstance(status, str) or _GEOMETRY_STATUS_MARKER not in status:
        blockers.append("geometry export status is not a measured read-only export")
    if geometry.get("motor_commands_sent") != 0:
        blockers.append("geometry export reports motor commands")
    try:
        self_collision = normalize_webots_sfbool(
            geometry.get("self_collision_enabled"), field_name="selfCollision"
        )
    except ValueError:
        blockers.append("live selfCollision value is not a documented Webots SFBool")
    else:
        if self_collision is not False:
            blockers.append("live selfCollision setting differs from the pinned PROTO")
    sim_time = _finite_number(geometry.get("sim_time_s"))
    if sim_time is None or sim_time < 0.0:
        blockers.append("geometry export simulation time is invalid")
    if geometry.get("robot_id") is None:
        blockers.append("geometry export robot identity is missing")
    scene = geometry.get("scene")
    if not isinstance(scene, str) or _NO_SPLIT_MARKER not in scene:
        blockers.append("geometry export scene declaration is missing or unsafe")

    objects = geometry.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        blockers.append("geometry export contains no bounding objects")
    else:
        for index, item in enumerate(objects):
            if not isinstance(item, Mapping) or not set(item).issuperset(_GEOMETRY_OBJECT_KEYS):
                blockers.append(f"geometry object {index} is incomplete")
                continue
            if not isinstance(item.get("shapes"), Sequence) or isinstance(
                item.get("shapes"), (str, bytes)
            ):
                blockers.append(f"geometry object {index} shapes are invalid")
        identity_def = snapshot.identity.get("robot_def")
        object_defs = {item.get("def") for item in objects if isinstance(item, Mapping)} | {
            item.get("name") for item in objects if isinstance(item, Mapping)
        }
        if not isinstance(identity_def, str) or identity_def not in object_defs:
            blockers.append("geometry export robot identity is not cross-checked")

    unsupported = geometry.get("unsupported_shapes")
    if not isinstance(unsupported, Sequence) or isinstance(unsupported, (str, bytes)):
        blockers.append("geometry export unsupported-shape list is invalid")
    for name in _FORBIDDEN_CALLS:
        if snapshot.call_counts.get(name, 0) != 0:
            blockers.append(f"forbidden geometry backend call was attempted: {name}")
    if snapshot.identity.get("physics_step_count") != 0:
        blockers.append("a physics step occurred in this geometry inspection session")
    return blockers


def finalize_geometry_inspection(
    config: GeometryInspectionConfig,
    preflight: GeometryInspectionPreflight,
    snapshot: GeometryInspectionSnapshot,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Return non-scored report, state, provenance and raw geometry artifacts."""
    diagnostic = config.diagnostic
    observation_blockers = _observation_blockers(diagnostic, snapshot)
    launch_blocker, launch_version = _launch_version_attestation(diagnostic, snapshot)
    controller_version = _observed_text(snapshot.identity.get("webots_version"))
    runtime_version = launch_version or controller_version
    session_blockers = _snapshot_blockers(diagnostic, snapshot)
    geometry_blockers = _geometry_blockers(config, snapshot)
    blockers = [*preflight.blockers, *session_blockers, *geometry_blockers]
    ready = not blockers
    objects = snapshot.geometry.get("objects", ())
    unsupported = snapshot.geometry.get("unsupported_shapes", ())
    raw_self_collision = snapshot.geometry.get("self_collision_enabled")
    try:
        normalized_self_collision: bool | None = normalize_webots_sfbool(
            raw_self_collision, field_name="selfCollision"
        )
    except ValueError:
        normalized_self_collision = None
    report: dict[str, object] = {
        "schema_version": GEOMETRY_INSPECTION_SCHEMA_VERSION,
        "outcome": "PASS" if ready else "FAILED",
        "geometry_inspection_mode": config.inspection_mode,
        "diagnostic_mode": diagnostic.diagnostic_mode,
        "scored_evaluation_allowed": False,
        "scored_evaluator": None,
        "scored_results_written": False,
        "frozen_campaign_modified": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "state_source": diagnostic.state_source,
        "split": diagnostic.split,
        "oracle_equivalent_to_perception": False,
        "backend_declaration": diagnostic.backend_declaration,
        "backend_kind": "webots",
        "launch_mode": diagnostic.mode,
        "endpoint": snapshot.ping.get("endpoint"),
        "simulator_session_id": snapshot.ping.get("session_id"),
        "running_world_sha256_observed": snapshot.ping.get("world_sha256"),
        "simulator_identity": dict(snapshot.identity),
        "simulator_version": runtime_version,
        "runtime_webots_version_observed": runtime_version,
        "runtime_webots_version_available": runtime_version is not None,
        "runtime_webots_version_verified": launch_blocker is None and launch_version is not None,
        "launch_webots_attestation": dict(snapshot.launch_attestation),
        "robot_def_observed": snapshot.identity.get("robot_def"),
        "robot_def_cross_checked": snapshot.identity.get("robot_def_cross_checked"),
        "robot_identity_declared": diagnostic.robot_identity,
        "observation_timestamp_s": snapshot.observation.t,
        "observation_age_s": snapshot.age_s,
        "observation_freshness_limit_s": diagnostic.observation_freshness_max_s,
        "observation_valid": not observation_blockers,
        "observation_blockers": observation_blockers,
        "geometry_hashes": dict(preflight.hashes),
        "geometry_object_count": len(objects) if isinstance(objects, Sequence) else None,
        "unsupported_shape_count": len(unsupported) if isinstance(unsupported, Sequence) else None,
        "unsupported_shapes": list(unsupported) if isinstance(unsupported, Sequence) else [],
        "self_collision_enabled": raw_self_collision,
        "self_collision_enabled_normalized": normalized_self_collision,
        "self_collision_enabled_observed_type": type(raw_self_collision).__name__,
        "motor_commands_sent": snapshot.geometry.get("motor_commands_sent"),
        "physics_step_count": snapshot.identity.get("physics_step_count"),
        "geometry_export_valid": not geometry_blockers,
        "geometry_blockers": geometry_blockers,
        "read_only_inspection_allowed": not preflight.blockers,
        "no_actuation": True,
        "backend_apply_calls": snapshot.call_counts.get("apply", 0),
        "backend_step_calls": snapshot.call_counts.get("step", 0),
        "backend_reset_calls": snapshot.call_counts.get("reset", 0),
        "backend_export_calls": snapshot.call_counts.get("export_kinematics", 0),
        "backend_inspect_calls": snapshot.call_counts.get("inspect", 0),
        "backend_ground_truth_calls": snapshot.call_counts.get("get_ground_truth", 0),
        "backend_inspect_geometry_calls": snapshot.call_counts.get("inspect_geometry", 0),
        "actuator_commands": [],
        "dependency_availability": dict(snapshot.dependency_availability),
        "safety_check_readiness": all(snapshot.dependency_availability.values()),
        "motion_allowed": False,
        "motion_readiness": "NOT_READY",
        "motion_blockers": list(preflight.motion_blockers),
        "blockers": blockers,
    }
    state = {
        "captured_at_utc": snapshot.captured_at_utc,
        "observation_timestamp_s": snapshot.observation.t,
        "observation_age_s": snapshot.age_s,
        "joint_pos": dict(snapshot.observation.robot.joint_pos),
        "joint_vel": dict(snapshot.observation.robot.joint_vel),
        "ee_pose": _pose_payload(snapshot.observation.robot.ee_pose),
        "base_pose_world": _pose_payload(snapshot.observation.robot.base_pose_world),
        "gripper_opening": snapshot.observation.robot.gripper_opening,
        "oracle_target_object": diagnostic.target_object,
        "oracle_target_pose": _pose_payload(
            snapshot.ground_truth.object_poses.get(diagnostic.target_object)
        ),
        "geometry_object_count": report["geometry_object_count"],
        "unsupported_shape_count": report["unsupported_shape_count"],
    }
    provenance = {
        "geometry_inspection_only": True,
        "authoritative_inert_report_sha256": AUTHORITATIVE_REPORT_SHA256,
        "config_sha256": preflight.hashes.get("config"),
        "world_sha256": preflight.hashes.get("world"),
        "urdf_sha256": preflight.hashes.get("urdf"),
        "registration_sha256": preflight.hashes.get("registration"),
        "robot_config_sha256": preflight.hashes.get("robot_config"),
        "scene_catalog_sha256": preflight.hashes.get("scene_catalog"),
        "proto_manifest_sha256": preflight.hashes.get("proto_manifest"),
        "arm_sweep_contract_sha256": preflight.hashes.get("arm_sweep_contract"),
        "session_id": snapshot.ping.get("session_id"),
        "captured_at_utc": snapshot.captured_at_utc,
        "frozen_campaign_modified": False,
        "scored_results_written": False,
    }
    geometry = {
        "schema_version": GEOMETRY_INSPECTION_SCHEMA_VERSION,
        "inspection_mode": config.inspection_mode,
        "motion_ready": False,
        "export": dict(snapshot.geometry),
    }
    return report, state, provenance, geometry


def failed_geometry_inspection(
    config: GeometryInspectionConfig,
    preflight: GeometryInspectionPreflight,
    reason: str,
) -> dict[str, object]:
    """Explicit fail-closed report used before any backend session is created."""
    return {
        "schema_version": GEOMETRY_INSPECTION_SCHEMA_VERSION,
        "outcome": "FAILED",
        "geometry_inspection_mode": config.inspection_mode,
        "scored_evaluation_allowed": False,
        "scored_evaluator": None,
        "scored_results_written": False,
        "frozen_campaign_modified": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "state_source": config.diagnostic.state_source,
        "split": config.diagnostic.split,
        "backend_declaration": config.diagnostic.backend_declaration,
        "geometry_hashes": dict(preflight.hashes),
        "read_only_inspection_allowed": False,
        "no_actuation": True,
        "backend_apply_calls": 0,
        "backend_step_calls": 0,
        "backend_reset_calls": 0,
        "backend_export_calls": 0,
        "backend_inspect_calls": 0,
        "backend_ground_truth_calls": 0,
        "backend_inspect_geometry_calls": 0,
        "actuator_commands": [],
        "motion_allowed": False,
        "motion_readiness": "NOT_READY",
        "motion_blockers": list(preflight.motion_blockers),
        "blockers": [reason, *preflight.blockers],
    }


def _observed_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def write_geometry_inspection_evidence(
    out_dir: Path,
    report: Mapping[str, object],
    geometry: Mapping[str, object] | None,
    state: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> tuple[Path, ...]:
    """Write exclusive artifacts into the caller-created output directory."""
    out = Path(out_dir)
    files = list(
        write_diagnostic_evidence(
            out,
            report,
            state,
            provenance,
        )
    )
    if geometry is not None:
        path = out / "geometry.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(geometry, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        files.append(path)
    return tuple(files)


__all__ = [
    "GEOMETRY_INSPECTION_MODE",
    "GEOMETRY_INSPECTION_SCHEMA_VERSION",
    "GeometryInspectionConfig",
    "GeometryInspectionPreflight",
    "GeometryInspectionSnapshot",
    "failed_geometry_inspection",
    "finalize_geometry_inspection",
    "load_geometry_inspection_config",
    "normalize_webots_sfbool",
    "preflight_geometry_inspection",
    "write_geometry_inspection_evidence",
]
