"""Pure, non-scored evidence rules for the no-actuation Webots diagnostic.

This module imports no simulator, backend, ROS or MuJoCo package. The real Webots session is
collected behind ``vlarobot.backends.worldmind_webots_diagnostic`` and injected as a snapshot.
Nothing here scores a campaign, selects parameters, reads TEST/TRAIN evidence or commands a robot.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from vlarobot.core.types import GroundTruth, Observation, Pose3D

WEBOTS_DIAGNOSTIC_SCHEMA_VERSION = 1
WEBOTS_DIAGNOSTIC_STATUS = "diagnostic_uncalibrated"
WEBOTS_DIAGNOSTIC_MODE = "read_only_inspection"
FROZEN_DIAGNOSTIC_SPLIT = "validation"
FROZEN_STATE_SOURCE = "oracle"
AUTHORITATIVE_REPORT_SHA256 = "23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "split",
        "state_source",
        "mode",
        "backend",
        "webots",
        "world",
        "world_sha256",
        "endpoint",
        "target_object",
        "robot_identity",
        "webots_version",
        "seed",
        "scenario",
        "observation_freshness_max_s",
        "geometry",
        "dependencies",
        "diagnostic",
    }
)
_FORBIDDEN_TOP_LEVEL_KEYS = frozenset(
    {
        "train",
        "test",
        "dataset",
        "scoring",
        "calibration",
        "components",
        "weights",
        "evaluator",
        "maximize",
    }
)
_GEOMETRY_KEYS = frozenset(
    {
        "urdf",
        "registration",
        "robot_config",
        "required_arm_links",
        "swept_arm_certificate",
    }
)
_DEPENDENCY_KEYS = frozenset(
    {
        "pose_solver_binding",
        "safety_binding",
        "verification_binding",
        "geometry_binding",
    }
)
_DIAGNOSTIC_KEYS = frozenset(
    {
        "mode",
        "no_actuation",
        "scored_evaluation_allowed",
        "evidence_collection_split",
    }
)
_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics")


@dataclass(frozen=True)
class WebotsDiagnosticConfig:
    """Fully reviewed read-only diagnostic contract; no scientific parameters are present."""

    schema_version: int
    status: str
    split: str
    state_source: str
    mode: Literal["launch", "connect"]
    backend_declaration: str
    webots: str
    world: Path
    world_sha256: str
    endpoint: str
    target_object: str
    robot_identity: str
    webots_version: str
    seed: int
    scenario: str
    observation_freshness_max_s: float
    urdf_path: Path
    registration_path: Path
    robot_config_path: Path
    required_arm_links: tuple[str, ...]
    swept_arm_certificate: Path | None
    pose_solver_binding: str
    safety_binding: str
    verification_binding: str
    geometry_binding: str
    diagnostic_mode: str
    no_actuation: bool
    scored_evaluation_allowed: bool
    evidence_collection_split: str
    source_path: Path


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Read-only backend facts collected by the backend-boundary adapter."""

    ping: Mapping[str, object]
    observation: Observation
    ground_truth: GroundTruth
    identity: Mapping[str, object]
    captured_at_utc: str
    age_s: float
    dependency_availability: Mapping[str, bool]
    call_counts: Mapping[str, int]


@dataclass(frozen=True)
class ConfigPreflight:
    blockers: tuple[str, ...]
    hashes: Mapping[str, str | None]
    motion_blockers: tuple[str, ...]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        raise ValueError(f"{name} has missing={missing} and unexpected={unexpected}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _sha256_text(value: object, name: str) -> str:
    text = _text(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return text


def _endpoint(value: object, name: str) -> str:
    text = _text(value, name)
    if not text.startswith("tcp://") or text.count(":") != 2:
        raise ValueError(f"{name} must be a tcp://host:port endpoint")
    port_text = text.rsplit(":", 1)[1]
    if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
        raise ValueError(f"{name} port must be in [1,65535]")
    return text


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _path(value: object, name: str, source: Path) -> Path:
    text = _text(value, name)
    candidate = Path(text).expanduser()
    return (
        (source.parent / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_webots_diagnostic_config(path: Path) -> WebotsDiagnosticConfig:
    """Load the strict read-only config; reject scoring, TEST/TRAIN and scientific parameters."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Webots diagnostic config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "Webots diagnostic config")
    unexpected = sorted(set(root) & _FORBIDDEN_TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(f"Webots diagnostic config forbids keys: {unexpected}")
    _keys(root, _TOP_LEVEL_KEYS, "Webots diagnostic config")

    if root["schema_version"] != WEBOTS_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unsupported Webots diagnostic schema_version")
    if root["status"] != WEBOTS_DIAGNOSTIC_STATUS:
        raise ValueError(f"Webots diagnostic status must be '{WEBOTS_DIAGNOSTIC_STATUS}'")
    if root["split"] != FROZEN_DIAGNOSTIC_SPLIT:
        raise ValueError("Webots read-only diagnostic is VALIDATION-only")
    if root["state_source"] != FROZEN_STATE_SOURCE:
        raise ValueError("Webots diagnostic state source must be declared as oracle")
    mode = _text(root["mode"], "mode")
    if mode not in ("launch", "connect"):
        raise ValueError("Webots diagnostic mode must be 'launch' or 'connect'")

    backend = _mapping(root["backend"], "backend")
    _keys(backend, frozenset({"kind", "declaration"}), "backend")
    if _text(backend["kind"], "backend.kind") != "webots":
        raise ValueError("this diagnostic requires backend.kind='webots'")
    declaration = _text(backend["declaration"], "backend.declaration")
    if "<" in declaration or ">" in declaration:
        raise ValueError("backend.declaration is still a blocked template placeholder")

    diagnostic = _mapping(root["diagnostic"], "diagnostic")
    _keys(diagnostic, _DIAGNOSTIC_KEYS, "diagnostic")
    if diagnostic["mode"] != WEBOTS_DIAGNOSTIC_MODE:
        raise ValueError(f"diagnostic.mode must be '{WEBOTS_DIAGNOSTIC_MODE}'")
    if diagnostic["no_actuation"] is not True:
        raise ValueError("diagnostic.no_actuation must be true")
    if diagnostic["scored_evaluation_allowed"] is not False:
        raise ValueError("a read-only diagnostic can never allow scored evaluation")
    if diagnostic["evidence_collection_split"] != FROZEN_DIAGNOSTIC_SPLIT:
        raise ValueError("diagnostic evidence collection is VALIDATION-only")

    geometry = _mapping(root["geometry"], "geometry")
    _keys(geometry, _GEOMETRY_KEYS, "geometry")
    raw_links = geometry["required_arm_links"]
    links = (
        tuple(
            _text(value, "geometry.required_arm_links")
            for value in cast(Sequence[object], raw_links)
        )
        if isinstance(raw_links, Sequence) and not isinstance(raw_links, (str, bytes))
        else ()
    )
    if not links or len(set(links)) != len(links):
        raise ValueError("geometry.required_arm_links must be unique and nonempty")
    certificate = geometry["swept_arm_certificate"]
    if certificate is not None:
        certificate = _path(certificate, "geometry.swept_arm_certificate", source)

    dependencies = _mapping(root["dependencies"], "dependencies")
    _keys(dependencies, _DEPENDENCY_KEYS, "dependencies")

    return WebotsDiagnosticConfig(
        schema_version=WEBOTS_DIAGNOSTIC_SCHEMA_VERSION,
        status=WEBOTS_DIAGNOSTIC_STATUS,
        split=FROZEN_DIAGNOSTIC_SPLIT,
        state_source=FROZEN_STATE_SOURCE,
        mode=mode,
        backend_declaration=declaration,
        webots=_text(root["webots"], "webots"),
        world=_path(root["world"], "world", source),
        world_sha256=_sha256_text(root["world_sha256"], "world_sha256"),
        endpoint=_endpoint(root["endpoint"], "endpoint"),
        target_object=_text(root["target_object"], "target_object"),
        robot_identity=_text(root["robot_identity"], "robot_identity"),
        webots_version=_text(root["webots_version"], "webots_version"),
        seed=_nonnegative_int(root["seed"], "seed"),
        scenario=_text(root["scenario"], "scenario"),
        observation_freshness_max_s=_positive_float(
            root["observation_freshness_max_s"], "observation_freshness_max_s"
        ),
        urdf_path=_path(geometry["urdf"], "geometry.urdf", source),
        registration_path=_path(geometry["registration"], "geometry.registration", source),
        robot_config_path=_path(geometry["robot_config"], "geometry.robot_config", source),
        required_arm_links=links,
        swept_arm_certificate=certificate,
        pose_solver_binding=_text(
            dependencies["pose_solver_binding"], "dependencies.pose_solver_binding"
        ),
        safety_binding=_text(dependencies["safety_binding"], "dependencies.safety_binding"),
        verification_binding=_text(
            dependencies["verification_binding"], "dependencies.verification_binding"
        ),
        geometry_binding=_text(dependencies["geometry_binding"], "dependencies.geometry_binding"),
        diagnostic_mode=WEBOTS_DIAGNOSTIC_MODE,
        no_actuation=True,
        scored_evaluation_allowed=False,
        evidence_collection_split=FROZEN_DIAGNOSTIC_SPLIT,
        source_path=source,
    )


def preflight_webots_diagnostic(config: WebotsDiagnosticConfig) -> ConfigPreflight:
    """Verify local geometry/config coverage before any simulator process is started."""
    blockers: list[str] = []
    paths = {
        "world": config.world,
        "urdf": config.urdf_path,
        "registration": config.registration_path,
        "robot_config": config.robot_config_path,
        "config": config.source_path,
    }
    if config.swept_arm_certificate is not None:
        paths["swept_arm_certificate"] = config.swept_arm_certificate
    hashes: dict[str, str | None] = {}
    for name, path in paths.items():
        digest = _sha256_file(path)
        hashes[name] = digest
        if digest is None:
            blockers.append(f"missing geometry/config file: {name}={path}")
    if hashes.get("world") is not None and hashes["world"] != config.world_sha256:
        blockers.append("world SHA-256 does not match the reviewed diagnostic config")
    motion_blockers: list[str] = ["diagnostic_no_actuation_mode"]
    if config.swept_arm_certificate is None or hashes.get("swept_arm_certificate") is None:
        motion_blockers.append("unverified-arm-sweep")
    return ConfigPreflight(tuple(blockers), hashes, tuple(motion_blockers))


def ensure_fresh_output(out_dir: Path) -> None:
    """Refuse an existing output path before any session, planner or evidence write."""
    if Path(out_dir).exists():
        raise FileExistsError(f"prior evidence directory must not be overwritten: {out_dir}")


def _finite_pose(pose: Pose3D) -> bool:
    values = (pose.position.x, pose.position.y, pose.position.z, *pose.orientation)
    return all(math.isfinite(float(value)) for value in values)


def _observation_blockers(
    config: WebotsDiagnosticConfig, snapshot: DiagnosticSnapshot
) -> list[str]:
    blockers: list[str] = []
    observation = snapshot.observation
    if not math.isfinite(observation.t) or observation.t < 0.0:
        blockers.append("observation timestamp is not finite and nonnegative")
    if not math.isfinite(snapshot.age_s) or snapshot.age_s < 0.0:
        blockers.append("observation age is not finite and nonnegative")
    if snapshot.age_s > config.observation_freshness_max_s:
        blockers.append("observation is stale")
    robot = observation.robot
    if not all(math.isfinite(float(value)) for value in robot.joint_pos.values()):
        blockers.append("robot joint positions are not finite")
    if not all(math.isfinite(float(value)) for value in robot.joint_vel.values()):
        blockers.append("robot joint velocities are not finite")
    if not math.isfinite(float(robot.gripper_opening)):
        blockers.append("gripper opening is not finite")
    if not _finite_pose(robot.ee_pose):
        blockers.append("end-effector pose is not finite")
    if robot.base_pose_world is not None and not _finite_pose(robot.base_pose_world):
        blockers.append("world-frame base pose is not finite")
    target = snapshot.ground_truth.object_poses.get(config.target_object)
    if target is None or not _finite_pose(target):
        blockers.append("declared oracle target pose is missing or not finite")
    return blockers


def _snapshot_blockers(config: WebotsDiagnosticConfig, snapshot: DiagnosticSnapshot) -> list[str]:
    blockers: list[str] = []
    session = snapshot.ping.get("session_id")
    if not isinstance(session, Mapping) or not {"pid", "start_time", "nonce"}.issubset(session):
        blockers.append("simulator session identity is missing")
    if snapshot.ping.get("world_sha256") != config.world_sha256:
        blockers.append("running Webots world SHA-256 does not match the reviewed config")
    if snapshot.identity.get("simulator") != "webots":
        blockers.append("backend identity is not Webots")
    if snapshot.identity.get("robot_name") != config.robot_identity:
        blockers.append("robot identity differs from the reviewed declaration")
    if snapshot.identity.get("webots_version") != config.webots_version:
        blockers.append("Webots version differs from the reviewed declaration")
    if snapshot.identity.get("first_command_step_count") is not None:
        blockers.append("an actuator command was issued in this backend session")
    if not all(snapshot.dependency_availability.values()):
        blockers.append("required solver/geometry dependency is unavailable")
    for name in _FORBIDDEN_CALLS:
        if snapshot.call_counts.get(name, 0) != 0:
            blockers.append(f"forbidden backend call was attempted: {name}")
    blockers.extend(_observation_blockers(config, snapshot))
    return blockers


def _pose_payload(pose: Pose3D | None) -> dict[str, object] | None:
    if pose is None:
        return None
    return {
        "position": [pose.position.x, pose.position.y, pose.position.z],
        "orientation": list(pose.orientation),
    }


def finalize_webots_diagnostic(
    config: WebotsDiagnosticConfig, preflight: ConfigPreflight, snapshot: DiagnosticSnapshot
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return exclusive, non-scored diagnostic, state and provenance artifacts."""
    blockers = [*preflight.blockers, *_snapshot_blockers(config, snapshot)]
    ready = not blockers
    report: dict[str, object] = {
        "schema_version": WEBOTS_DIAGNOSTIC_SCHEMA_VERSION,
        "outcome": "PASS" if ready else "FAILED",
        "diagnostic_mode": WEBOTS_DIAGNOSTIC_MODE,
        "scored_evaluation_allowed": False,
        "scored_evaluator": None,
        "scored_results_written": False,
        "frozen_campaign_modified": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "state_source": config.state_source,
        "oracle_declaration": config.backend_declaration,
        "oracle_equivalent_to_perception": False,
        "split": config.split,
        "backend_declaration": config.backend_declaration,
        "backend_kind": "webots",
        "launch_mode": config.mode,
        "endpoint": snapshot.ping.get("endpoint"),
        "simulator_session_id": snapshot.ping.get("session_id"),
        "simulator_identity": dict(snapshot.identity),
        "simulator_version": snapshot.identity.get("webots_version"),
        "robot_identity_declared": config.robot_identity,
        "observation_timestamp_s": snapshot.observation.t,
        "observation_age_s": snapshot.age_s,
        "observation_freshness_limit_s": config.observation_freshness_max_s,
        "observation_valid": ready and not preflight.blockers,
        "geometry_hashes": dict(preflight.hashes),
        "required_arm_links": list(config.required_arm_links),
        "swept_arm_certificate": (
            None if config.swept_arm_certificate is None else str(config.swept_arm_certificate)
        ),
        "motion_allowed": False,
        "motion_blockers": list(preflight.motion_blockers),
        "read_only_inspection_allowed": not preflight.blockers,
        "no_actuation": True,
        "backend_apply_calls": snapshot.call_counts.get("apply", 0),
        "backend_step_calls": snapshot.call_counts.get("step", 0),
        "backend_reset_calls": snapshot.call_counts.get("reset", 0),
        "backend_export_calls": snapshot.call_counts.get("export_kinematics", 0),
        "actuator_commands": [],
        "dependency_availability": dict(snapshot.dependency_availability),
        "safety_check_readiness": all(snapshot.dependency_availability.values()),
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
        "oracle_target_object": config.target_object,
        "oracle_target_pose": _pose_payload(
            snapshot.ground_truth.object_poses.get(config.target_object)
        ),
    }
    provenance = {
        "diagnostic_only": True,
        "report_sha256": AUTHORITATIVE_REPORT_SHA256,
        "config_sha256": preflight.hashes.get("config"),
        "world_sha256": preflight.hashes.get("world"),
        "urdf_sha256": preflight.hashes.get("urdf"),
        "registration_sha256": preflight.hashes.get("registration"),
        "robot_config_sha256": preflight.hashes.get("robot_config"),
        "session_id": snapshot.ping.get("session_id"),
        "captured_at_utc": snapshot.captured_at_utc,
        "frozen_campaign_modified": False,
        "scored_results_written": False,
    }
    return report, state, provenance


def failed_webots_diagnostic(
    config: WebotsDiagnosticConfig,
    preflight: ConfigPreflight,
    reason: str,
    dependency_availability: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Explicit fail-closed report used before any backend session is created."""
    availability = dict(dependency_availability or {})
    return {
        "schema_version": WEBOTS_DIAGNOSTIC_SCHEMA_VERSION,
        "outcome": "FAILED",
        "diagnostic_mode": WEBOTS_DIAGNOSTIC_MODE,
        "scored_evaluation_allowed": False,
        "scored_evaluator": None,
        "scored_results_written": False,
        "frozen_campaign_modified": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "state_source": config.state_source,
        "backend_declaration": config.backend_declaration,
        "geometry_hashes": dict(preflight.hashes),
        "motion_allowed": False,
        "motion_blockers": list(preflight.motion_blockers),
        "read_only_inspection_allowed": False,
        "no_actuation": True,
        "backend_apply_calls": 0,
        "backend_step_calls": 0,
        "backend_reset_calls": 0,
        "backend_export_calls": 0,
        "actuator_commands": [],
        "dependency_availability": availability,
        "safety_check_readiness": all(availability.values()) if availability else False,
        "blockers": [reason, *preflight.blockers],
    }


def write_diagnostic_evidence(
    out_dir: Path,
    report: Mapping[str, object],
    state: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> tuple[Path, ...]:
    """Write into the caller-created exclusive directory; never overwrite prior evidence."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = [out / "diagnostic.json"]
    if files[0].exists():
        raise FileExistsError(f"prior diagnostic evidence must not be overwritten: {files[0]}")
    with files[0].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for name, payload in (("state.json", state), ("provenance.json", provenance)):
        if payload is None:
            continue
        path = out / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        files.append(path)
    return tuple(files)


__all__ = [
    "ConfigPreflight",
    "DiagnosticSnapshot",
    "WebotsDiagnosticConfig",
    "ensure_fresh_output",
    "failed_webots_diagnostic",
    "finalize_webots_diagnostic",
    "load_webots_diagnostic_config",
    "preflight_webots_diagnostic",
    "write_diagnostic_evidence",
]
