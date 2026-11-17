"""Pure evidence rules for the bounded, local-only full-arm geometry diagnostic.

This module imports no simulator, backend, ROS or MuJoCo package. Geometry observations are
collected at the backend boundary and injected as a snapshot. Nothing here launches Webots,
solves IK, applies a command, advances physics, scores an episode, or reads TEST/TRAIN evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

GEOMETRY_DIAGNOSTIC_SCHEMA_VERSION = 1
GEOMETRY_DIAGNOSTIC_STATUS = "diagnostic_uncalibrated"
GEOMETRY_DIAGNOSTIC_MODE = "local_file_inspection"
FROZEN_DIAGNOSTIC_SPLIT = "validation"
FROZEN_STATE_SOURCE = "oracle"
AUTHORITATIVE_REPORT_SHA256 = "23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b"
_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics", "inspect", "get_ground_truth")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "split",
        "state_source",
        "mode",
        "backend",
        "geometry",
        "hashes",
        "diagnostic",
    }
)
_FORBIDDEN_TOP_LEVEL_KEYS = frozenset(
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
        "webots",
        "endpoint",
        "seed",
        "episodes",
    }
)
_GEOMETRY_KEYS = frozenset(
    {
        "urdf",
        "registration",
        "robot_config",
        "world",
        "required_arm_links",
        "required_objects",
        "permitted_contact_objects",
        "swept_arm_certificate",
    }
)
_HASH_KEYS = frozenset(
    {"urdf_sha256", "registration_sha256", "robot_config_sha256", "world_sha256"}
)
_DIAGNOSTIC_KEYS = frozenset(
    {
        "local_files_only",
        "no_webots_launch",
        "no_actuation",
        "no_ik",
        "no_trajectory",
        "scored_evaluation_allowed",
        "evidence_collection_split",
    }
)


@dataclass(frozen=True)
class GeometryDiagnosticConfig:
    """Strict local-file geometry contract; no scientific or motion parameters are present."""

    schema_version: int
    status: str
    split: str
    state_source: str
    mode: str
    backend_kind: str
    backend_declaration: str
    urdf_path: Path
    registration_path: Path
    robot_config_path: Path
    world_path: Path
    required_arm_links: tuple[str, ...]
    required_objects: frozenset[str]
    permitted_contact_objects: frozenset[str]
    swept_arm_certificate: Path | None
    urdf_sha256: str
    registration_sha256: str
    robot_config_sha256: str
    world_sha256: str
    source_path: Path


@dataclass(frozen=True)
class GeometryEvidence:
    """Observed local file facts supplied by the backend-boundary collector."""

    world_inventory: Mapping[str, object]
    urdf_inventory: Mapping[str, object]
    collision_capsules: Mapping[str, object]
    hashes: Mapping[str, str]
    registration_valid: bool
    external_proto_count: int
    self_collision_enabled: bool | None
    call_counts: Mapping[str, int]


@dataclass(frozen=True)
class GeometryPreflight:
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


def _path(value: object, name: str, source: Path) -> Path:
    candidate = Path(_text(value, name)).expanduser()
    return (
        (source.parent / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _string_set(value: object, name: str) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of strings")
    result = frozenset(_text(item, name) for item in value)
    if len(result) != len(list(value)):
        raise ValueError(f"{name} must contain unique strings")
    return result


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_geometry_diagnostic_config(path: Path) -> GeometryDiagnosticConfig:
    """Load the strict local-only contract; reject scoring, motion and simulator launch fields."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"geometry diagnostic config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "geometry diagnostic config")
    forbidden = sorted(set(root) & _FORBIDDEN_TOP_LEVEL_KEYS)
    if forbidden:
        raise ValueError(f"geometry diagnostic config forbids keys: {forbidden}")
    _keys(root, _TOP_LEVEL_KEYS, "geometry diagnostic config")
    if root["schema_version"] != GEOMETRY_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unsupported geometry diagnostic schema_version")
    if root["status"] != GEOMETRY_DIAGNOSTIC_STATUS:
        raise ValueError(f"geometry diagnostic status must be '{GEOMETRY_DIAGNOSTIC_STATUS}'")
    if root["split"] != FROZEN_DIAGNOSTIC_SPLIT:
        raise ValueError("geometry diagnostic is VALIDATION-only")
    if root["state_source"] != FROZEN_STATE_SOURCE:
        raise ValueError("geometry diagnostic state source must be declared as oracle")
    if root["mode"] != GEOMETRY_DIAGNOSTIC_MODE:
        raise ValueError(f"geometry diagnostic mode must be '{GEOMETRY_DIAGNOSTIC_MODE}'")

    backend = _mapping(root["backend"], "backend")
    _keys(backend, frozenset({"kind", "declaration"}), "backend")
    if _text(backend["kind"], "backend.kind") != "local_geometry_files":
        raise ValueError("geometry diagnostic requires backend.kind='local_geometry_files'")
    declaration = _text(backend["declaration"], "backend.declaration")
    if "<" in declaration or ">" in declaration:
        raise ValueError("backend.declaration is still a blocked template placeholder")

    diagnostic = _mapping(root["diagnostic"], "diagnostic")
    _keys(diagnostic, _DIAGNOSTIC_KEYS, "diagnostic")
    for key in ("local_files_only", "no_webots_launch", "no_actuation", "no_ik", "no_trajectory"):
        if diagnostic[key] is not True:
            raise ValueError(f"diagnostic.{key} must be true")
    if diagnostic["scored_evaluation_allowed"] is not False:
        raise ValueError("a geometry diagnostic can never allow scored evaluation")
    if diagnostic["evidence_collection_split"] != FROZEN_DIAGNOSTIC_SPLIT:
        raise ValueError("geometry diagnostic evidence collection is VALIDATION-only")

    geometry = _mapping(root["geometry"], "geometry")
    _keys(geometry, _GEOMETRY_KEYS, "geometry")
    links = tuple(
        _text(value, "geometry.required_arm_links")
        for value in cast(Sequence[object], geometry["required_arm_links"])
    )
    if not links or len(set(links)) != len(links):
        raise ValueError("geometry.required_arm_links must be unique and nonempty")
    required_objects = _string_set(geometry["required_objects"], "geometry.required_objects")
    permitted = _string_set(
        geometry["permitted_contact_objects"], "geometry.permitted_contact_objects"
    )
    certificate = geometry["swept_arm_certificate"]
    certificate_path = (
        None
        if certificate is None
        else _path(certificate, "geometry.swept_arm_certificate", source)
    )
    hashes = _mapping(root["hashes"], "hashes")
    _keys(hashes, _HASH_KEYS, "hashes")
    return GeometryDiagnosticConfig(
        schema_version=GEOMETRY_DIAGNOSTIC_SCHEMA_VERSION,
        status=GEOMETRY_DIAGNOSTIC_STATUS,
        split=FROZEN_DIAGNOSTIC_SPLIT,
        state_source=FROZEN_STATE_SOURCE,
        mode=GEOMETRY_DIAGNOSTIC_MODE,
        backend_kind=_text(backend["kind"], "backend.kind"),
        backend_declaration=declaration,
        urdf_path=_path(geometry["urdf"], "geometry.urdf", source),
        registration_path=_path(geometry["registration"], "geometry.registration", source),
        robot_config_path=_path(geometry["robot_config"], "geometry.robot_config", source),
        world_path=_path(geometry["world"], "geometry.world", source),
        required_arm_links=links,
        required_objects=required_objects,
        permitted_contact_objects=permitted,
        swept_arm_certificate=certificate_path,
        urdf_sha256=_sha256_text(hashes["urdf_sha256"], "hashes.urdf_sha256"),
        registration_sha256=_sha256_text(
            hashes["registration_sha256"], "hashes.registration_sha256"
        ),
        robot_config_sha256=_sha256_text(
            hashes["robot_config_sha256"], "hashes.robot_config_sha256"
        ),
        world_sha256=_sha256_text(hashes["world_sha256"], "hashes.world_sha256"),
        source_path=source,
    )


def preflight_geometry_diagnostic(config: GeometryDiagnosticConfig) -> GeometryPreflight:
    """Verify pinned local files before any evidence is collected."""
    paths = {
        "urdf": config.urdf_path,
        "registration": config.registration_path,
        "robot_config": config.robot_config_path,
        "world": config.world_path,
        "config": config.source_path,
    }
    if config.swept_arm_certificate is not None:
        paths["swept_arm_certificate"] = config.swept_arm_certificate
    hashes: dict[str, str | None] = {}
    blockers: list[str] = []
    for name, path in paths.items():
        digest = _sha256_file(path)
        hashes[name] = digest
        if digest is None:
            blockers.append(f"missing geometry/config file: {name}={path}")
    for name, expected in (
        ("urdf", config.urdf_sha256),
        ("registration", config.registration_sha256),
        ("robot_config", config.robot_config_sha256),
        ("world", config.world_sha256),
    ):
        if hashes.get(name) is not None and hashes[name] != expected:
            blockers.append(f"{name} SHA-256 does not match the reviewed diagnostic config")
    motion_blockers = [
        "diagnostic_no_webots_launch",
        "diagnostic_no_actuation",
        "unverified-arm-sweep",
    ]
    return GeometryPreflight(tuple(blockers), hashes, tuple(motion_blockers))


def ensure_fresh_output(out_dir: Path) -> None:
    """Refuse an existing output path before any evidence write."""
    if Path(out_dir).exists():
        raise FileExistsError(f"prior evidence directory must not be overwritten: {out_dir}")


def _capsule_rows(evidence: GeometryEvidence) -> tuple[dict[str, object], ...]:
    links = evidence.collision_capsules.get("links", [])
    if not isinstance(links, Sequence):
        return ()
    rows: list[dict[str, object]] = []
    for row in links:
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return tuple(rows)


def finalize_geometry_diagnostic(
    config: GeometryDiagnosticConfig, preflight: GeometryPreflight, evidence: GeometryEvidence
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return exclusive, non-scored geometry consistency, state and provenance artifacts."""
    blockers = list(preflight.blockers)
    for name in _FORBIDDEN_CALLS:
        if evidence.call_counts.get(name, 0) != 0:
            blockers.append(f"forbidden backend capability was invoked: {name}")
    if not evidence.registration_valid:
        blockers.append("corrected URDF-to-base registration is not a valid rigid transform")

    capsules = _capsule_rows(evidence)
    covered_links = {str(row.get("link")) for row in capsules if row.get("collision_count", 0)}
    missing_links = sorted(set(config.required_arm_links) - covered_links)
    if missing_links:
        blockers.append(f"missing required collision geometry: {missing_links}")
    unsupported = sorted(
        str(shape)
        for row in capsules
        for shape in cast(Sequence[object], row.get("unsupported_shapes", ()))
    )
    if unsupported:
        blockers.append(f"unsupported URDF collision shapes: {unsupported}")

    coverage_gaps: list[str] = []
    if evidence.external_proto_count:
        coverage_gaps.append(
            f"world references {evidence.external_proto_count} external PROTO definition(s); "
            "source geometry is not locally resolved"
        )
    if evidence.self_collision_enabled is None:
        coverage_gaps.append("robot self-collision setting was not observed")
    elif evidence.self_collision_enabled is False:
        coverage_gaps.append("robot self-collision is disabled; no approved exclusion pairs exist")
    if config.swept_arm_certificate is None:
        coverage_gaps.append("no owner-reviewed full-arm swept certificate")
    if config.permitted_contact_objects:
        coverage_gaps.append(
            f"permitted contact exceptions are declared but unapproved: "
            f"{sorted(config.permitted_contact_objects)}"
        )
    coverage_gaps.extend(
        [
            "floor Plane primitives are outside the capsule/AABB checker",
            "table nested solids are not exposed in the live geometry export",
            "finger, gripper and carried-object envelopes are not approved collision geometry",
        ]
    )

    speed_bound = evidence.collision_capsules.get("speed_bound_m_s")
    finite_speed = (
        isinstance(speed_bound, (int, float))
        and not isinstance(speed_bound, bool)
        and math.isfinite(float(speed_bound))
        and float(speed_bound) > 0
    )
    raw_count = evidence.collision_capsules.get("capsule_count", 0)
    capsule_count = (
        raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) else 0
    )
    if not finite_speed:
        blockers.append("derived point-speed certificate is not finite and positive")
    margin_ok = bool(evidence.collision_capsules.get("margin_positive")) and bool(
        evidence.collision_capsules.get("substeps_valid")
    )
    if not margin_ok:
        blockers.append("sweep margin or interpolation substeps are invalid")

    report: dict[str, object] = {
        "schema_version": GEOMETRY_DIAGNOSTIC_SCHEMA_VERSION,
        "outcome": "PASS" if not blockers else "FAILED",
        "diagnostic_mode": GEOMETRY_DIAGNOSTIC_MODE,
        "backend_kind": config.backend_kind,
        "backend_declaration": config.backend_declaration,
        "local_files_only": True,
        "no_webots_launch": True,
        "no_actuation": True,
        "no_ik": True,
        "no_trajectory": True,
        "scored_evaluation_allowed": False,
        "scored_results_written": False,
        "frozen_campaign_modified": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "split": config.split,
        "state_source": config.state_source,
        "oracle_equivalent_to_perception": False,
        "geometry_hashes": dict(preflight.hashes),
        "required_arm_links": list(config.required_arm_links),
        "covered_arm_links": sorted(covered_links),
        "missing_arm_links": missing_links,
        "collision_capsule_count": capsule_count,
        "collision_capsules": list(capsules),
        "unsupported_collision_shapes": unsupported,
        "required_objects": sorted(config.required_objects),
        "permitted_contact_objects": sorted(config.permitted_contact_objects),
        "point_speed_bound_m_s": speed_bound,
        "margin_positive": bool(evidence.collision_capsules.get("margin_positive")),
        "interpolation_substeps": evidence.collision_capsules.get("substeps"),
        "substeps_valid": bool(evidence.collision_capsules.get("substeps_valid")),
        "registration_valid": evidence.registration_valid,
        "external_proto_count": evidence.external_proto_count,
        "self_collision_enabled": evidence.self_collision_enabled,
        "world_inventory": dict(evidence.world_inventory),
        "urdf_inventory": dict(evidence.urdf_inventory),
        "coverage_gaps": coverage_gaps,
        "forbidden_call_counts": {
            name: evidence.call_counts.get(name, 0) for name in _FORBIDDEN_CALLS
        },
        "motion_allowed": False,
        "motion_blockers": list(preflight.motion_blockers) + coverage_gaps,
        "engineering_gates": {
            "complete_required_arm_collision_capsules": not missing_links and not unsupported,
            # A diagnostic PASS means evidence was collected, not that all geometry is covered.
            "complete_relevant_collision_geometry": not coverage_gaps,
            "correct_coordinate_transforms": evidence.registration_valid,
            "conservative_clearance_margin": bool(
                evidence.collision_capsules.get("margin_positive")
            ),
            "supported_contact_exceptions": not config.permitted_contact_objects,
            "valid_transition_interpolation": bool(
                evidence.collision_capsules.get("substeps_valid")
            ),
            "reviewed_reference_cases": False,
            "owner_certificate": False,
            "motion_ready": False,
        },
        "owner_safety_approval_present": False,
        "motion_readiness": "NOT_READY",
        "blockers": blockers,
    }
    state = {
        "inspection_only": True,
        "world_sha256": preflight.hashes.get("world"),
        "urdf_sha256": preflight.hashes.get("urdf"),
        "registration_sha256": preflight.hashes.get("registration"),
        "robot_config_sha256": preflight.hashes.get("robot_config"),
        "required_arm_links": list(config.required_arm_links),
        "covered_arm_links": sorted(covered_links),
        "self_collision_enabled": evidence.self_collision_enabled,
        "external_proto_count": evidence.external_proto_count,
    }
    provenance = {
        "diagnostic_only": True,
        "local_file_inspection": True,
        "report_sha256": AUTHORITATIVE_REPORT_SHA256,
        "config_sha256": preflight.hashes.get("config"),
        "geometry_hashes": dict(preflight.hashes),
        "frozen_campaign_modified": False,
        "scored_results_written": False,
    }
    return report, state, provenance


def failed_geometry_diagnostic(
    config: GeometryDiagnosticConfig, preflight: GeometryPreflight, reason: str
) -> dict[str, object]:
    """Explicit fail-closed report used before any local geometry evidence is collected."""
    return {
        "schema_version": GEOMETRY_DIAGNOSTIC_SCHEMA_VERSION,
        "outcome": "FAILED",
        "diagnostic_mode": GEOMETRY_DIAGNOSTIC_MODE,
        "local_files_only": True,
        "no_webots_launch": True,
        "no_actuation": True,
        "no_ik": True,
        "no_trajectory": True,
        "scored_evaluation_allowed": False,
        "frozen_campaign_modified": False,
        "test_split_accessed": False,
        "train_split_accessed": False,
        "split": config.split,
        "state_source": config.state_source,
        "geometry_hashes": dict(preflight.hashes),
        "motion_allowed": False,
        "motion_blockers": list(preflight.motion_blockers),
        "engineering_gates": {},
        "owner_safety_approval_present": False,
        "motion_readiness": "NOT_READY",
        "blockers": [reason, *preflight.blockers],
    }


def write_geometry_evidence(
    out_dir: Path,
    report: Mapping[str, object],
    state: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> tuple[Path, ...]:
    """Write into the caller-created exclusive directory; never overwrite prior evidence."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = [out / "diagnostic.json"]
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
    "GeometryDiagnosticConfig",
    "GeometryEvidence",
    "GeometryPreflight",
    "ensure_fresh_output",
    "failed_geometry_diagnostic",
    "finalize_geometry_diagnostic",
    "load_geometry_diagnostic_config",
    "preflight_geometry_diagnostic",
    "write_geometry_evidence",
]
