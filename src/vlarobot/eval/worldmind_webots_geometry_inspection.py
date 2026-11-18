"""Pure evidence rules for the dedicated read-only Webots geometry inspection.

This module imports no simulator, controller, ROS or MuJoCo package. It reuses the file-only pinned
PROTO source loader. The live export is collected by
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

import numpy as np
import yaml

from vlarobot.backends.webots.proto_sources import load_proto_source_manifest
from vlarobot.core.types import Pose3D
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
from vlarobot.world_model.scene_geometry import axis_angle_matrix, load_scene_collision_catalog

GEOMETRY_INSPECTION_SCHEMA_VERSION = 2
TABLE_TARGET_SCHEMA_VERSION = 1
TABLE_SOURCE_NAMES: tuple[str, ...] = (
    "main frame",
    "foot 1",
    "foot 2",
    "foot 3",
    "foot 4",
)
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
        "table_targets",
    }
)
_TABLE_TARGET_KEYS = frozenset(
    {
        "parent_type",
        "nested_type",
        "bounding_shape_type",
        "source_names",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "size_tolerance_m",
        "transform_translation_tolerance_m",
        "transform_rotation_tolerance_rad",
    }
)
_TABLE_OBSERVATION_KEYS = frozenset({"schema_version", "rows", "errors", "interface"})
_TABLE_ROW_KEYS = frozenset(
    {
        "source_name",
        "status",
        "reason",
        "source_protos",
        "runtime_node",
        "hierarchy",
        "evaluated",
        "transforms",
        "unsupported_reasons",
    }
)
_TABLE_VERDICTS = ("Matched", "Mismatched", "Missing", "Unsupported")


def _table_identity_matches(value: object, expected: str) -> bool:
    return isinstance(value, Mapping) and expected in {
        value.get("type"),
        value.get("proto_type"),
        value.get("base_type"),
    }


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
class TableTargetConfig:
    """Strict selector and explicit numeric tolerances for the five nested table boxes."""

    parent_type: str
    nested_type: str
    bounding_shape_type: str
    source_names: tuple[str, ...]
    translation_tolerance_m: float
    rotation_tolerance_rad: float
    size_tolerance_m: float
    transform_translation_tolerance_m: float
    transform_rotation_tolerance_rad: float


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
    table_targets: TableTargetConfig


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
    targets_raw = _mapping(section.get("table_targets"), "geometry_inspection.table_targets")
    _keys(targets_raw, _TABLE_TARGET_KEYS, "geometry_inspection.table_targets")
    parent_type = _text(targets_raw["parent_type"], "geometry_inspection.table_targets.parent_type")
    nested_type = _text(targets_raw["nested_type"], "geometry_inspection.table_targets.nested_type")
    shape_type = _text(
        targets_raw["bounding_shape_type"],
        "geometry_inspection.table_targets.bounding_shape_type",
    )
    if (parent_type, nested_type, shape_type) != ("Table", "SolidBox", "Box"):
        raise ValueError("table_targets selector must be Table/SolidBox/Box")
    raw_names = targets_raw["source_names"]
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError("geometry_inspection.table_targets.source_names must be a list")
    source_names = tuple(
        _text(item, "geometry_inspection.table_targets.source_names[]") for item in raw_names
    )
    if source_names != TABLE_SOURCE_NAMES:
        raise ValueError("table_targets.source_names must pin the five Table box names in order")
    tolerance_values: dict[str, float] = {}
    for name in (
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "size_tolerance_m",
        "transform_translation_tolerance_m",
        "transform_rotation_tolerance_rad",
    ):
        value = _finite_number(targets_raw[name])
        if value is None or value < 0.0:
            raise ValueError(
                f"geometry_inspection.table_targets.{name} must be finite and nonnegative"
            )
        tolerance_values[name] = value
    table_targets = TableTargetConfig(
        parent_type=parent_type,
        nested_type=nested_type,
        bounding_shape_type=shape_type,
        source_names=source_names,
        **tolerance_values,
    )
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
        table_targets=table_targets,
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


def _float_sequence(value: object) -> tuple[float, ...] | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    numbers = [_finite_number(item) for item in value]
    if any(number is None for number in numbers):
        return None
    return tuple(float(number) for number in numbers if number is not None)


def _number_vector(value: object, size: int, name: str) -> tuple[float, ...] | None:
    numbers = _float_sequence(value)
    return numbers if numbers is not None and len(numbers) == size else None


def _pose_matrix(value: object, name: str) -> np.ndarray | None:
    numbers = _number_vector(value, 16, name)
    if numbers is None:
        return None
    matrix = np.asarray(numbers, dtype=float).reshape(4, 4)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0):
        return None
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0):
        return None
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-8):
        return None
    return matrix


def _pose4(rotation: object, translation: object) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return matrix


def pose3d_to_matrix(pose: Pose3D | None) -> np.ndarray | None:
    """Convert an observed xyzw ``Pose3D`` to a finite rigid transform, or None if invalid."""
    if pose is None:
        return None
    try:
        position = pose.position.as_array().reshape(3)
        orientation = np.asarray(pose.orientation, dtype=float).reshape(4)
    except Exception:
        return None
    if not np.isfinite(position).all() or not np.isfinite(orientation).all():
        return None
    norm = float(np.linalg.norm(orientation))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return None
    x, y, z, w = (float(value) for value in orientation)
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    matrix = _pose4(rotation, position)
    return _pose_matrix(matrix.reshape(16), "T_world_base_live")


def _snapshot_live_base(
    snapshot: GeometryInspectionSnapshot,
) -> tuple[np.ndarray | None, str | None]:
    observation = getattr(snapshot, "observation", None)
    robot = getattr(observation, "robot", None) if observation is not None else None
    pose = getattr(robot, "base_pose_world", None) if robot is not None else None
    matrix = pose3d_to_matrix(pose)
    if matrix is None:
        return None, "observed live robot-base pose is unavailable for runtime base comparison"
    return matrix, None


def _derive_live_base_expectation(
    expected: Mapping[str, object], T_world_base_live: np.ndarray
) -> Mapping[str, object] | None:
    world_translation = _number_vector(
        expected.get("world_translation_m"), 3, "world_translation_m"
    )
    world_rotation = expected.get("world_rotation_matrix")
    if world_translation is None or world_rotation is None:
        return None
    try:
        T_world_target = _pose4(
            np.asarray(world_rotation, dtype=float).reshape(3, 3), world_translation
        )
        T_base_live = np.linalg.inv(T_world_base_live) @ T_world_target
    except Exception:
        return None
    return {
        "base_translation_m": tuple(float(value) for value in T_base_live[:3, 3]),
        "base_rotation_matrix": [row[:] for row in T_base_live[:3, :3].tolist()],
        "T_world_base_live": [row[:] for row in T_world_base_live.tolist()],
        "derivation": "T_BT = inverse(T_WB_live_observed) @ T_WT_pinned_source",
    }


def _rotation_angle(observed: np.ndarray, expected: np.ndarray) -> float | None:
    rotation = observed[:3, :3] @ expected[:3, :3].T
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    return angle if math.isfinite(angle) else None


def table_source_provenance(config: GeometryInspectionConfig) -> dict[str, dict[str, str]]:
    """Resolve only the pinned Table/SolidBox source identities and hashes."""
    manifest = load_proto_source_manifest(
        config.proto_manifest_path,
        world_path=config.diagnostic.world,
        expected_world_sha256=config.diagnostic.world_sha256,
    )
    sources = {item.name: item for item in (*manifest.external, *manifest.nested)}
    provenance: dict[str, dict[str, str]] = {}
    for name in ("Table", "SolidBox"):
        source = sources.get(name)
        if source is None:
            raise ValueError(f"pinned PROTO source manifest has no {name} entry")
        provenance[name] = {
            "name": source.name,
            "url": source.url,
            "path": str(source.path),
            "sha256": source.sha256,
        }
    return provenance


def table_target_descriptor(config: GeometryInspectionConfig) -> dict[str, object]:
    """Build the selector-only descriptor forwarded to the supervisor-side probe."""
    targets = config.table_targets
    return {
        "schema_version": TABLE_TARGET_SCHEMA_VERSION,
        "parent_type": targets.parent_type,
        "nested_type": targets.nested_type,
        "bounding_shape_type": targets.bounding_shape_type,
        "source_names": list(targets.source_names),
        "source_protos": table_source_provenance(config),
    }


def _expected_table_geometry(config: GeometryInspectionConfig) -> dict[str, object]:
    """Derive expected five-box geometry from pinned sources; never from runtime observations."""
    from vlarobot.eval.worldmind_live_geometry_reconciliation import (
        expand_table_source_geometry,
    )

    if _sha256_file(config.scene_catalog_path) != config.scene_catalog_sha256:
        raise ValueError("scene_catalog SHA-256 does not match the reviewed geometry config")
    manifest = load_proto_source_manifest(
        config.proto_manifest_path,
        world_path=config.diagnostic.world,
        expected_world_sha256=config.diagnostic.world_sha256,
    )
    sources = {item.name: item for item in (*manifest.external, *manifest.nested)}
    catalog = load_scene_collision_catalog(config.scene_catalog_path)
    expansion, blockers = expand_table_source_geometry(
        config.diagnostic.world,
        sources["Table"],
        sources["SolidBox"],
        catalog,
        math.inf,
        False,
        config.diagnostic.world_sha256,
    )
    if blockers:
        raise ValueError(f"source table expansion is blocked: {'; '.join(blockers)}")
    # This reconciliation uses an unbounded source expansion only for arithmetic expectations.
    # Workspace reach fields would be meaningless here and cannot be serialized as evidence.
    expansion.pop("reach_bound_m", None)
    raw_boxes = expansion.get("boxes")
    if isinstance(raw_boxes, list):
        for box in raw_boxes:
            if isinstance(box, dict):
                box.pop("distance_to_base_origin_m", None)
                box.pop("reachability", None)
    return expansion


def _table_check(
    name: str,
    observed: object,
    expected: object,
    tolerance: float,
    *,
    metric: str,
    unit: str,
) -> dict[str, object]:
    observed_numbers = _float_sequence(observed)
    expected_numbers = _float_sequence(expected)
    if observed_numbers is None or expected_numbers is None:
        return {"name": name, "matched": False, "metric": metric, "reason": "missing observation"}
    observed_array = np.asarray(observed_numbers, dtype=float)
    expected_array = np.asarray(expected_numbers, dtype=float)
    if observed_array.shape != expected_array.shape or not np.isfinite(observed_array).all():
        return {"name": name, "matched": False, "metric": metric, "reason": "invalid observation"}
    error = float(np.max(np.abs(observed_array - expected_array)))
    return {
        "name": name,
        "matched": bool(error <= tolerance),
        "metric": metric,
        "unit": unit,
        "tolerance": tolerance,
        "max_abs_error": error,
    }


def _append_rotation_check(
    checks: list[dict[str, object]],
    name: str,
    observed_pose: np.ndarray | None,
    expected_rotation: object,
    tolerance: float,
) -> None:
    if observed_pose is None or expected_rotation is None:
        checks.append({"name": name, "matched": False, "reason": "missing observation"})
        return
    try:
        expected_matrix = np.asarray(expected_rotation, dtype=float).reshape(3, 3)
    except Exception:
        checks.append({"name": name, "matched": False, "reason": "invalid rotation"})
        return
    angle = _rotation_angle(observed_pose, expected_matrix)
    if angle is None:
        checks.append({"name": name, "matched": False, "reason": "invalid rotation"})
    else:
        checks.append(
            {
                "name": name,
                "matched": bool(angle <= tolerance),
                "metric": "geodesic",
                "unit": "rad",
                "tolerance": tolerance,
                "angle": angle,
            }
        )


def _reconcile_table_row(
    config: GeometryInspectionConfig,
    row: Mapping[str, object],
    expected: Mapping[str, object],
    descriptor: Mapping[str, object],
    source_protos: Mapping[str, dict[str, str]],
    live_expected: Mapping[str, object] | None = None,
    live_base_reason: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Compare one raw runtime row with source-derived expectations; never backfill observations."""
    reasons: list[str] = []
    checks: list[dict[str, object]] = []
    audit_checks: list[dict[str, object]] = []
    source_name = row.get("source_name")
    if row.get("source_protos") != descriptor.get("source_protos"):
        reasons.append("source PROTO provenance differs from the pinned descriptor")
    runtime_node = row.get("runtime_node")
    unsupported_identity = False
    if not isinstance(runtime_node, Mapping):
        reasons.append("runtime node identity is missing")
        runtime_node = {}
    else:
        if not _table_identity_matches(runtime_node, config.table_targets.nested_type):
            unsupported_identity = True
            reasons.append("runtime node type is not the pinned nested type")
        if runtime_node.get("name") != source_name:
            reasons.append("runtime node name is not the pinned source name")
    hierarchy = row.get("hierarchy")
    if not isinstance(hierarchy, Mapping) or hierarchy.get("reachable") is not True:
        reasons.append("node is not reachable through the supported Supervisor interface")
        hierarchy = {}
    else:
        table_ancestor = hierarchy.get("table_ancestor")
        if not _table_identity_matches(table_ancestor, config.table_targets.parent_type):
            reasons.append("runtime Table parent ancestry is missing")
    evaluated = row.get("evaluated")
    if not isinstance(evaluated, Mapping):
        reasons.append("evaluated runtime fields are missing")
        evaluated = {}
    transforms = row.get("transforms")
    if not isinstance(transforms, Mapping):
        reasons.append("runtime transforms are missing")
        transforms = {}

    observed_translation = _number_vector(
        evaluated.get("translation_m"), 3, "evaluated.translation_m"
    )
    observed_size = _number_vector(evaluated.get("size_m"), 3, "evaluated.size_m")
    observed_rotation = _number_vector(
        evaluated.get("rotation_axis_angle"), 4, "evaluated.rotation_axis_angle"
    )
    checks.append(
        _table_check(
            "local_translation_m",
            observed_translation,
            expected.get("local_translation_m"),
            config.table_targets.translation_tolerance_m,
            metric="max_abs",
            unit="m",
        )
    )
    checks.append(
        _table_check(
            "size_m",
            observed_size,
            expected.get("size_m"),
            config.table_targets.size_tolerance_m,
            metric="max_abs",
            unit="m",
        )
    )
    expected_rotation = _number_vector(
        expected.get("local_rotation_axis_angle"), 4, "local_rotation_axis_angle"
    )
    if observed_rotation is None or expected_rotation is None:
        checks.append(
            {"name": "local_rotation_rad", "matched": False, "reason": "missing observation"}
        )
    else:
        try:
            observed_rotation_matrix = axis_angle_matrix(observed_rotation)
            expected_rotation_matrix = axis_angle_matrix(expected_rotation)
        except (TypeError, ValueError):
            checks.append(
                {"name": "local_rotation_rad", "matched": False, "reason": "invalid rotation"}
            )
        else:
            angle = _rotation_angle(observed_rotation_matrix, expected_rotation_matrix)
            if angle is None:
                checks.append(
                    {"name": "local_rotation_rad", "matched": False, "reason": "invalid rotation"}
                )
            else:
                checks.append(
                    {
                        "name": "local_rotation_rad",
                        "matched": bool(angle <= config.table_targets.rotation_tolerance_rad),
                        "metric": "geodesic",
                        "unit": "rad",
                        "tolerance": config.table_targets.rotation_tolerance_rad,
                        "angle": angle,
                    }
                )

    raw_enable = evaluated.get("enable_bounding_object")
    try:
        enable = normalize_webots_sfbool(raw_enable, field_name="enableBoundingObject")
    except ValueError:
        enable = None
        checks.append(
            {
                "name": "enable_bounding_object",
                "matched": False,
                "reason": "ambiguous Webots SFBool",
            }
        )
    else:
        checks.append({"name": "enable_bounding_object", "matched": enable is True})
    if evaluated.get("bounding_object_type") != config.table_targets.bounding_shape_type:
        checks.append(
            {
                "name": "bounding_object_type",
                "matched": False,
                "reason": "bounding object is not the pinned Box shape",
            }
        )
    checks.append(
        _table_check(
            "bounding_object_size_m",
            _number_vector(evaluated.get("bounding_object_size_m"), 3, "boundingObject.size"),
            expected.get("size_m"),
            config.table_targets.size_tolerance_m,
            metric="max_abs",
            unit="m",
        )
    )

    world_pose = _pose_matrix(transforms.get("T_world"), "T_world")
    base_pose = _pose_matrix(transforms.get("T_base"), "T_base")
    checks.append(
        _table_check(
            "T_world_translation_m",
            None if world_pose is None else world_pose[:3, 3],
            expected.get("world_translation_m"),
            config.table_targets.transform_translation_tolerance_m,
            metric="max_abs",
            unit="m",
        )
    )
    _append_rotation_check(
        checks,
        "T_world_rotation_rad",
        world_pose,
        expected.get("world_rotation_matrix"),
        config.table_targets.transform_rotation_tolerance_rad,
    )

    if live_expected is not None:
        primary_base_expected = live_expected
        base_comparison_source = "observed_live_base"
    else:
        primary_base_expected = expected
        base_comparison_source = "pinned_initial_base_fallback"
        if live_base_reason:
            reasons.append(live_base_reason)

    checks.append(
        _table_check(
            "T_base_translation_m",
            None if base_pose is None else base_pose[:3, 3],
            primary_base_expected.get("base_translation_m"),
            config.table_targets.transform_translation_tolerance_m,
            metric="max_abs",
            unit="m",
        )
    )
    _append_rotation_check(
        checks,
        "T_base_rotation_rad",
        base_pose,
        primary_base_expected.get("base_rotation_matrix"),
        config.table_targets.transform_rotation_tolerance_rad,
    )

    audit_checks.append(
        _table_check(
            "audit_pinned_T_base_translation_m",
            None if base_pose is None else base_pose[:3, 3],
            expected.get("base_translation_m"),
            config.table_targets.transform_translation_tolerance_m,
            metric="max_abs",
            unit="m",
        )
    )
    _append_rotation_check(
        audit_checks,
        "audit_pinned_T_base_rotation_rad",
        base_pose,
        expected.get("base_rotation_matrix"),
        config.table_targets.transform_rotation_tolerance_rad,
    )
    if live_expected is not None:
        audit_checks.append(
            _table_check(
                "audit_live_T_base_translation_m",
                None if base_pose is None else base_pose[:3, 3],
                live_expected.get("base_translation_m"),
                config.table_targets.transform_translation_tolerance_m,
                metric="max_abs",
                unit="m",
            )
        )
        _append_rotation_check(
            audit_checks,
            "audit_live_T_base_rotation_rad",
            base_pose,
            live_expected.get("base_rotation_matrix"),
            config.table_targets.transform_rotation_tolerance_rad,
        )

    failed = [
        f"{check['name']}: {check.get('reason', check.get('max_abs_error', check.get('angle')))}"
        for check in checks
        if not check["matched"]
    ]
    reasons.extend(failed)
    unsupported_reasons = row.get("unsupported_reasons")
    observed_unsupported: list[str] = []
    if isinstance(unsupported_reasons, Sequence) and not isinstance(
        unsupported_reasons, (str, bytes)
    ):
        observed_unsupported = [str(item) for item in unsupported_reasons]
        reasons.extend(observed_unsupported)
    if row.get("status") != "observed":
        reasons.append(f"probe status={row.get('status')!r}")
    if row.get("status") == "missing":
        verdict = "Missing"
    elif (
        observed_unsupported
        or unsupported_identity
        or row.get("status") == "unsupported"
        or not isinstance(runtime_node, Mapping)
        or not runtime_node
        or not _table_identity_matches(
            hierarchy.get("table_ancestor"), config.table_targets.parent_type
        )
    ):
        verdict = "Unsupported"
    else:
        verdict = "Matched" if not reasons else "Mismatched"
    detail = {
        "source_name": source_name,
        "verdict": verdict,
        "runtime_verification": "VERIFIED" if verdict == "Matched" else "UNVERIFIED",
        "probe_status": row.get("status"),
        "probe_reason": row.get("reason"),
        "runtime_node": dict(runtime_node),
        "hierarchy": dict(hierarchy),
        "evaluated": dict(evaluated),
        "transforms": dict(transforms),
        "expected": dict(expected),
        "expected_base_comparison": dict(primary_base_expected),
        "base_comparison": {
            "primary": base_comparison_source,
            "live_expected_available": live_expected is not None,
            "pinned_initial_base_preserved": True,
        },
        "audit_comparisons": {
            "pinned_initial_base": {
                "translation_m": expected.get("base_translation_m"),
                "rotation_matrix": expected.get("base_rotation_matrix"),
            },
            "observed_live_base": live_expected,
            "checks": audit_checks,
        },
        "source_provenance": {
            "world_sha256": config.diagnostic.world_sha256,
            "table_proto": source_protos.get("Table"),
            "solid_box_proto": source_protos.get("SolidBox"),
            "derivation": "pinned source arithmetic; not a runtime observation",
        },
        "checks": checks,
        "reasons": reasons,
    }
    return detail, [
        f"table target {source_name}: {verdict} ({'; '.join(reasons)})"
    ] if reasons else []


def reconcile_table_targets(
    config: GeometryInspectionConfig, snapshot: GeometryInspectionSnapshot
) -> dict[str, object]:
    """Reconcile raw supervisor observations with pinned source geometry, fail-closed.

    Observed world-frame geometry is compared against pinned source geometry. Observed
    base-frame geometry is compared against the expected base-relative pose derived from
    pinned world geometry and the observed live robot-base pose in the same snapshot
    (``T_BT = inverse(T_WB_live) @ T_WT``). The original pinned-initial-base comparison
    is retained for every target as an audit diagnostic and never determines verdict.
    """
    expected_geometry = _expected_table_geometry(config)
    descriptor = table_target_descriptor(config)
    source_protos = table_source_provenance(config)
    boxes_value = expected_geometry.get("boxes")
    if not isinstance(boxes_value, Sequence) or isinstance(boxes_value, (str, bytes)):
        raise ValueError("source table expansion produced no boxes")
    expected_boxes = {
        str(item["name"]): item
        for item in boxes_value
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    raw = snapshot.geometry.get("table_targets")
    blockers: list[str] = []
    rows: Sequence[object] = ()
    if not isinstance(raw, Mapping) or not set(raw).issuperset(_TABLE_OBSERVATION_KEYS):
        blockers.append("geometry export omitted a complete table_targets observation")
    else:
        if raw.get("schema_version") != TABLE_TARGET_SCHEMA_VERSION:
            blockers.append("table target observation schema_version is unsupported")
        if raw.get("interface") != descriptor:
            blockers.append("table target observation descriptor differs from the pinned contract")
        raw_rows = raw.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            blockers.append("table target observation rows are invalid")
        else:
            rows = raw_rows
        raw_errors = raw.get("errors")
        if not isinstance(raw_errors, Sequence) or isinstance(raw_errors, (str, bytes)):
            blockers.append("table target observation errors are invalid")
        else:
            blockers.extend(f"table probe error: {item}" for item in raw_errors)

    T_world_base_live, live_base_reason = _snapshot_live_base(snapshot)
    live_base_available = T_world_base_live is not None
    if not live_base_available and live_base_reason:
        blockers.append(live_base_reason)

    targets: list[dict[str, object]] = []
    for name in config.table_targets.source_names:
        candidates = [
            item for item in rows if isinstance(item, Mapping) and item.get("source_name") == name
        ]
        if not candidates:
            targets.append(
                {
                    "source_name": name,
                    "verdict": "Missing",
                    "runtime_verification": "UNVERIFIED",
                    "reasons": ["no runtime row observed; source values were not substituted"],
                }
            )
            blockers.append(f"table target {name}: Missing (no runtime row observed)")
            continue
        if len(candidates) > 1:
            targets.append(
                {
                    "source_name": name,
                    "verdict": "Unsupported",
                    "runtime_verification": "UNVERIFIED",
                    "reasons": ["duplicate runtime rows for one pinned source name"],
                }
            )
            blockers.append(f"table target {name}: Unsupported (duplicate runtime rows)")
            continue
        row = candidates[0]
        if not set(row).issuperset(_TABLE_ROW_KEYS):
            targets.append(
                {
                    "source_name": name,
                    "verdict": "Unsupported",
                    "runtime_verification": "UNVERIFIED",
                    "reasons": ["runtime row omitted required fields"],
                }
            )
            blockers.append(f"table target {name}: Unsupported (incomplete runtime row)")
            continue
        expected_box = expected_boxes[name]
        live_expected = (
            _derive_live_base_expectation(expected_box, T_world_base_live)
            if T_world_base_live is not None
            else None
        )
        detail, row_blockers = _reconcile_table_row(
            config,
            row,
            expected_box,
            descriptor,
            source_protos,
            live_expected=live_expected,
            live_base_reason=live_base_reason,
        )
        targets.append(detail)
        blockers.extend(row_blockers)
    counts = {verdict: 0 for verdict in _TABLE_VERDICTS}
    for item in targets:
        verdict = str(item.get("verdict"))
        if verdict in counts:
            counts[verdict] += 1
    grounded = counts["Matched"] == len(config.table_targets.source_names) and not blockers
    return {
        "schema_version": TABLE_TARGET_SCHEMA_VERSION,
        "status": "runtime_grounded" if grounded else "unverified",
        "runtime_grounded": grounded,
        "source_provenance": {
            "world_sha256": config.diagnostic.world_sha256,
            "table_proto": source_protos["Table"],
            "solid_box_proto": source_protos["SolidBox"],
            "scene_catalog_sha256": config.scene_catalog_sha256,
            "method": "pinned source world geometry compared with raw runtime observations; "
            "runtime base comparison uses the observed live robot-base pose from the snapshot",
        },
        "base_comparison": {
            "primary": "observed_live_base" if live_base_available else "unavailable",
            "audit_pinned_initial_base_preserved": True,
            "observed_live_base_available": live_base_available,
            "T_world_base_live_observed": (
                [row[:] for row in T_world_base_live.tolist()] if live_base_available else None
            ),
        },
        "tolerances": {
            "translation_tolerance_m": config.table_targets.translation_tolerance_m,
            "rotation_tolerance_rad": config.table_targets.rotation_tolerance_rad,
            "size_tolerance_m": config.table_targets.size_tolerance_m,
            "transform_translation_tolerance_m": (
                config.table_targets.transform_translation_tolerance_m
            ),
            "transform_rotation_tolerance_rad": (
                config.table_targets.transform_rotation_tolerance_rad
            ),
        },
        "expected_source_geometry": expected_geometry,
        "targets": targets,
        "summary": counts,
        "blockers": blockers,
    }


def _string_items(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _failed_table_report(config: GeometryInspectionConfig, reason: str) -> dict[str, object]:
    """Fail-closed table report used when pinned source reconciliation cannot run."""
    return {
        "schema_version": TABLE_TARGET_SCHEMA_VERSION,
        "status": "unverified",
        "runtime_grounded": False,
        "targets": [
            {
                "source_name": name,
                "verdict": "Missing",
                "runtime_verification": "UNVERIFIED",
                "reasons": [reason],
            }
            for name in config.table_targets.source_names
        ],
        "summary": {verdict: 0 for verdict in _TABLE_VERDICTS}
        | {"Missing": len(config.table_targets.source_names)},
        "blockers": [reason],
    }


def _geometry_blockers(
    config: GeometryInspectionConfig,
    snapshot: GeometryInspectionSnapshot,
    table_report: Mapping[str, object] | None = None,
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
    if table_report is not None:
        blockers.extend(_string_items(table_report.get("blockers", ())))
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
    try:
        table_report = reconcile_table_targets(config, snapshot)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        table_report = _failed_table_report(config, f"table target reconciliation failed: {exc}")
    geometry_blockers = _geometry_blockers(config, snapshot, table_report)
    blockers = [*preflight.blockers, *session_blockers, *geometry_blockers]
    ready = not blockers
    motion_blockers = list(preflight.motion_blockers)
    if table_report.get("runtime_grounded") is not True:
        motion_blockers.append("unverified-runtime-table-transform")
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
        "table_target_inspection": table_report,
        "table_transform_verification": table_report.get("status"),
        "table_transform_grounded": table_report.get("runtime_grounded"),
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
        "motion_blockers": motion_blockers,
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
        "table_target_reconciliation": table_report,
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
    "TableTargetConfig",
    "failed_geometry_inspection",
    "finalize_geometry_inspection",
    "load_geometry_inspection_config",
    "normalize_webots_sfbool",
    "pose3d_to_matrix",
    "preflight_geometry_inspection",
    "reconcile_table_targets",
    "table_source_provenance",
    "table_target_descriptor",
    "write_geometry_inspection_evidence",
]
