"""Pure component reconciliation of the preserved live Webots geometry evidence.

This module never launches Webots, imports the controller API, solves IK, moves the robot,
scores an episode or reads a data split. It consumes only the hash-pinned JSON artifacts,
pinned YAML catalogs, copied PROTO sources and an independent collision-geometry fixture.
Every verdict is diagnostic; no verdict in this module authorizes motion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml

from vlarobot.backends import worldmind_bindings as bindings
from vlarobot.backends.webots.proto_sources import (
    ProtoSource,
    ProtoSourceManifest,
    load_proto_source_manifest,
)
from vlarobot.eval.worldmind_webots_geometry_inspection import normalize_webots_sfbool
from vlarobot.world_model.arm_sweep import (
    _rpy_axes,
    evaluate_sweep_safety_bound,
    required_substeps,
    urdf_collision_capsules,
)
from vlarobot.world_model.carried_object import (
    assess_carried_object_state,
    carried_lifecycle_report,
)
from vlarobot.world_model.scene_geometry import (
    SceneCollisionCatalog,
    axis_angle_matrix,
    load_scene_collision_catalog,
)
from vlarobot.world_model.self_collision import (
    build_self_collision_coverage,
    validate_contract_pair_coverage,
)

LIVE_GEOMETRY_RECONCILIATION_SCHEMA_VERSION = 1
CLASSIFICATIONS = frozenset(
    {
        "matched_and_conservatively_represented",
        "observed_but_not_conservatively_represented",
        "missing_from_live_export",
        "unsupported",
        "irrelevant_with_documented_justification",
    }
)
Classification = Literal[
    "matched_and_conservatively_represented",
    "observed_but_not_conservatively_represented",
    "missing_from_live_export",
    "unsupported",
    "irrelevant_with_documented_justification",
]

_RELATIVE_PATH_KEYS = frozenset({"path"})
_PROTO_BLOCK_RE_TEMPLATE = r"\b{name}\s*\{{(.*?)\n\}}"
_NUMBER_TOKEN = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_FIELD_VECTOR_RE_TEMPLATE = rf"^\s*{{field}}\s+((?:{_NUMBER_TOKEN}\s*)+)(?:#.*)?$"


@dataclass(frozen=True)
class EvidenceFile:
    """A pinned diagnostic artifact whose hash is checked before the file is read."""

    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class LiveReconciliationConfig:
    """Strictly loaded diagnostic reconciliation contract."""

    source_path: Path
    evidence: Mapping[str, EvidenceFile]
    pinned: Mapping[str, Path]
    expected: Mapping[str, object]
    tolerances: Mapping[str, float]
    sweep: Mapping[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def _resolve(root: Path, value: object, name: str) -> Path:
    path = Path(_text(value, name)).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has unexpected keys: {sorted(set(value) ^ expected)}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _nonnegative_optional(value: object, name: str) -> float | None:
    if value is None:
        return None
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _evidence_file(root: Path, value: object, name: str) -> EvidenceFile:
    row = _mapping(value, f"evidence.{name}")
    _keys(row, frozenset({"path", "sha256"}), f"evidence.{name}")
    path = _resolve(root, row["path"], f"evidence.{name}.path")
    digest = _digest(row["sha256"], f"evidence.{name}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"pinned evidence file is missing: {name}={path}")
    observed = _sha256(path)
    if observed != digest:
        raise ValueError(f"pinned evidence SHA-256 mismatch for {name}: {observed}")
    return EvidenceFile(name, path, digest)


def load_live_reconciliation_config(
    path: Path | None = None,
) -> LiveReconciliationConfig:
    """Load the contract and verify every pinned diagnostic hash before any JSON is consumed."""
    source = (path or DEFAULT_LIVE_RECONCILIATION_CONFIG).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"live geometry reconciliation config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "live geometry reconciliation config")
    _keys(
        root,
        frozenset(
            {"schema_version", "status", "evidence", "pinned", "expected", "tolerances", "sweep"}
        ),
        "live geometry reconciliation config",
    )
    if root["schema_version"] != LIVE_GEOMETRY_RECONCILIATION_SCHEMA_VERSION:
        raise ValueError("unsupported live geometry reconciliation schema_version")
    if _text(root["status"], "status") != "diagnostic_not_certified":
        raise ValueError("live reconciliation status must be diagnostic_not_certified")

    evidence_raw = _mapping(root["evidence"], "evidence")
    evidence_names = frozenset(
        {
            "geometry",
            "diagnostic",
            "state",
            "provenance",
            "log",
            "reference_collision_geometry",
            "reference_safety_cases",
        }
    )
    _keys(evidence_raw, evidence_names, "evidence")
    evidence = {
        name: _evidence_file(source.parent, evidence_raw[name], name)
        for name in sorted(evidence_names)
    }

    pinned_raw = _mapping(root["pinned"], "pinned")
    pinned_names = frozenset(
        {"world", "urdf", "scene_catalog", "proto_manifest", "arm_sweep_contract"}
    )
    _keys(pinned_raw, pinned_names, "pinned")
    pinned = {
        name: _resolve(source.parent, pinned_raw[name], f"pinned.{name}")
        for name in sorted(pinned_names)
    }
    for name, pinned_path in pinned.items():
        if not pinned_path.is_file():
            raise FileNotFoundError(f"pinned source is missing: {name}={pinned_path}")

    expected = _mapping(root["expected"], "expected")
    _keys(
        expected,
        frozenset(
            {
                "world_sha256",
                "diagnostic_outcome",
                "robot_def",
                "self_collision_enabled",
                "object_count",
                "supported_shape_count",
                "unsupported_shape_count",
                "floor_owners",
                "pipe_section_count",
                "straight_stairs_rail_count",
                "table_export_count",
                "can_def",
                "gripper_defs",
                "arm_torso_defs",
            }
        ),
        "expected",
    )
    _digest(expected["world_sha256"], "expected.world_sha256")
    if type(expected["self_collision_enabled"]) is not bool:
        raise ValueError("expected.self_collision_enabled must be a strict bool")
    for name in ("floor_owners", "gripper_defs", "arm_torso_defs"):
        if not isinstance(expected[name], Sequence) or isinstance(expected[name], (str, bytes)):
            raise ValueError(f"expected.{name} must be a list")
    for name in (
        "object_count",
        "supported_shape_count",
        "unsupported_shape_count",
        "pipe_section_count",
        "straight_stairs_rail_count",
        "table_export_count",
    ):
        if isinstance(expected[name], bool) or not isinstance(expected[name], int):
            raise ValueError(f"expected.{name} must be an integer")
    if _sha256(pinned["world"]) != expected["world_sha256"]:
        raise ValueError("pinned world SHA-256 does not match the reconciliation contract")

    tolerances_raw = _mapping(root["tolerances"], "tolerances")
    tolerance_names = frozenset(
        {
            "rigid_matrix_atol",
            "normal_dot_min",
            "plane_offset_m",
            "primitive_rtol",
            "primitive_atol",
        }
    )
    _keys(tolerances_raw, tolerance_names, "tolerances")
    tolerances = {
        name: _finite(tolerances_raw[name], f"tolerances.{name}") for name in tolerance_names
    }
    for name in ("rigid_matrix_atol", "plane_offset_m", "primitive_rtol", "primitive_atol"):
        if tolerances[name] < 0.0:
            raise ValueError(f"tolerances.{name} must be nonnegative")

    sweep_raw = _mapping(root["sweep"], "sweep")
    sweep_names = frozenset(
        {
            "margin_m",
            "point_speed_bound_m_s",
            "dt_s",
            "configured_substeps",
            "adaptive_substeps",
            "max_joint_displacement_bound_rad",
            "geometry_approximation_error_bound_m",
            "numerical_error_bound_m",
            "transform_error_bound_m",
            "observed_one_pose_aabb_undercoverage_m",
            "observed_one_pose_aabb_undercoverage_is_global_bound",
        }
    )
    _keys(sweep_raw, sweep_names, "sweep")
    sweep: dict[str, Any] = {
        "margin_m": _finite(sweep_raw["margin_m"], "sweep.margin_m"),
        "point_speed_bound_m_s": _finite(
            sweep_raw["point_speed_bound_m_s"], "sweep.point_speed_bound_m_s"
        ),
        "dt_s": _finite(sweep_raw["dt_s"], "sweep.dt_s"),
        "max_joint_displacement_bound_rad": _nonnegative_optional(
            sweep_raw["max_joint_displacement_bound_rad"],
            "sweep.max_joint_displacement_bound_rad",
        ),
        "geometry_approximation_error_bound_m": _nonnegative_optional(
            sweep_raw["geometry_approximation_error_bound_m"],
            "sweep.geometry_approximation_error_bound_m",
        ),
        "numerical_error_bound_m": _nonnegative_optional(
            sweep_raw["numerical_error_bound_m"], "sweep.numerical_error_bound_m"
        ),
        "transform_error_bound_m": _nonnegative_optional(
            sweep_raw["transform_error_bound_m"], "sweep.transform_error_bound_m"
        ),
        "observed_one_pose_aabb_undercoverage_m": _finite(
            sweep_raw["observed_one_pose_aabb_undercoverage_m"],
            "sweep.observed_one_pose_aabb_undercoverage_m",
        ),
    }
    for name in ("configured_substeps", "adaptive_substeps"):
        value = sweep_raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"sweep.{name} must be a positive integer")
        sweep[name] = value
    if type(sweep_raw["observed_one_pose_aabb_undercoverage_is_global_bound"]) is not bool:
        raise ValueError(
            "sweep.observed_one_pose_aabb_undercoverage_is_global_bound must be a strict bool"
        )
    sweep["observed_one_pose_aabb_undercoverage_is_global_bound"] = sweep_raw[
        "observed_one_pose_aabb_undercoverage_is_global_bound"
    ]

    return LiveReconciliationConfig(
        source_path=source,
        evidence=evidence,
        pinned=pinned,
        expected=expected,
        tolerances=tolerances,
        sweep=sweep,
    )


DEFAULT_LIVE_RECONCILIATION_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "geometry" / "live-geometry-evidence-v1.yaml"
)


def _load_verified_json(item: EvidenceFile) -> Mapping[str, object]:
    """Re-verify the pinned hash immediately before parsing the artifact."""
    if _sha256(item.path) != item.sha256:
        raise ValueError(f"evidence changed after config load: {item.name}")
    payload = json.loads(item.path.read_text(encoding="utf-8"))
    return _mapping(payload, item.name)


def _matrix(value: object, name: str, atol: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (16,):
        array = array.reshape(4, 4)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = array[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=max(atol, 1e-8))
        or not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=atol, rtol=0.0)
    ):
        raise ValueError(f"{name} is not a proper rigid transform")
    return array


def _vector3(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def _near(a: float, b: float, cfg: LiveReconciliationConfig) -> bool:
    return math.isclose(
        a,
        b,
        rel_tol=cfg.tolerances["primitive_rtol"],
        abs_tol=cfg.tolerances["primitive_atol"],
    )


def _component(
    *,
    name: str,
    classification: Classification,
    live: Mapping[str, object],
    pinned: Mapping[str, object],
    reason: str,
    blockers: Sequence[str] = (),
    conservative_representation: bool = False,
) -> dict[str, object]:
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"unsupported component classification: {classification}")
    return {
        "name": name,
        "classification": classification,
        "conservative_representation": conservative_representation,
        "live": dict(live),
        "pinned": dict(pinned),
        "reason": reason,
        "blockers": list(blockers),
    }


def _object_index(
    objects: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    by_def: dict[str, Mapping[str, object]] = {}
    by_name: dict[str, Mapping[str, object]] = {}
    for item in objects:
        marker = item.get("def")
        if isinstance(marker, str) and marker:
            if marker in by_def:
                raise ValueError(f"duplicate live object DEF: {marker}")
            by_def[marker] = item
        label = item.get("name")
        if isinstance(label, str) and label:
            by_name[label] = item
    return by_def, by_name


def _shape_parameters(shape: Mapping[str, object]) -> Mapping[str, object]:
    parameters = shape.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("live shape has no parameters mapping")
    return cast(Mapping[str, object], parameters)


def _verify_export_contract(
    cfg: LiveReconciliationConfig,
    geometry: Mapping[str, object],
    diagnostic: Mapping[str, object],
) -> tuple[list[str], Sequence[Mapping[str, object]]]:
    blockers: list[str] = []
    expected = cfg.expected
    if diagnostic.get("outcome") != expected["diagnostic_outcome"]:
        blockers.append("preserved diagnostic outcome is not the accepted FAILED inspection")
    diagnostic_blockers = diagnostic.get("blockers")
    if not isinstance(diagnostic_blockers, Sequence) or not any(
        isinstance(item, str) and "selfCollision" in item for item in diagnostic_blockers
    ):
        blockers.append("preserved FAILED diagnostic does not name the selfCollision blocker")
    export = geometry.get("export")
    if not isinstance(export, Mapping):
        raise ValueError("geometry.json has no export object")
    export = cast(Mapping[str, object], export)
    if export.get("robot_id") != 794:
        blockers.append("live robot identity differs from the accepted inspection")
    if export.get("scene") is None or "no TRAIN/VALIDATION/TEST" not in str(export["scene"]):
        blockers.append("live export scene declaration is missing or unsafe")
    if export.get("motor_commands_sent") != 0:
        blockers.append("live export reports motor commands")
    raw_self_collision = export.get("self_collision_enabled")
    try:
        normalized = normalize_webots_sfbool(raw_self_collision, field_name="selfCollision")
    except ValueError:
        blockers.append("live selfCollision value is not a documented Webots SFBool")
    else:
        if normalized is not expected["self_collision_enabled"]:
            blockers.append("live selfCollision normalization differs from the pinned PROTO")
    objects = export.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        raise ValueError("geometry.json export has no objects list")
    typed_objects = [
        cast(Mapping[str, object], item) for item in objects if isinstance(item, Mapping)
    ]
    if len(typed_objects) != len(objects) or len(typed_objects) != expected["object_count"]:
        blockers.append("live object count differs from the accepted inspection")
    shapes = [
        shape for item in typed_objects for shape in cast(Sequence[object], item.get("shapes", ()))
    ]
    supported = sum(
        1 for shape in shapes if isinstance(shape, Mapping) and shape.get("supported") is True
    )
    unsupported = [
        shape
        for shape in shapes
        if isinstance(shape, Mapping) and shape.get("supported") is not True
    ]
    if supported != expected["supported_shape_count"]:
        blockers.append("live supported-shape count differs from the accepted inspection")
    if len(unsupported) != expected["unsupported_shape_count"]:
        blockers.append("live unsupported-shape count differs from the accepted inspection")
    if {shape.get("type") for shape in unsupported} != {"Plane"}:
        blockers.append("unexpected unsupported shape type in the preserved export")
    return blockers, typed_objects


def _validate_floors(
    cfg: LiveReconciliationConfig,
    catalog: SceneCollisionCatalog,
    objects: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    expected_owners = list(cast(Sequence[str], cfg.expected["floor_owners"]))
    candidates = [item for item in objects if item.get("type") == "Floor"]
    matched: list[dict[str, object]] = []
    atol = cfg.tolerances["rigid_matrix_atol"]
    for item in candidates:
        name = item.get("name")
        record: dict[str, object] = {"owner": name, "matched_pinned_plane": None}
        if name not in expected_owners:
            blockers.append(f"unpinned Floor object in live export: {name}")
        shapes = item.get("shapes")
        if not isinstance(shapes, Sequence) or len(shapes) != 1:
            blockers.append(f"Floor {name} does not have exactly one bounding shape")
            matched.append(record)
            continue
        shape = cast(Mapping[str, object], shapes[0])
        if shape.get("type") != "Plane":
            blockers.append(f"Floor {name} bounding shape is not a Plane")
            matched.append(record)
            continue
        size = _shape_parameters(shape).get("size")
        size_array = np.asarray(size, dtype=np.float64) if size is not None else np.zeros(0)
        if size_array.shape != (2,) or not np.isfinite(size_array).all() or np.any(size_array <= 0):
            blockers.append(f"Floor {name} has an invalid Plane size")
            matched.append(record)
            continue
        T_world_owner = _matrix(item.get("T_world_owner"), f"Floor {name} world transform", atol)
        T_owner_shape = _matrix(shape.get("T_owner_shape"), f"Floor {name} shape transform", atol)
        T_world_shape = T_world_owner @ T_owner_shape
        world_point = T_world_shape[:3, 3]
        world_normal = T_world_shape[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
        T_base_world = catalog.T_base_world
        base_point = T_base_world[:3, :3] @ world_point + T_base_world[:3, 3]
        base_normal = T_base_world[:3, :3] @ world_normal
        base_normal /= np.linalg.norm(base_normal)
        best_name: str | None = None
        best_dot = -2.0
        for pinned in catalog.geometry.floors:
            pinned_normal = np.asarray(pinned.normal_base, dtype=np.float64)
            dot = float(base_normal @ pinned_normal)
            if dot > best_dot:
                best_dot, best_name = dot, pinned.name
        if best_name is None or best_dot < cfg.tolerances["normal_dot_min"]:
            blockers.append(f"Floor {name} normal is not one of the two pinned horizontal planes")
            matched.append(record)
            continue
        pinned = next(item for item in catalog.geometry.floors if item.name == best_name)
        pinned_normal = np.asarray(pinned.normal_base, dtype=np.float64)
        pinned_point = np.asarray(pinned.point_base, dtype=np.float64)
        offset_error = abs(float(base_normal @ base_point - pinned_normal @ pinned_point))
        if offset_error > cfg.tolerances["plane_offset_m"]:
            blockers.append(f"Floor {name} plane offset differs from pinned {best_name}")
            matched.append(record)
            continue
        record.update(
            {
                "matched_pinned_plane": best_name,
                "normal_alignment": best_dot,
                "plane_offset_error_m": offset_error,
                "world_point": world_point.tolist(),
                "world_normal": world_normal.tolist(),
                "size": size_array.tolist(),
            }
        )
        matched.append(record)

    owners = [item.get("owner") for item in matched if item.get("matched_pinned_plane") is not None]
    if set(owners) != set(expected_owners) or len(owners) != len(expected_owners):
        blockers.append("both pinned horizontal floor planes are not matched exactly once")
    if len(candidates) != len(expected_owners):
        blockers.append("live export does not contain exactly the two pinned Floor objects")
    detail = {
        "supported_horizontal_plane_count": len(expected_owners),
        "observed_floor_objects": len(candidates),
        "matched": matched,
        "arbitrary_orientation_policy": "reject",
        "unsupported_plane_policy": "fail_closed_except_the_two_pinned_horizontal_cases",
    }
    return detail, blockers


def _world_block(text: str, name: str) -> str:
    match = re.search(_PROTO_BLOCK_RE_TEMPLATE.format(name=re.escape(name)), text, re.DOTALL)
    if match is None:
        raise ValueError(f"world has no {name} instance block")
    return match.group(1)


def _proto_interface_block(text: str, name: str) -> str:
    match = re.search(rf"\bPROTO\s+{re.escape(name)}\s*\[(.*?)\n\]", text, re.DOTALL)
    if match is None:
        raise ValueError(f"PROTO {name} interface block is missing")
    return match.group(1)


def _field_values(block: str, field: str) -> tuple[str, ...] | None:
    match = re.search(_FIELD_VECTOR_RE_TEMPLATE.format(field=re.escape(field)), block, re.MULTILINE)
    if match is None:
        return None
    return tuple(match.group(1).split())


def _field_string(block: str, field: str) -> str | None:
    match = re.search(rf'^\s*{re.escape(field)}\s+"([^"]*)"', block, re.MULTILINE)
    return match.group(1) if match else None


def _matrix_axis_angle(rotation: np.ndarray) -> list[float]:
    """Return an xyzw axis-angle for a proper rotation matrix (pure source arithmetic)."""
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    if angle <= 1e-12:
        return [0.0, 0.0, 1.0, 0.0]
    sine = math.sin(angle)
    axis = np.asarray(
        [
            (rotation[2, 1] - rotation[1, 2]) / (2.0 * sine),
            (rotation[0, 2] - rotation[2, 0]) / (2.0 * sine),
            (rotation[1, 0] - rotation[0, 1]) / (2.0 * sine),
        ]
    )
    return [*axis.tolist(), angle]


def expand_table_source_geometry(
    world_path: Path,
    table_source: ProtoSource,
    solid_source: ProtoSource,
    catalog: Any,
    reach_bound_m: float,
    geometry_error_available: bool,
    world_sha256: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    """Expand pinned Table/SolidBox sources without claiming runtime grounding.

    The returned translation, rotation and size values are source-derived expectations only. They
    are never copied into a runtime observation when a live node is missing or uninspectable.
    """
    blockers: list[str] = []
    table_text = table_source.path.read_text(encoding="utf-8")
    solid_text = solid_source.path.read_text(encoding="utf-8")
    root_bounding_objects = len(re.findall(r"\bboundingObject\b", table_text))
    nested_solid_boxes = len(re.findall(r"\bSolidBox\s*\{", table_text))
    solidbox_bounding_box = bool(re.search(r"boundingObject\s+Box\s*\{", solid_text))
    if root_bounding_objects != 0 or nested_solid_boxes < 3 or not solidbox_bounding_box:
        blockers.append("pinned Table/SolidBox nested collision structure is not as expected")

    table_block = _world_block(world_path.read_text(encoding="utf-8"), "Table")
    translation_values = _field_values(table_block, "translation")
    size_values = _field_values(table_block, "size")
    if translation_values is None or len(translation_values) != 3:
        blockers.append("pinned world Table translation is incomplete")
        translation_values = ("0", "0", "0")
    if size_values is None or len(size_values) != 3:
        blockers.append("pinned world Table size is incomplete")
        size_values = ("1", "1.8", "0.74")
    translation = np.asarray([float(value) for value in translation_values])
    size = np.asarray([float(value) for value in size_values])

    interface = _proto_interface_block(table_text, "Table")
    solid_interface = _proto_interface_block(solid_text, "SolidBox")
    rotation_default = tuple(
        float(value) for value in _proto_default(interface, "rotation", "SFRotation")
    )
    local_rotation = tuple(
        float(value) for value in _proto_default(solid_interface, "rotation", "SFRotation")
    )
    feet_size = np.asarray(
        [float(value) for value in _proto_default(interface, "feetSize", "SFVec2f")]
    )
    feet_padding = float(_proto_default(interface, "feetPadding", "SFFloat")[0])
    frame_thickness = float(_proto_default(interface, "frameThickness", "SFFloat")[0])
    solidbox_blocks = re.findall(r"\bSolidBox\s*\{([^{}]*)\}", table_text, re.DOTALL)
    if any(re.search(r"^\s*rotation\s+", block, re.MULTILINE) for block in solidbox_blocks):
        blockers.append("pinned Table overrides a nested SolidBox rotation, which is unsupported")
    rotation_values = _field_values(table_block, "rotation")
    root_rotation_axis_angle = (
        tuple(float(value) for value in rotation_values[:4])
        if rotation_values is not None and len(rotation_values) >= 4
        else rotation_default
    )
    root_rotation = axis_angle_matrix(root_rotation_axis_angle)
    R_base_world = catalog.T_base_world[:3, :3]

    main_size = np.asarray((size[0], size[1], frame_thickness))
    main_local = np.asarray((0.0, 0.0, size[2] - 0.5 * frame_thickness))
    foot_local_x = 0.5 * size[0] - feet_padding - 0.5 * feet_size[0]
    foot_local_y = 0.5 * size[1] - feet_padding - 0.5 * feet_size[1]
    foot_local_z = 0.5 * (size[2] - frame_thickness)
    foot_size = np.asarray((feet_size[0], feet_size[1], size[2] - frame_thickness))
    definitions = (
        ("main frame", main_local, main_size),
        ("foot 1", (foot_local_x, foot_local_y, foot_local_z), foot_size),
        ("foot 2", (-foot_local_x, foot_local_y, foot_local_z), foot_size),
        ("foot 3", (foot_local_x, -foot_local_y, foot_local_z), foot_size),
        ("foot 4", (-foot_local_x, -foot_local_y, foot_local_z), foot_size),
    )

    boxes: list[dict[str, object]] = []
    for name, local, box_size in definitions:
        local_array = np.asarray(local)
        world_center = translation + root_rotation @ local_array
        base_center = R_base_world @ world_center + catalog.T_base_world[:3, 3]
        base_rotation = R_base_world @ root_rotation
        base_half = np.abs(base_rotation) @ (box_size / 2.0)
        closest_point = np.maximum(np.abs(base_center) - base_half, 0.0)
        distance = float(np.linalg.norm(closest_point))
        if distance > reach_bound_m and geometry_error_available:
            reachability = "outside_conservative_workspace"
        elif distance > reach_bound_m:
            reachability = "outside_nominal_workspace_unproven"
        else:
            reachability = "within_conservative_workspace"
        boxes.append(
            {
                "name": name,
                "root_translation_m": translation.tolist(),
                "local_translation_m": local_array.tolist(),
                "local_rotation_axis_angle": list(local_rotation),
                "world_translation_m": world_center.tolist(),
                "world_rotation_axis_angle": list(root_rotation_axis_angle),
                "world_rotation_matrix": root_rotation.tolist(),
                "size_m": box_size.tolist(),
                "base_translation_m": base_center.tolist(),
                "base_rotation_axis_angle": _matrix_axis_angle(base_rotation),
                "base_rotation_matrix": base_rotation.tolist(),
                "base_half_extents_m": base_half.tolist(),
                "distance_to_base_origin_m": distance,
                "reachability": reachability,
            }
        )

    coarse = catalog.geometry.boxes[0]
    coarse_center = np.asarray(coarse.center_base)
    coarse_half = np.asarray(coarse.half_base)
    contained = all(
        bool(
            np.all(
                np.abs(np.asarray(item["base_translation_m"]) - coarse_center)
                <= coarse_half - np.asarray(item["base_half_extents_m"]) + 1e-8
            )
        )
        for item in boxes
    )
    if not contained:
        blockers.append("source-derived table boxes are not contained by the pinned coarse proxy")
    return (
        {
            "boxes": boxes,
            "source_hashes": {
                "world_sha256": world_sha256 or _sha256(world_path),
                "table_proto_sha256": table_source.sha256,
                "solid_box_proto_sha256": solid_source.sha256,
                "scene_catalog_sha256": catalog.geometry.catalog_sha256,
            },
            "root_translation_m": translation.tolist(),
            "root_rotation_axis_angle": list(root_rotation_axis_angle),
            "reach_bound_m": reach_bound_m,
            "coarse_catalog_proxy_contains_all_boxes": contained,
            "cross_check": (
                "source arithmetic and coarse-proxy containment are independent of the live "
                "export, but neither grounds a runtime transform"
            ),
        },
        blockers,
    )


def _table_source_expansion(
    cfg: LiveReconciliationConfig,
    manifest: ProtoSourceManifest,
    catalog: Any,
    reach_bound_m: float,
    geometry_error_available: bool,
) -> tuple[dict[str, object], list[str]]:
    """Expand pinned Table/SolidBox sources without claiming runtime grounding."""
    sources = {item.name: item for item in (*manifest.external, *manifest.nested)}
    return expand_table_source_geometry(
        cfg.pinned["world"],
        sources["Table"],
        sources["SolidBox"],
        catalog,
        reach_bound_m,
        geometry_error_available,
        str(cfg.expected["world_sha256"]),
    )


def _trace_table(
    cfg: LiveReconciliationConfig,
    manifest: ProtoSourceManifest,
    objects: Sequence[Mapping[str, object]],
    catalog: Any,
    sweep_detail: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    by_def, by_name = _object_index(objects)
    table_hits = [
        item
        for item in objects
        if item.get("type") == "Table" or item.get("name") == "Table" or item.get("def") == "Table"
    ]
    if len(table_hits) != cfg.expected["table_export_count"]:
        blockers.append("table presence in live export differs from the grounded absence finding")
    sources = {item.name: item for item in (*manifest.external, *manifest.nested)}
    table_source = sources["Table"]
    solid_source = sources["SolidBox"]
    table_text = table_source.path.read_text(encoding="utf-8")
    solid_text = solid_source.path.read_text(encoding="utf-8")
    root_bounding_objects = len(re.findall(r"\bboundingObject\b", table_text))
    nested_solid_boxes = len(re.findall(r"\bSolidBox\s*\{", table_text))
    solidbox_bounding_box = bool(re.search(r"boundingObject\s+Box\s*\{", solid_text))
    if root_bounding_objects != 0 or nested_solid_boxes < 3 or not solidbox_bounding_box:
        blockers.append("pinned Table/SolidBox nested collision structure is not as expected")
    world_text = cfg.pinned["world"].read_text(encoding="utf-8")
    table_block = _world_block(world_text, "Table")
    translation = _field_values(table_block, "translation")
    size = _field_values(table_block, "size")
    if translation is None or len(translation) != 3 or size is None or len(size) != 3:
        blockers.append("pinned world Table transform or size is incomplete")
        transform = None
    else:
        transform = {
            "translation": [float(value) for value in translation],
            "size": [float(value) for value in size],
        }
    safety = _mapping(sweep_detail["safety_bound"], "swept_collision_validation.safety_bound")
    reach_bound = float(cast(float, sweep_detail["max_urdf_reach_m"])) + float(
        cast(float, safety["margin_m"])
    )
    registration = safety.get("registration_uncertainty_bound_m")
    if registration is not None:
        reach_bound += float(cast(float, registration))
    geometry_available = safety.get("geometry_approximation_error_bound_m") is not None
    source_expansion, source_blockers = _table_source_expansion(
        cfg, manifest, catalog, reach_bound, geometry_available
    )
    blockers.extend(source_blockers)
    detail = {
        "live_export_count": len(table_hits),
        "root_proto_bounding_object_count": root_bounding_objects,
        "nested_solidbox_count": nested_solid_boxes,
        "solidbox_bounding_object_is_box": solidbox_bounding_box,
        "pinned_world_transform_and_size": transform,
        "probe_traversal_finding": (
            "geometry_probe records only nodes with a direct boundingObject field; Table's root "
            "Solid has none, and its nested SolidBox internals are not exposed through children "
            "without device ancestry. The table has no device ancestry, so it is absent."
        ),
        "status": "source_derived_unverified",
        "runtime_collision_tree_grounded": False,
        "source_expansion": source_expansion,
        "missing_runtime_observation": (
            "A read-only Webots expansion of the five nested Table SolidBox nodes is absent: "
            "resolved PROTO hashes, evaluated translation/rotation/size/enableBoundingObject, "
            "resulting boundingObject Box.size, and each node's world/base pose."
        ),
    }
    return detail, blockers


def _proto_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def _proto_default(block: str, field: str, kind: str) -> tuple[str, ...]:
    match = re.search(
        rf"field\s+{re.escape(kind)}\s+{re.escape(field)}\s+([^\s#]+(?:[ \t]+[^\s#]+)*)",
        block,
    )
    if match is None:
        raise ValueError(f"PROTO is missing field {field}")
    return tuple(match.group(1).split())


def _trace_pipe_sections(
    cfg: LiveReconciliationConfig,
    manifest: ProtoSourceManifest,
    objects: Sequence[Mapping[str, object]],
    sweep_speed_m_s: float,
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    source = next(item for item in manifest.external if item.name == "PipeSection")
    proto_text = source.path.read_text(encoding="utf-8")
    proto_block = _proto_interface_block(proto_text, "PipeSection")
    chain = {
        "boundingObject": "boundingObject USE PIPE",
        "boundingObject_line": _proto_line(proto_text, "boundingObject USE PIPE"),
        "definition": "DEF PIPE Cylinder",
        "definition_line": _proto_line(proto_text, "DEF PIPE Cylinder"),
        "height_forwarding": "height IS height",
        "radius_forwarding": "radius IS radius",
        "translation_forwarding": "translation IS translation",
        "rotation_forwarding": "rotation IS rotation",
    }
    if not all(
        token in proto_text
        for token in (
            "boundingObject USE PIPE",
            "DEF PIPE Cylinder",
            "height IS height",
            "radius IS radius",
        )
    ):
        blockers.append("pinned PipeSection IS/USE forwarding chain is incomplete")
    defaults = {
        "height": float(_proto_default(proto_block, "height", "SFFloat")[0]),
        "radius": float(_proto_default(proto_block, "radius", "SFFloat")[0]),
        "translation": tuple(
            float(value) for value in _proto_default(proto_block, "translation", "SFVec3f")
        ),
        "rotation": tuple(
            float(value) for value in _proto_default(proto_block, "rotation", "SFRotation")
        ),
    }
    world_text = cfg.pinned["world"].read_text(encoding="utf-8")
    world_blocks = re.findall(
        _PROTO_BLOCK_RE_TEMPLATE.format(name="PipeSection"), world_text, re.DOTALL
    )
    live_objects = [item for item in objects if item.get("type") == "PipeSection"]
    if len(live_objects) != cfg.expected["pipe_section_count"]:
        blockers.append("PipeSection live instance count differs from the preserved export")
    if len(world_blocks) != len(live_objects):
        blockers.append("PipeSection world and live instance counts differ")
    by_name = {str(item.get("name")): item for item in live_objects}
    instances: list[dict[str, object]] = []
    envelopes: list[dict[str, object]] = []
    for block in world_blocks:
        name = _field_string(block, "name") or "pipe section"
        live = by_name.get(name)
        if live is None:
            blockers.append(f"world PipeSection {name} is missing from the live export")
            continue
        translation_values = _field_values(block, "translation") or defaults["translation"]
        rotation_values = _field_values(block, "rotation") or defaults["rotation"]
        height = float((_field_values(block, "height") or (str(defaults["height"]),))[0])
        radius = float((_field_values(block, "radius") or (str(defaults["radius"]),))[0])
        expected_position = np.asarray(translation_values[:3], dtype=np.float64)
        expected_rotation = axis_angle_matrix([float(value) for value in rotation_values[:4]])
        live_T_world = _matrix(
            live.get("T_world_owner"),
            f"PipeSection {name} world transform",
            cfg.tolerances["rigid_matrix_atol"],
        )
        position_error = float(np.linalg.norm(live_T_world[:3, 3] - expected_position))
        rotation_error = float(np.max(np.abs(live_T_world[:3, :3] - expected_rotation[:3, :3])))
        shapes = cast(Sequence[Mapping[str, object]], live.get("shapes", ()))
        if len(shapes) != 1 or shapes[0].get("type") != "Cylinder":
            blockers.append(f"PipeSection {name} live shape is not a single Cylinder")
            continue
        parameters = _shape_parameters(shapes[0])
        measured_height = _finite(parameters.get("height"), f"PipeSection {name} height")
        measured_radius = _finite(parameters.get("radius"), f"PipeSection {name} radius")
        parameter_match = _near(measured_height, height, cfg) and _near(
            measured_radius, radius, cfg
        )
        transform_match = (
            position_error <= cfg.tolerances["plane_offset_m"]
            and rotation_error <= cfg.tolerances["rigid_matrix_atol"]
        )
        if not (parameter_match and transform_match):
            blockers.append(f"PipeSection {name} live cylinder does not match the pinned instance")
        T_base_owner = _matrix(
            live.get("T_base_owner"),
            f"PipeSection {name} base transform",
            cfg.tolerances["rigid_matrix_atol"],
        )
        T_owner_shape = _matrix(
            shapes[0].get("T_owner_shape"),
            f"PipeSection {name} shape transform",
            cfg.tolerances["rigid_matrix_atol"],
        )
        T_base_shape = T_base_owner @ T_owner_shape
        axis_z = np.asarray((0.0, 0.0, 1.0))
        start = T_base_shape[:3, :3] @ (-0.5 * measured_height * axis_z) + T_base_shape[:3, 3]
        end = T_base_shape[:3, :3] @ (0.5 * measured_height * axis_z) + T_base_shape[:3, 3]
        envelopes.append(
            {
                "name": name,
                "start_base": start.tolist(),
                "end_base": end.tolist(),
                "radius_m": measured_radius,
                "max_point_speed_m_s": sweep_speed_m_s,
            }
        )
        instances.append(
            {
                "name": name,
                "pinned_height_m": height,
                "live_height_m": measured_height,
                "pinned_radius_m": radius,
                "live_radius_m": measured_radius,
                "position_error_m": position_error,
                "rotation_matrix_error": rotation_error,
                "matches": parameter_match and transform_match,
            }
        )
    detail = {
        "forwarding_chain": chain,
        "instance_count": len(instances),
        "instances": instances,
        "conservative_capsule_envelopes": envelopes,
        "measured_live_geometry": True,
        "runtime_certified": False,
    }
    return detail, blockers


def _trace_straight_stairs_rail(
    cfg: LiveReconciliationConfig,
    manifest: ProtoSourceManifest,
    objects: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    source = next(item for item in manifest.external if item.name == "StraightStairsRail")
    proto_text = source.path.read_text(encoding="utf-8")
    chain = [
        {
            "field": "boundingObject USE BALUSTER_GEOM",
            "line": _proto_line(proto_text, "boundingObject USE BALUSTER_GEOM"),
            "target": "DEF BALUSTER_GEOM Cylinder",
        },
        {
            "field": "boundingObject USE NEWEL",
            "line": _proto_line(proto_text, "boundingObject USE NEWEL"),
            "target": "DEF NEWEL Shape",
        },
        {
            "field": "boundingObject USE GEOM",
            "line": _proto_line(proto_text, "boundingObject USE GEOM"),
            "target": "DEF GEOM TexturedParallelepiped",
        },
    ]
    unresolved_target = "TexturedParallelepiped"
    pinned_names = {item.name for item in (*manifest.external, *manifest.nested)}
    if unresolved_target in pinned_names:
        blockers.append(
            "unexpected pinned TexturedParallelepiped source invalidates the gap finding"
        )
    if any(item["line"] is None for item in chain):
        blockers.append("pinned StraightStairsRail USE forwarding chain is incomplete")
    if "StraightStairsRail" not in manifest.unresolved:
        blockers.append("StraightStairsRail is no longer explicitly blocked by the PROTO manifest")
    world_text = cfg.pinned["world"].read_text(encoding="utf-8")
    world_count = len(
        re.findall(
            _PROTO_BLOCK_RE_TEMPLATE.format(name="StraightStairsRail"), world_text, re.DOTALL
        )
    )
    live_count = sum(1 for item in objects if item.get("type") == "StraightStairsRail")
    if live_count != cfg.expected["straight_stairs_rail_count"]:
        blockers.append("StraightStairsRail live instance count differs from the preserved export")
    detail = {
        "forwarding_chain": chain,
        "unresolved_leaf": unresolved_target,
        "unresolved_leaf_in_pinned_manifest": unresolved_target in pinned_names,
        "world_instance_count": world_count,
        "live_export_count": live_count,
        "conservative_enclosure_available": False,
        "reason_not_approved": (
            "forwarded TexturedParallelepiped geometry and JavaScript-computed transforms are "
            "not pinned, and no live object grounds a runtime transform"
        ),
    }
    return detail, blockers


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    axis = end - start
    length_squared = float(axis @ axis)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip((point - start) @ axis / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + alpha * axis)))


_BOX_CORNERS = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def _urdf_origin_transform(node: ET.Element, name: str) -> np.ndarray:
    origin = node.find("origin")
    if origin is None:
        raise ValueError(f"URDF {name} has no origin")
    xyz = tuple(float(value) for value in origin.attrib.get("xyz", "0 0 0").split())
    rpy = tuple(float(value) for value in origin.attrib.get("rpy", "0 0 0").split())
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"URDF {name} origin is incomplete")
    transform = np.eye(4)
    transform[:3, :3] = _rpy_axes(rpy)
    transform[:3, 3] = xyz
    return transform


def _urdf_fixed_joint_transform(urdf_path: Path, name: str, parent: str, child: str) -> np.ndarray:
    root = ET.parse(Path(urdf_path)).getroot()
    joint = root.find(f'./joint[@name="{name}"]')
    if joint is None or joint.attrib.get("type") != "fixed":
        raise ValueError(f"pinned URDF fixed joint {name} is missing")
    actual_parent = joint.find("parent")
    actual_child = joint.find("child")
    if (
        actual_parent is None
        or actual_child is None
        or actual_parent.attrib.get("link") != parent
        or actual_child.attrib.get("link") != child
    ):
        raise ValueError(f"pinned URDF joint {name} parent/child chain changed")
    return _urdf_origin_transform(joint, f"joint {name}")


def _urdf_front_body_radius(
    urdf_path: Path, start: np.ndarray, end: np.ndarray
) -> tuple[float, list[dict[str, object]]]:
    """Conservatively bound the static `front` gripper body collisions in the front frame."""
    root = ET.parse(Path(urdf_path)).getroot()
    link = root.find('./link[@name="front"]')
    if link is None:
        raise ValueError("pinned URDF front gripper body link is missing")
    details: list[dict[str, object]] = []
    for index, collision in enumerate(link.findall("collision")):
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError(f"front collision {index} has no single geometry")
        kind, shape_node = next(iter([(item.tag, item) for item in geometry]))
        if kind == "box":
            size = np.asarray([float(value) for value in shape_node.attrib.get("size", "").split()])
            if size.shape != (3,) or np.any(size <= 0):
                raise ValueError(f"front box {index} has an invalid size")
            half = size / 2.0
        elif kind == "cylinder":
            radius = float(shape_node.attrib["radius"])
            length = float(shape_node.attrib["length"])
            if radius <= 0 or length <= 0:
                raise ValueError(f"front cylinder {index} has invalid dimensions")
            half = np.asarray((radius, radius, length / 2.0))
        elif kind == "sphere":
            radius = float(shape_node.attrib["radius"])
            if radius <= 0:
                raise ValueError(f"front sphere {index} has an invalid radius")
            half = np.full(3, radius)
        else:
            raise ValueError(f"unsupported front gripper body geometry: {kind}")
        transform = _urdf_origin_transform(collision, f"front collision {index}")
        required = max(
            _point_segment_distance(
                transform[:3, :3] @ (signs * half) + transform[:3, 3], start, end
            )
            for signs in _BOX_CORNERS
        )
        details.append(
            {
                "index": index,
                "geometry": kind,
                "half_extents_m": half.tolist(),
                "required_radius_m": required,
            }
        )
    if not details:
        raise ValueError("pinned URDF front gripper body has no collision primitives")
    return max(float(cast(float, item["required_radius_m"])) for item in details), details


def _gripper_radius(
    cfg: LiveReconciliationConfig,
    catalog: SceneCollisionCatalog,
    objects: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    by_def, _ = _object_index(objects)
    tool = by_def.get("WRIST_CONNECTOR")
    gripper = catalog.geometry.gripper
    if tool is None or gripper is None:
        blockers.append("tool frame or pinned gripper envelope is missing")
        return {"envelope_validation_available": False}, blockers
    atol = cfg.tolerances["rigid_matrix_atol"]
    T_base_wrist = _matrix(tool.get("T_base_owner"), "WRIST_CONNECTOR base transform", atol)
    T_wrist_front = _urdf_fixed_joint_transform(
        cfg.pinned["urdf"],
        "wrist_ft_tool_link_front_joint",
        "wrist_ft_tool_link",
        "front",
    )
    T_base_front = T_base_wrist @ T_wrist_front
    start = np.asarray(gripper.start_tool, dtype=np.float64)
    end = np.asarray(gripper.end_tool, dtype=np.float64)
    shift = np.asarray(gripper.ee_to_front_translation_m, dtype=np.float64)
    if gripper.frame == "front" and (shift.shape != (3,) or not np.isfinite(shift).all()):
        blockers.append("gripper ee-to-front translation is invalid")
        return {"envelope_validation_available": False}, blockers
    if gripper.frame == "wrist_ft_tool_link" and np.any(shift != 0.0):
        blockers.append("legacy wrist-frame gripper envelope cannot carry an ee-to-front offset")

    def measure_fingers(
        segment_start: np.ndarray,
        segment_end: np.ndarray,
        T_base_segment: np.ndarray,
    ) -> tuple[float, list[dict[str, object]]]:
        T_segment_base = np.linalg.inv(T_base_segment)
        required = 0.0
        rows: list[dict[str, object]] = []
        for marker in list(cast(Sequence[str], cfg.expected["gripper_defs"])):
            finger = by_def.get(marker)
            if finger is None:
                blockers.append(f"live export is missing gripper {marker}")
                continue
            shapes = cast(Sequence[Mapping[str, object]], finger.get("shapes", ()))
            if len(shapes) != 3 or any(shape.get("type") != "Box" for shape in shapes):
                blockers.append(f"gripper {marker} does not have three live Box shapes")
                continue
            T_base_owner = _matrix(finger.get("T_base_owner"), f"{marker} base transform", atol)
            finger_max = 0.0
            for index, shape in enumerate(shapes):
                T_owner_shape = _matrix(
                    shape.get("T_owner_shape"), f"{marker} shape {index} transform", atol
                )
                size = np.asarray(_shape_parameters(shape).get("size"), dtype=np.float64)
                if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                    blockers.append(f"gripper {marker} shape {index} has invalid Box size")
                    continue
                T_segment_shape = T_segment_base @ T_base_owner @ T_owner_shape
                finger_max = max(
                    finger_max,
                    max(
                        _point_segment_distance(
                            T_segment_shape[:3, :3] @ (signs * size / 2.0) + T_segment_shape[:3, 3],
                            segment_start,
                            segment_end,
                        )
                        for signs in _BOX_CORNERS
                    ),
                )
            required = max(required, finger_max)
            rows.append({"name": marker, "required_capsule_radius_m": finger_max})
        return required, rows

    if gripper.frame == "front":
        reference_base = T_base_front
    elif gripper.frame == "wrist_ft_tool_link":
        reference_base = T_base_wrist
    else:
        blockers.append(f"unsupported gripper envelope frame: {gripper.frame}")
        return {"envelope_validation_available": False}, blockers

    frame_fingers, finger_rows = measure_fingers(start, end, reference_base)
    wrong_frame_fingers, _ = measure_fingers(start, end, T_base_wrist)
    # Runtime-literal diagnostic: ee_pose is the finger-pad frame, which is -shift from URDF
    # front. Express that tool-frame segment back in front coordinates for comparison only.
    runtime_start, runtime_end = start - shift, end - shift
    runtime_fingers, _ = measure_fingers(runtime_start, runtime_end, T_base_front)
    try:
        frame_body, body_rows = _urdf_front_body_radius(cfg.pinned["urdf"], start, end)
        runtime_body, _ = _urdf_front_body_radius(cfg.pinned["urdf"], runtime_start, runtime_end)
    except (OSError, ValueError, ET.ParseError) as error:
        blockers.append(f"static gripper body evidence is unavailable: {error}")
        frame_body, body_rows, runtime_body = 0.0, [], 0.0

    frame_required = max(frame_fingers, frame_body)
    runtime_required = max(runtime_fingers, runtime_body)
    pinned_radius = float(gripper.radius_m)
    conservative = frame_required <= pinned_radius + cfg.tolerances["primitive_atol"]
    if not conservative:
        blockers.append(
            "pinned gripper capsule does not enclose the frame-correct finger/body boxes"
        )
    return {
        "envelope_validation_available": True,
        "frame": gripper.frame,
        "ee_to_front_translation_m": shift.tolist(),
        "pinned_radius_m": pinned_radius,
        "frame_correct_finger_required_radius_m": frame_fingers,
        "frame_correct_body_required_radius_m": frame_body,
        "frame_correct_required_radius_m": frame_required,
        "wrong_frame_required_radius_m": wrong_frame_fingers,
        "runtime_literal_finger_required_radius_m": runtime_fingers,
        "runtime_literal_body_required_radius_m": runtime_body,
        "runtime_literal_required_radius_m": runtime_required,
        "conservative": conservative,
        "fingers": finger_rows,
        "static_body_primitives": body_rows,
        "mismatch_resolution": (
            "The apparent 0.2365 m finger requirement came from interpreting the pinned "
            "front-frame segment in WRIST_CONNECTOR coordinates. Recovering the URDF front "
            "origin from "
            "ee_pose + [0,0,0.1741] leaves 0.065875 m including static body boxes, below the "
            "unchanged 0.1 m radius."
        ),
        "static_single_pose_only": True,
    }, blockers


def _can_envelope(
    cfg: LiveReconciliationConfig,
    catalog: SceneCollisionCatalog,
    objects: Sequence[Mapping[str, object]],
    state: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    by_def, _ = _object_index(objects)
    tool = by_def.get("WRIST_CONNECTOR")
    can = by_def.get(str(cfg.expected["can_def"]))
    pinned_can = catalog.geometry.carried_objects[0]
    if can is None or tool is None:
        blockers.append("CAN or tool transform is missing from the live export")
        return {"dynamic_state_proven": False}, blockers
    shapes = cast(Sequence[Mapping[str, object]], can.get("shapes", ()))
    if len(shapes) != 1 or shapes[0].get("type") != "Cylinder":
        blockers.append("live CAN is not a single Cylinder")
        return {"dynamic_state_proven": False}, blockers
    parameters = _shape_parameters(shapes[0])
    radius = _finite(parameters.get("radius"), "CAN radius")
    height = _finite(parameters.get("height"), "CAN height")
    static_match = _near(radius, pinned_can.radius_m, cfg) and _near(
        height, pinned_can.height_m, cfg
    )
    if not static_match:
        blockers.append("live CAN cylinder does not match the pinned static envelope")
    atol = cfg.tolerances["rigid_matrix_atol"]
    T_base_tool = _matrix(tool.get("T_base_owner"), "WRIST_CONNECTOR base transform", atol)
    T_tool_base = np.linalg.inv(T_base_tool)
    T_base_can = _matrix(can.get("T_base_owner"), "CAN base transform", atol)
    T_tool_can = T_tool_base @ T_base_can
    detail = {
        "static_geometry_matches_pinned": static_match,
        "live_radius_m": radius,
        "live_height_m": height,
        "tool_frame_center": T_tool_can[:3, 3].tolist(),
        "tool_frame_distance_m": float(np.linalg.norm(T_tool_can[:3, 3])),
        "gripper_opening_from_state": state.get("gripper_opening"),
        "attachment_state": "unknown",
        "release_state": "unknown",
        "carried_pose_propagation_proven": False,
        "dynamic_state_proven": False,
        "reason_static_is_insufficient": (
            "one passive CAN pose and matching radius/height do not establish grasp attachment, "
            "release, or carried-pose propagation during a swept transition"
        ),
    }
    blockers.extend(
        [
            "CAN attachment and release state is not observed",
            "carried-CAN pose propagation is not proven across transition states",
        ]
    )
    return detail, blockers


def _sweep_analysis(
    cfg: LiveReconciliationConfig,
    catalog: SceneCollisionCatalog,
    sweep_contract: Any,
    reference: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    blockers: list[str] = []
    sweep = cfg.sweep
    margin = float(sweep["margin_m"])
    speed = float(sweep["point_speed_bound_m_s"])
    dt = float(sweep["dt_s"])
    configured = int(sweep["configured_substeps"])
    adaptive = int(sweep["adaptive_substeps"])
    required = required_substeps(
        speed_bound_m_s=speed,
        dt_s=dt,
        margin_m=margin,
        minimum=configured,
        maximum=4096,
    )
    if adaptive != required:
        blockers.append("pinned adaptive substep count does not match the conservative requirement")
    primitives = urdf_collision_capsules(
        cfg.pinned["urdf"],
        tuple(sweep_contract.required_arm_links),
        max_point_speed_m_s=speed,
    )
    max_reach = 0.0
    T_base_urdf = catalog.T_base_urdf
    for group in primitives.values():
        for primitive in group:
            for point in (primitive.capsule.start, primitive.capsule.end):
                vector = np.asarray(point, dtype=np.float64)
                base_point = T_base_urdf[:3, :3] @ vector + T_base_urdf[:3, 3]
                max_reach = max(
                    max_reach, float(np.linalg.norm(base_point)) + primitive.capsule.radius_m
                )
    registration_uncertainty = (
        _finite(catalog.base_pose_tolerance["position_m"], "base_pose_position_tolerance")
        + _finite(catalog.base_pose_tolerance["orientation_rad"], "base_pose_orientation_tolerance")
        * max_reach
    )
    joint_bound = _nonnegative_optional(
        sweep["max_joint_displacement_bound_rad"], "sweep.max_joint_displacement_bound_rad"
    )
    geometry_error = _nonnegative_optional(
        sweep["geometry_approximation_error_bound_m"],
        "sweep.geometry_approximation_error_bound_m",
    )
    numerical_error = _nonnegative_optional(
        sweep["numerical_error_bound_m"], "sweep.numerical_error_bound_m"
    )
    transform_error = _nonnegative_optional(
        sweep["transform_error_bound_m"], "sweep.transform_error_bound_m"
    )
    is_global = bool(sweep["observed_one_pose_aabb_undercoverage_is_global_bound"])
    if not is_global:
        geometry_error = None
        blockers.append("one-pose AABB undercoverage is not a global geometry-error bound")
    if sweep_contract.status != "blocked" or sweep_contract.certificate is not None:
        blockers.append("arm-sweep contract is not the blocked template")
    bound = evaluate_sweep_safety_bound(
        margin_m=margin,
        point_speed_bound_m_s=speed,
        dt_s=dt,
        substeps=adaptive,
        max_joint_displacement_bound_rad=joint_bound,
        joint_reach_bound_m=max_reach,
        geometry_approximation_error_bound_m=geometry_error,
        registration_uncertainty_bound_m=registration_uncertainty,
        numerical_error_bound_m=numerical_error,
        transform_error_bound_m=transform_error,
    )
    if not bound.certified:
        blockers.extend(f"swept bound: {item}" for item in bound.blockers)
    errors = reference.get("matched_primitive_errors")
    if not isinstance(errors, Sequence):
        raise ValueError("reference collision fixture has no matched_primitive_errors")
    fixture_undercoverage = max(
        _finite(
            _mapping(cast(object, item), "reference error").get("aabb_undercoverage_m"),
            "reference aabb_undercoverage_m",
        )
        for item in errors
    )
    if not _near(
        fixture_undercoverage,
        float(sweep["observed_one_pose_aabb_undercoverage_m"]),
        cfg,
    ):
        blockers.append("independent reference undercoverage does not match the pinned observation")
    detail = {
        "fixed_eight_step_travel_m": speed * dt / configured,
        "adaptive_substeps": adaptive,
        "adaptive_travel_m": speed * dt / adaptive,
        "max_urdf_reach_m": max_reach,
        "registration_uncertainty_bound_m": registration_uncertainty,
        "independent_reference_max_one_pose_aabb_undercoverage_m": fixture_undercoverage,
        "reference_is_global_bound": is_global,
        "safety_bound": bound.report(),
        "certified": bound.certified,
        "verdict": "FAIL_CLOSED" if not bound.certified else "PROVEN_FOR_SYNTHETIC_CONTRACT",
    }
    return detail, blockers


def reconcile_preserved_live_geometry(
    config_path: Path | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable, non-scored component and swept-safety reconciliation."""
    cfg = load_live_reconciliation_config(config_path)
    geometry = _load_verified_json(cfg.evidence["geometry"])
    diagnostic = _load_verified_json(cfg.evidence["diagnostic"])
    state = _load_verified_json(cfg.evidence["state"])
    reference = _load_verified_json(cfg.evidence["reference_collision_geometry"])
    _load_verified_json(cfg.evidence["provenance"])
    log_item = cfg.evidence["log"]
    if _sha256(log_item.path) != log_item.sha256:
        raise ValueError("preserved log hash changed after config load")

    blockers, objects = _verify_export_contract(cfg, geometry, diagnostic)
    catalog = load_scene_collision_catalog(cfg.pinned["scene_catalog"])
    manifest = load_proto_source_manifest(
        cfg.pinned["proto_manifest"],
        world_path=cfg.pinned["world"],
        expected_world_sha256=str(cfg.expected["world_sha256"]),
    )
    sweep_contract = bindings.load_arm_sweep_config(cfg.pinned["arm_sweep_contract"])

    floor_detail, floor_blockers = _validate_floors(cfg, catalog, objects)
    pipe_detail, pipe_blockers = _trace_pipe_sections(
        cfg, manifest, objects, float(cfg.sweep["point_speed_bound_m_s"])
    )
    rail_detail, rail_blockers = _trace_straight_stairs_rail(cfg, manifest, objects)
    gripper_detail, gripper_blockers = _gripper_radius(cfg, catalog, objects)
    can_detail, can_blockers = _can_envelope(cfg, catalog, objects, state)
    sweep_detail, sweep_blockers = _sweep_analysis(cfg, catalog, sweep_contract, reference)
    table_detail, table_blockers = _trace_table(cfg, manifest, objects, catalog, sweep_detail)

    coverage = build_self_collision_coverage(
        cfg.pinned["urdf"],
        sweep_contract.required_arm_links,
        max_point_speed_m_s=float(cfg.sweep["point_speed_bound_m_s"]),
        gripper_name=catalog.geometry.gripper.name if catalog.geometry.gripper else None,
        carried_object_id=catalog.geometry.carried_objects[0].name
        if catalog.geometry.carried_objects
        else None,
    )
    excluded = sweep_contract.adjacent_pairs
    self_collision_detail: dict[str, object] = {
        **coverage.report(),
        "contract_excluded_pairs": sorted(sorted(pair) for pair in excluded),
        "contract_coverage_validated": validate_contract_pair_coverage(coverage, excluded),
        "adjacency_auto_exempt": False,
        "grasp_contact_auto_exempt": False,
    }
    self_collision_blockers: list[str] = []
    if excluded:
        self_collision_blockers.append("contract attempts to exempt adjacent collision pairs")
    if not coverage.coverage_proven:
        self_collision_blockers.append("URDF collision-body pair coverage is incomplete")
    if not self_collision_detail["contract_coverage_validated"]:
        self_collision_blockers.append("self-collision contract does not check every required pair")

    can_lifecycle = assess_carried_object_state(
        "ambiguous",
        static_geometry_verified=bool(can_detail.get("static_geometry_matches_pinned")),
    )
    can_detail["lifecycle"] = can_lifecycle.report()

    by_def, by_name = _object_index(objects)
    arm_defs = [str(value) for value in cast(Sequence[object], cfg.expected["arm_torso_defs"])]
    missing_arm = [name for name in arm_defs if name not in by_def and name not in by_name]
    if missing_arm:
        blockers.append(f"live export is missing arm/torso nodes: {missing_arm}")
    sensors = [item for item in objects if item.get("type") in {"Astra", "HokuyoUrg04lxug01"}]
    known_types = {
        "Floor",
        "Tiago",
        "Can",
        "PipeSection",
        "Astra",
        "HokuyoUrg04lxug01",
        *{str(by_def[name].get("type")) for name in arm_defs if name in by_def},
        *{str(by_name[name].get("type")) for name in arm_defs if name in by_name},
    }
    other_types: dict[str, int] = {}
    for item in objects:
        type_name = str(item.get("type"))
        if type_name not in known_types:
            other_types[type_name] = other_types.get(type_name, 0) + 1

    export_payload = geometry.get("export")
    self_collision_raw = (
        cast(Mapping[str, object], export_payload).get("self_collision_enabled")
        if isinstance(export_payload, Mapping)
        else None
    )
    component_blockers = [
        *floor_blockers,
        *table_blockers,
        *pipe_blockers,
        *rail_blockers,
        *gripper_blockers,
        *can_blockers,
        *self_collision_blockers,
    ]
    components = [
        _component(
            name="robot_arm_and_torso",
            classification="observed_but_not_conservatively_represented",
            live={"expected_defs": arm_defs, "missing_defs": missing_arm},
            pinned={
                "required_arm_links": list(sweep_contract.required_arm_links),
                "self_collision_enabled_raw": self_collision_raw,
            },
            reason=(
                "live arm/torso bounding objects are observed, but URDF capsule containment is not "
                "owner-reviewed and selfCollision is disabled"
            ),
            blockers=["arm/torso swept containment is not owner-reviewed"],
        ),
        _component(
            name="gripper_fingers",
            classification="matched_and_conservatively_represented"
            if gripper_detail.get("conservative") is True and not gripper_blockers
            else "observed_but_not_conservatively_represented",
            live=gripper_detail,
            pinned={
                "radius_m": catalog.geometry.gripper.radius_m if catalog.geometry.gripper else None,
                "frame": gripper_detail.get("frame"),
            },
            reason=(
                "frame-corrected static finger and body boxes fit the unchanged pinned radius; "
                "dynamic gripper motion still requires the blocked swept certificate"
            ),
            blockers=gripper_blockers,
            conservative_representation=gripper_detail.get("conservative") is True
            and not gripper_blockers,
        ),
        _component(
            name="CAN",
            classification="observed_but_not_conservatively_represented",
            live=can_detail,
            pinned={
                "radius_m": catalog.geometry.carried_objects[0].radius_m,
                "height_m": catalog.geometry.carried_objects[0].height_m,
            },
            reason=(
                "static CAN geometry matches, but attachment/release and carried pose are unproven"
            ),
            blockers=can_blockers,
        ),
        _component(
            name="table",
            classification="missing_from_live_export",
            live=table_detail,
            pinned={"static_box": cast(Sequence[object], catalog.report()["static_boxes"])[0]},
            reason=(
                "table collision exists only in nested PROTO solids; runtime transform is "
                "ungrounded"
            ),
            blockers=table_blockers,
        ),
        _component(
            name="floor",
            classification="matched_and_conservatively_represented"
            if not floor_blockers
            else "observed_but_not_conservatively_represented",
            live=floor_detail,
            pinned={
                "planes": [
                    {
                        "name": item.name,
                        "point_base": list(item.point_base),
                        "normal_base": list(item.normal_base),
                    }
                    for item in catalog.geometry.floors
                ]
            },
            reason=(
                "exactly two horizontal pinned floor planes are matched; other orientations "
                "are rejected"
            ),
            blockers=floor_blockers,
            conservative_representation=not floor_blockers,
        ),
        _component(
            name="PipeSection",
            classification="observed_but_not_conservatively_represented",
            live=pipe_detail,
            pinned={"manifest_status": "unresolved_forwarded_geometry"},
            reason=(
                "live cylinders match pinned instances and yield review envelopes, but "
                "forwarding remains blocked"
            ),
            blockers=pipe_blockers,
        ),
        _component(
            name="StraightStairsRail",
            classification="missing_from_live_export",
            live=rail_detail,
            pinned={"manifest_status": "unresolved_forwarded_geometry"},
            reason=(
                "forwarded TexturedParallelepiped transforms are unpinned and no live instance "
                "grounds them"
            ),
            blockers=rail_blockers,
        ),
        _component(
            name="other_instantiated_obstacles",
            classification="unsupported",
            live={"type_counts": other_types},
            pinned={"unreduced_obstacle_policy": "block_all_without_reviewed_primitives"},
            reason=(
                "remaining obstacle PROTOs are observed but not reduced to reviewed collision "
                "primitives"
            ),
            blockers=["unreduced obstacle primitives remain in the scene"],
        ),
        _component(
            name="onboard_sensors",
            classification="irrelevant_with_documented_justification",
            live={
                "count": len(sensors),
                "types": sorted({str(item.get("type")) for item in sensors}),
            },
            pinned={"policy": "not_task_obstacles"},
            reason=(
                "fixed onboard camera/LiDAR shells are not task obstacles; they are not used as "
                "free-space evidence"
            ),
        ),
    ]
    component_blockers.extend(blockers)
    component_blockers.extend(sweep_blockers)
    deduplicated: list[str] = []
    for item in component_blockers:
        if item not in deduplicated:
            deduplicated.append(item)

    report: dict[str, object] = {
        "schema_version": LIVE_GEOMETRY_RECONCILIATION_SCHEMA_VERSION,
        "status": "diagnostic_not_certified",
        "evidence_verified": True,
        "evidence_hashes": {name: item.sha256 for name, item in cfg.evidence.items()},
        "accepted_inspection": {
            "outcome": diagnostic.get("outcome"),
            "raw_self_collision": (
                cast(Mapping[str, object], geometry.get("export", {})).get("self_collision_enabled")
                if isinstance(geometry.get("export"), Mapping)
                else None
            ),
            "normalized_self_collision": normalize_webots_sfbool(
                self_collision_raw, field_name="selfCollision"
            ),
            "webots_rerun_performed": False,
        },
        "motion_allowed": False,
        "motion_readiness": "NOT_READY",
        "components": components,
        "classification_counts": {
            name: sum(1 for item in components if item["classification"] == name)
            for name in sorted(CLASSIFICATIONS)
        },
        "dynamic_envelopes": {
            "gripper": gripper_detail,
            "can": can_detail,
            "self_collision_pairs": {
                "self_collision_enabled": normalize_webots_sfbool(
                    self_collision_raw, field_name="selfCollision"
                ),
                **self_collision_detail,
            },
            "carried_lifecycle": {
                **carried_lifecycle_report(),
                "current_assessment": can_lifecycle.report(),
            },
            "grasp_contact_exceptions": {
                "permitted_contact_objects": sorted(sweep_contract.permitted_contact_objects),
                "approved": bool(sweep_contract.permitted_contact_objects),
                "reason": (
                    "no owner-reviewed grasp-contact exception exists in the blocked contract"
                ),
            },
        },
        "swept_collision_validation": sweep_detail,
        "blockers": deduplicated,
    }
    return report


__all__ = [
    "CLASSIFICATIONS",
    "LIVE_GEOMETRY_RECONCILIATION_SCHEMA_VERSION",
    "EvidenceFile",
    "LiveReconciliationConfig",
    "expand_table_source_geometry",
    "load_live_reconciliation_config",
    "reconcile_preserved_live_geometry",
]
