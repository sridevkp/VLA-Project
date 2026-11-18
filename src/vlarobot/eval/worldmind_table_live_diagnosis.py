"""Pure offline diagnosis of the preserved five-box Webots table inspection.

This module never launches Webots, opens a socket, moves the robot, solves IK, reads a data
split or writes evidence. It verifies every recorded artifact hash before parsing JSON, then
audits the preserved reconciliation and derives an independent live-base transform from the
already captured robot-base pose. The original pinned-base comparison is retained unchanged;
the live-base comparison is additive diagnostic arithmetic, not a pass or an approval.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from vlarobot.eval.worldmind_webots_diagnostic import AUTHORITATIVE_REPORT_SHA256
from vlarobot.eval.worldmind_webots_geometry_inspection import (
    GeometryInspectionConfig,
    _expected_table_geometry,
    preflight_geometry_inspection,
)
from vlarobot.world_model.scene_geometry import load_scene_collision_catalog

TABLE_LIVE_DIAGNOSIS_SCHEMA_VERSION = 1
TABLE_LIVE_LOG_NAME = "worldmind-webots-geometry-inspection.log"
TABLE_LIVE_EVIDENCE_FILES: tuple[str, ...] = (
    "diagnostic.json",
    "geometry.json",
    "state.json",
    "provenance.json",
    TABLE_LIVE_LOG_NAME,
)
TABLE_LIVE_V1_EVIDENCE_SHA256: Mapping[str, str] = {
    "diagnostic.json": "24673d587adf6a77ccb3eacb053a3123b1f8404a634945d524a9745cc34a6921",
    "geometry.json": "09adc279a4709ad8b79219d2b33748362c51e0fd32f84f44fce37e60d7d32607",
    "state.json": "061a477bed8de7eab202ce49f603a66613da0079d52442bd4f08a5d0dcd16287",
    "provenance.json": "67e028a83369de00e164a21c1299a4a78296b69f166ea3ed5e80f7c2e4af8e0a",
    TABLE_LIVE_LOG_NAME: "622c0d4559cd0ddc54bbc7f22ec6cc145b5bf099758791433c860611b9c1ec80",
}
_LOCAL_WORLD_CHECK_NAMES: tuple[str, ...] = (
    "local_translation_m",
    "local_rotation_rad",
    "size_m",
    "enable_bounding_object",
    "bounding_object_size_m",
    "T_world_translation_m",
    "T_world_rotation_rad",
)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} is not a SHA-256 digest")
    return digest


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _string_sequence(value: object, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return [str(item) for item in value]


def _vector(value: object, size: int, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain exactly {size} finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain exactly {size} finite numbers")
        numbers.append(number)
    return numbers


def verify_table_evidence_hashes(
    evidence_dir: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Hash and pin every table-live artifact before any JSON content is consumed."""
    root = Path(evidence_dir)
    if set(expected_hashes) != set(TABLE_LIVE_EVIDENCE_FILES):
        raise ValueError("table-live evidence hash set is incomplete or ambiguous")
    verified: dict[str, str] = {}
    for name in TABLE_LIVE_EVIDENCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"table-live evidence file is missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = _digest(expected_hashes[name], f"expected hash for {name}")
        if digest != expected:
            raise ValueError(f"table-live evidence hash mismatch: {name}")
        verified[name] = digest
    return verified


def pose4_from_xyzw(position: object, orientation: object) -> np.ndarray:
    """Build a rigid world pose from a finite xyzw quaternion and translation."""
    translation = _vector(position, 3, "pose position")
    quaternion = _vector(orientation, 4, "pose orientation (xyzw)")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("pose orientation must be a unit xyzw quaternion")
    x, y, z, w = quaternion
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return _pose4(rotation, translation)


def _pose4(rotation: object, translation: object) -> np.ndarray:
    matrix_rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix_translation = np.asarray(translation, dtype=float).reshape(3)
    return _require_rigid(_compose_pose(matrix_rotation, matrix_translation))


def _compose_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _require_rigid(matrix: np.ndarray, name: str = "transform") -> np.ndarray:
    result = np.asarray(matrix, dtype=float).reshape(4, 4)
    rotation = result[:3, :3]
    rigid = np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0)
    orthonormal = np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0)
    if not (rigid and orthonormal and np.linalg.det(rotation) > 0.0):
        raise ValueError(f"{name} is not a finite rigid transform")
    return result


def _pose4_from_flat(value: object, name: str) -> np.ndarray:
    numbers = _vector(value, 16, name)
    matrix = np.asarray(numbers, dtype=float).reshape(4, 4)
    return _require_rigid(matrix, name)


def _rotation_geodesic(observed: np.ndarray, expected: np.ndarray) -> float:
    product = np.asarray(observed, dtype=float) @ np.asarray(expected, dtype=float).T
    cosine = (float(np.trace(product)) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    if not math.isfinite(angle):
        raise ValueError("rotation geodesic is not finite")
    return angle


def _compare_poses(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    return {
        "max_abs_translation_error_m": float(np.max(np.abs(observed[:3, 3] - expected[:3, 3]))),
        "rotation_geodesic_error_rad": _rotation_geodesic(observed[:3, :3], expected[:3, :3]),
    }


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(dict[str, object], payload)


def _verify_recorded_source_hashes(
    provenance: Mapping[str, object],
    diagnostic: Mapping[str, object],
    preflight_hashes: Mapping[str, str | None],
) -> None:
    authoritative = provenance.get("authoritative_inert_report_sha256")
    if authoritative != AUTHORITATIVE_REPORT_SHA256:
        raise ValueError("preserved provenance reports a different authoritative report hash")
    config_hash = provenance.get("config_sha256")
    if config_hash != preflight_hashes.get("config"):
        raise ValueError("preserved provenance config hash does not match the pinned config")
    for name, digest in preflight_hashes.items():
        if name == "config" or digest is None:
            continue
        recorded = provenance.get(f"{name}_sha256")
        if recorded != digest:
            raise ValueError(f"preserved provenance hash mismatch: {name}")
    geometry_hashes = _mapping(diagnostic.get("geometry_hashes"), "geometry_hashes")
    if set(geometry_hashes) != set(preflight_hashes):
        raise ValueError("preserved diagnostic hash set differs from the pinned geometry contract")
    for name, digest in preflight_hashes.items():
        if geometry_hashes.get(name) != digest:
            raise ValueError(f"preserved diagnostic hash mismatch: {name}")
    if diagnostic.get("running_world_sha256_observed") != preflight_hashes.get("world"):
        raise ValueError("preserved running-world hash does not match the pinned world")


def _recorded_check(record: Mapping[str, object], name: str) -> Mapping[str, object]:
    checks = record.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise ValueError(f"preserved main-frame checks are invalid for {name}")
    matches = [item for item in checks if isinstance(item, Mapping) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"preserved main-frame check is missing or ambiguous: {name}")
    return matches[0]


def _matches_recorded(value: float, recorded: object, name: str) -> bool:
    numeric = isinstance(recorded, (int, float)) and not isinstance(recorded, bool)
    expected = recorded if numeric else None
    if expected is None:
        raise ValueError(f"preserved {name} error is not numeric")
    return math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12)


def diagnose_preserved_table_evidence(
    config: GeometryInspectionConfig,
    evidence_dir: Path,
    expected_hashes: Mapping[str, str] = TABLE_LIVE_V1_EVIDENCE_SHA256,
) -> dict[str, object]:
    """Audit preserved table-live evidence and explain both observed gaps without relaxing gates."""
    evidence_hashes = verify_table_evidence_hashes(evidence_dir, expected_hashes)
    preflight = preflight_geometry_inspection(config)
    if preflight.blockers:
        raise ValueError(f"pinned geometry preflight failed: {'; '.join(preflight.blockers)}")
    root = Path(evidence_dir)
    diagnostic = _load_json(root / "diagnostic.json")
    geometry = _load_json(root / "geometry.json")
    state = _load_json(root / "state.json")
    provenance = _load_json(root / "provenance.json")
    _verify_recorded_source_hashes(provenance, diagnostic, preflight.hashes)

    if diagnostic.get("motion_allowed") is not False or diagnostic.get("motion_readiness") != (
        "NOT_READY"
    ):
        raise ValueError("preserved diagnostic does not maintain the motion veto")
    if diagnostic.get("no_actuation") is not True:
        raise ValueError("preserved diagnostic does not declare no actuation")
    forbidden_call_fields = {
        "apply": "backend_apply_calls",
        "step": "backend_step_calls",
        "reset": "backend_reset_calls",
        "export_kinematics": "backend_export_calls",
        "inspect": "backend_inspect_calls",
        "get_ground_truth": "backend_ground_truth_calls",
    }
    for name, field_name in forbidden_call_fields.items():
        if diagnostic.get(field_name) != 0:
            raise ValueError(f"preserved diagnostic reports forbidden calls: {name}")
    if diagnostic.get("motor_commands_sent") != 0 or diagnostic.get("physics_step_count") != 0:
        raise ValueError("preserved diagnostic reports motion or physics activity")

    export = _mapping(geometry.get("export"), "geometry export")
    raw_targets = _mapping(export.get("table_targets"), "raw table_targets")
    preserved_report = _mapping(geometry.get("table_target_reconciliation"), "preserved report")
    raw_rows = raw_targets.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("preserved raw table rows are invalid")
    rows = {
        str(item.get("source_name")): item
        for item in raw_rows
        if isinstance(item, Mapping) and isinstance(item.get("source_name"), str)
    }
    preserved_targets = preserved_report.get("targets")
    if not isinstance(preserved_targets, Sequence) or isinstance(preserved_targets, (str, bytes)):
        raise ValueError("preserved reconciliation targets are invalid")
    reports = {
        str(item.get("source_name")): item
        for item in preserved_targets
        if isinstance(item, Mapping) and isinstance(item.get("source_name"), str)
    }
    main_raw = _mapping(rows.get("main frame"), "preserved main-frame raw row")
    main_report = _mapping(reports.get("main frame"), "preserved main-frame reconciliation")
    transforms = _mapping(main_raw.get("transforms"), "preserved main-frame transforms")
    observed_T_base = _pose4_from_flat(transforms.get("T_base"), "observed T_base")

    expected_geometry = _expected_table_geometry(config)
    expected_boxes = expected_geometry.get("boxes")
    if not isinstance(expected_boxes, Sequence) or isinstance(expected_boxes, (str, bytes)):
        raise ValueError("pinned source expansion produced no boxes")
    expected_main = next(
        (
            _mapping(item, "expected main frame")
            for item in expected_boxes
            if isinstance(item, Mapping) and item.get("name") == "main frame"
        ),
        None,
    )
    if expected_main is None:
        raise ValueError("pinned source expansion has no main frame")
    T_world_target = _pose4(
        expected_main.get("world_rotation_matrix"), expected_main.get("world_translation_m")
    )

    live_pose = _mapping(state.get("base_pose_world"), "state base_pose_world")
    T_world_base_live = pose4_from_xyzw(live_pose.get("position"), live_pose.get("orientation"))
    catalog = load_scene_collision_catalog(config.scene_catalog_path)
    T_world_base_pinned = _require_rigid(catalog.T_world_base, "pinned T_world_base")
    T_base_urdf = _require_rigid(catalog.T_base_urdf, "pinned T_base_urdf")

    robot_def = diagnostic.get("robot_def_observed")
    objects = export.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("preserved geometry export objects are invalid")
    robot_object = next(
        (item for item in objects if isinstance(item, Mapping) and item.get("def") == robot_def),
        None,
    )
    if robot_object is None:
        raise ValueError("preserved geometry export has no cross-checked robot object")
    T_world_base_export = _pose4_from_flat(
        robot_object.get("T_world_owner"), "exported robot T_world_owner"
    )
    base_world_consistency = _compare_poses(T_world_base_live, T_world_base_export)

    T_base_pinned_expected = np.linalg.inv(T_world_base_pinned) @ T_world_target
    T_base_live_expected = np.linalg.inv(T_world_base_live) @ T_world_target
    pinned_comparison = _compare_poses(observed_T_base, T_base_pinned_expected)
    live_comparison = _compare_poses(observed_T_base, T_base_live_expected)
    alternatives = {
        "wrong_order_pinned": T_world_target @ np.linalg.inv(T_world_base_pinned),
        "wrong_order_live": T_world_target @ np.linalg.inv(T_world_base_live),
        "urdf_premultiplied_live": T_base_urdf @ T_base_live_expected,
        "urdf_inverse_premultiplied_live": np.linalg.inv(T_base_urdf) @ T_base_live_expected,
    }
    alternative_comparisons = {
        name: _compare_poses(observed_T_base, candidate) for name, candidate in alternatives.items()
    }

    recorded_translation = _recorded_check(main_report, "T_base_translation_m")
    recorded_rotation = _recorded_check(main_report, "T_base_rotation_rad")
    reproduction_ok = _matches_recorded(
        pinned_comparison["max_abs_translation_error_m"],
        recorded_translation.get("max_abs_error"),
        "T_base translation",
    ) and _matches_recorded(
        pinned_comparison["rotation_geodesic_error_rad"],
        recorded_rotation.get("angle"),
        "T_base rotation",
    )
    if not reproduction_ok:
        raise ValueError("preserved T_base mismatch cannot be reproduced from pinned sources")

    evaluated_raw = _mapping(main_raw.get("evaluated"), "preserved main-frame evaluated fields")
    local_world_ok = evaluated_raw.get(
        "bounding_object_type"
    ) == config.table_targets.bounding_shape_type and all(
        _recorded_check(main_report, name).get("matched") is True
        for name in _LOCAL_WORLD_CHECK_NAMES
    )
    base_delta = {
        "translation_delta_m": (T_world_base_live[:3, 3] - T_world_base_pinned[:3, 3]).tolist(),
        "max_abs_translation_delta_m": float(
            np.max(np.abs(T_world_base_live[:3, 3] - T_world_base_pinned[:3, 3]))
        ),
        "rotation_delta_rad": _rotation_geodesic(
            T_world_base_live[:3, :3], T_world_base_pinned[:3, :3]
        ),
        "pinned_probe_tolerance_m": config.table_targets.transform_translation_tolerance_m,
        "pinned_probe_tolerance_rad": config.table_targets.transform_rotation_tolerance_rad,
        "scene_catalog_position_tolerance_m": catalog.base_pose_tolerance["position_m"],
        "scene_catalog_orientation_tolerance_rad": catalog.base_pose_tolerance["orientation_rad"],
    }
    strict_t = config.table_targets.transform_translation_tolerance_m
    strict_r = config.table_targets.transform_rotation_tolerance_rad
    pinned_within_tolerance = (
        pinned_comparison["max_abs_translation_error_m"] <= strict_t
        and pinned_comparison["rotation_geodesic_error_rad"] <= strict_r
    )
    live_within_tolerance = (
        live_comparison["max_abs_translation_error_m"] <= strict_t
        and live_comparison["rotation_geodesic_error_rad"] <= strict_r
    )
    rotation_delta_matches_residual = math.isclose(
        base_delta["rotation_delta_rad"],
        pinned_comparison["rotation_geodesic_error_rad"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if pinned_within_tolerance:
        base_diagnosis = "pinned_base_comparison_matches"
    elif (
        local_world_ok
        and live_within_tolerance
        and rotation_delta_matches_residual
        and base_world_consistency["max_abs_translation_error_m"] <= 1e-12
        and base_world_consistency["rotation_geodesic_error_rad"] <= 1e-12
    ):
        base_diagnosis = "consistent_pinned_vs_live_base_pose_drift"
    else:
        base_diagnosis = "unexplained_or_alternative_frame_error"

    expected_names = [
        str(item.get("name"))
        for item in expected_boxes
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ]
    missing_feet = [
        name
        for name in expected_names
        if name.startswith("foot ") and str(rows.get(name, {}).get("status")) == "missing"
    ]
    table_candidates = raw_targets.get("table_candidates")
    probe_errors = raw_targets.get("errors")
    main_runtime_id = _mapping(main_raw.get("runtime_node"), "main-frame runtime_node").get("id")
    id_collapse_indicators = (
        bool(missing_feet)
        and main_runtime_id == -1
        and probe_errors == []
        and isinstance(table_candidates, Sequence)
        and len(table_candidates) == 1
    )
    feet_diagnosis = {
        "pinned_source_names": expected_names,
        "missing_runtime_rows": missing_feet,
        "main_frame_runtime_id": main_runtime_id,
        "table_candidate_count": (
            len(table_candidates) if isinstance(table_candidates, Sequence) else None
        ),
        "probe_errors": probe_errors,
        "diagnosis": (
            "probe_id_sentinel_collapse_diagnosed"
            if id_collapse_indicators
            else "unexplained_or_genuinely_absent"
        ),
        "runtime_grounded": False,
        "consequence": (
            "Missing feet are not closed by this evidence; source arithmetic is never "
            "substituted for runtime observations."
        ),
    }

    return {
        "schema_version": TABLE_LIVE_DIAGNOSIS_SCHEMA_VERSION,
        "evidence_dir": str(root),
        "evidence_sha256": dict(evidence_hashes),
        "source_hash_verification": {
            "pinned_preflight_blockers": list(preflight.blockers),
            "provenance_geometry_hashes_verified": True,
        },
        "preserved_outcome": {
            "outcome": diagnostic.get("outcome"),
            "summary": dict(_mapping(preserved_report.get("summary"), "preserved summary")),
            "runtime_grounded": preserved_report.get("runtime_grounded"),
            "strict_tolerances_retained": {
                "translation_m": strict_t,
                "rotation_rad": strict_r,
            },
        },
        "feet_diagnosis": feet_diagnosis,
        "base_pose_diagnosis": {
            "diagnosis": base_diagnosis,
            "local_and_world_checks_match": local_world_ok,
            "original_pinned_comparison_retained": pinned_comparison,
            "derived_live_base_comparison": live_comparison,
            "pinned_within_strict_tolerance": pinned_within_tolerance,
            "live_within_strict_tolerance": live_within_tolerance,
            "base_pose_delta": base_delta,
            "base_world_frame_consistency": base_world_consistency,
            "alternative_transform_comparisons": alternative_comparisons,
            "pinned_T_world_base": T_world_base_pinned.tolist(),
            "live_T_world_base": T_world_base_live.tolist(),
            "pinned_T_base_expected": T_base_pinned_expected.tolist(),
            "live_T_base_expected": T_base_live_expected.tolist(),
            "observed_T_base": observed_T_base.tolist(),
            "pinned_T_base_urdf": T_base_urdf.tolist(),
            "composition": "T_BT = inverse(T_WB) @ T_WT",
            "tolerance_changed": False,
        },
        "motion": {
            "motion_allowed": diagnostic.get("motion_allowed"),
            "motion_readiness": diagnostic.get("motion_readiness"),
            "motion_blockers": _string_sequence(
                diagnostic.get("motion_blockers", ()), "motion_blockers"
            ),
        },
        "runtime_grounded": preserved_report.get("runtime_grounded") is True,
    }


__all__ = [
    "TABLE_LIVE_EVIDENCE_FILES",
    "TABLE_LIVE_V1_EVIDENCE_SHA256",
    "TABLE_LIVE_DIAGNOSIS_SCHEMA_VERSION",
    "diagnose_preserved_table_evidence",
    "pose4_from_xyzw",
    "verify_table_evidence_hashes",
]
