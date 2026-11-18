"""Inert offline audit of the preserved five-box Webots table inspection.

No test launches Webots, opens a socket, moves a robot, solves IK, reads a data split or writes
to the preserved evidence directory. Every JSON artifact is hash-verified before it is parsed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vlarobot.eval.worldmind_table_live_diagnosis import (
    TABLE_LIVE_V1_EVIDENCE_SHA256,
    diagnose_preserved_table_evidence,
)
from vlarobot.eval.worldmind_webots_geometry_inspection import load_geometry_inspection_config

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "worldmind_webots_geometry_inspection.yaml"
)
EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "worldmind"
    / "validation-s0-webots-table-live-v1"
)


def _diagnosis() -> dict[str, object]:
    config = load_geometry_inspection_config(CONFIG_PATH)
    report = diagnose_preserved_table_evidence(config, EVIDENCE_DIR, TABLE_LIVE_V1_EVIDENCE_SHA256)
    assert isinstance(report, dict)
    return report


def test_preserved_evidence_hashes_and_source_hashes_are_verified_before_use():
    report = _diagnosis()
    assert report["evidence_sha256"] == dict(TABLE_LIVE_V1_EVIDENCE_SHA256)
    source = report["source_hash_verification"]
    assert isinstance(source, dict)
    assert source["pinned_preflight_blockers"] == []
    assert source["provenance_geometry_hashes_verified"] is True


def test_preserved_outcome_and_motion_veto_are_unchanged():
    report = _diagnosis()
    preserved = report["preserved_outcome"]
    assert isinstance(preserved, dict)
    assert preserved["outcome"] == "FAILED"
    assert preserved["summary"] == {
        "Matched": 0,
        "Mismatched": 1,
        "Missing": 4,
        "Unsupported": 0,
    }
    assert preserved["runtime_grounded"] is False
    tolerances = preserved["strict_tolerances_retained"]
    assert tolerances == {"translation_m": 0.0001, "rotation_rad": 0.0001}
    motion = report["motion"]
    assert motion == {
        "motion_allowed": False,
        "motion_readiness": "NOT_READY",
        "motion_blockers": [
            "diagnostic_no_actuation_mode",
            "unverified-arm-sweep",
            "unresolved_forwarded_geometry",
            "unverified-runtime-table-transform",
        ],
    }


def test_foot_gap_is_diagnosed_without_backfilling_runtime_rows():
    report = _diagnosis()
    feet = report["feet_diagnosis"]
    assert isinstance(feet, dict)
    assert feet["pinned_source_names"] == [
        "main frame",
        "foot 1",
        "foot 2",
        "foot 3",
        "foot 4",
    ]
    assert feet["missing_runtime_rows"] == ["foot 1", "foot 2", "foot 3", "foot 4"]
    assert feet["main_frame_runtime_id"] == -1
    assert feet["table_candidate_count"] == 1
    assert feet["probe_errors"] == []
    assert feet["diagnosis"] == "probe_id_sentinel_collapse_diagnosed"
    assert feet["runtime_grounded"] is False
    assert report["runtime_grounded"] is False


def test_base_mismatch_is_explained_by_pinned_vs_live_base_pose():
    report = _diagnosis()
    base = report["base_pose_diagnosis"]
    assert isinstance(base, dict)
    assert base["diagnosis"] == "consistent_pinned_vs_live_base_pose_drift"
    assert base["local_and_world_checks_match"] is True
    assert base["pinned_within_strict_tolerance"] is False
    assert base["live_within_strict_tolerance"] is True
    assert base["tolerance_changed"] is False
    assert base["composition"] == "T_BT = inverse(T_WB) @ T_WT"

    original = base["original_pinned_comparison_retained"]
    derived = base["derived_live_base_comparison"]
    assert isinstance(original, dict)
    assert isinstance(derived, dict)
    assert original["max_abs_translation_error_m"] == pytest.approx(0.002728548164635969, abs=1e-15)
    assert original["rotation_geodesic_error_rad"] == pytest.approx(0.003505397960478432, abs=1e-15)
    assert derived["max_abs_translation_error_m"] <= 1e-12
    assert derived["rotation_geodesic_error_rad"] <= 1e-12

    delta = base["base_pose_delta"]
    assert isinstance(delta, dict)
    assert delta["max_abs_translation_delta_m"] == pytest.approx(0.0002759895230020071, abs=1e-15)
    assert delta["rotation_delta_rad"] == pytest.approx(0.003505397960478432, abs=1e-15)
    assert delta["pinned_probe_tolerance_m"] == 0.0001
    assert delta["pinned_probe_tolerance_rad"] == 0.0001


def test_world_frame_consistency_composition_order_and_registration_discrepancy():
    report = _diagnosis()
    base = report["base_pose_diagnosis"]
    assert isinstance(base, dict)
    consistency = base["base_world_frame_consistency"]
    assert isinstance(consistency, dict)
    assert consistency["max_abs_translation_error_m"] <= 1e-12
    assert consistency["rotation_geodesic_error_rad"] <= 1e-9

    alternatives = base["alternative_transform_comparisons"]
    assert isinstance(alternatives, dict)
    wrong_pinned = alternatives["wrong_order_pinned"]
    wrong_live = alternatives["wrong_order_live"]
    urdf = alternatives["urdf_premultiplied_live"]
    urdf_inverse = alternatives["urdf_inverse_premultiplied_live"]
    assert wrong_pinned["max_abs_translation_error_m"] > 5.0
    assert wrong_live["max_abs_translation_error_m"] > 5.0
    assert urdf["rotation_geodesic_error_rad"] > 0.07
    assert urdf_inverse["rotation_geodesic_error_rad"] > 0.07


def test_wrong_evidence_hash_fails_closed_before_parsing():
    config = load_geometry_inspection_config(CONFIG_PATH)
    hashes = dict(TABLE_LIVE_V1_EVIDENCE_SHA256)
    hashes["geometry.json"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch: geometry.json"):
        diagnose_preserved_table_evidence(config, EVIDENCE_DIR, hashes)
