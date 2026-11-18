"""Inert checks for preserved live geometry reconciliation and swept-bound fail-closed logic.

No test in this module launches Webots, opens a socket, moves the robot, solves IK, scores an
episode or reads TEST/TRAIN data. All evidence is hash-pinned under ``configs/geometry``.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from vlarobot.eval.worldmind_live_geometry_reconciliation import (
    _load_verified_json,
    _validate_floors,
    _verify_export_contract,
    load_live_reconciliation_config,
    reconcile_preserved_live_geometry,
)
from vlarobot.world_model.arm_geometry import SweptArmCheck
from vlarobot.world_model.arm_sweep import (
    evaluate_sweep_safety_bound,
    required_substeps,
)
from vlarobot.world_model.scene_geometry import load_scene_collision_catalog


def test_config_verifies_every_pinned_artifact_hash_before_use():
    cfg = load_live_reconciliation_config()
    assert cfg.evidence["geometry"].sha256 == (
        "034061c296abcada98b32fb2f22d7b421354a3a90bebf58e5feb6cea8b1a69d9"
    )
    assert len(cfg.evidence) == 7
    assert cfg.evidence["reference_safety_cases"].sha256 == (
        "b5e3d1f6de9a1c3da896a2da961dff95a2713b25c8f31b42102b3ecfa236a188"
    )
    with pytest.raises(ValueError, match="evidence changed after config load: geometry"):
        _load_verified_json(replace(cfg.evidence["geometry"], sha256="0" * 64))


def test_reconciliation_accepts_failed_inspection_and_classifies_all_components():
    report = reconcile_preserved_live_geometry()
    assert report["evidence_verified"] is True
    assert report["accepted_inspection"]["outcome"] == "FAILED"
    assert report["accepted_inspection"]["raw_self_collision"] == 0
    assert report["accepted_inspection"]["normalized_self_collision"] is False
    assert report["accepted_inspection"]["webots_rerun_performed"] is False
    assert report["motion_allowed"] is False
    assert report["motion_readiness"] == "NOT_READY"

    by_name = {item["name"]: item for item in report["components"]}
    assert by_name["floor"]["classification"] == "matched_and_conservatively_represented"
    assert (
        by_name["robot_arm_and_torso"]["classification"]
        == "observed_but_not_conservatively_represented"
    )
    assert by_name["gripper_fingers"]["classification"] == "matched_and_conservatively_represented"
    assert by_name["CAN"]["classification"] == "observed_but_not_conservatively_represented"
    assert by_name["table"]["classification"] == "missing_from_live_export"
    assert by_name["PipeSection"]["classification"] == "observed_but_not_conservatively_represented"
    assert by_name["StraightStairsRail"]["classification"] == "missing_from_live_export"
    assert by_name["other_instantiated_obstacles"]["classification"] == "unsupported"
    assert (
        by_name["onboard_sensors"]["classification"] == "irrelevant_with_documented_justification"
    )
    assert report["classification_counts"]["matched_and_conservatively_represented"] == 2
    assert len(report["blockers"]) >= 6


def test_table_absence_is_grounded_in_nested_proto_structure():
    report = reconcile_preserved_live_geometry()
    table = next(item for item in report["components"] if item["name"] == "table")
    live = table["live"]
    assert live["live_export_count"] == 0
    assert live["root_proto_bounding_object_count"] == 0
    assert live["nested_solidbox_count"] >= 3
    assert live["solidbox_bounding_object_is_box"] is True
    assert live["runtime_collision_tree_grounded"] is False
    assert live["pinned_world_transform_and_size"] == {
        "translation": [-0.23882837, -4.8402928, 0.0],
        "size": [4.5, 1.5, 0.75],
    }
    assert (
        "geometry_probe records only nodes with a direct boundingObject"
        in live["probe_traversal_finding"]
    )
    assert live["status"] == "source_derived_unverified"
    assert live["runtime_collision_tree_grounded"] is False
    expansion = live["source_expansion"]
    assert len(expansion["boxes"]) == 5
    assert expansion["coarse_catalog_proxy_contains_all_boxes"] is True
    assert expansion["source_hashes"]["world_sha256"] == (
        "35d87f3601813a47e932bb8558740443710ba15421f07e1f3c67c6e5fd027706"
    )
    assert "evaluated translation/rotation/size" in live["missing_runtime_observation"]


def test_forwarded_geometry_is_traced_and_unsupported_forwarding_stays_blocked():
    report = reconcile_preserved_live_geometry()
    pipe = next(item for item in report["components"] if item["name"] == "PipeSection")
    rail = next(item for item in report["components"] if item["name"] == "StraightStairsRail")
    assert pipe["live"]["forwarding_chain"]["boundingObject"] == "boundingObject USE PIPE"
    assert pipe["live"]["forwarding_chain"]["definition"] == "DEF PIPE Cylinder"
    assert pipe["live"]["instance_count"] == 7
    assert all(item["matches"] for item in pipe["live"]["instances"])
    assert len(pipe["live"]["conservative_capsule_envelopes"]) == 7
    assert pipe["live"]["runtime_certified"] is False
    assert rail["live"]["live_export_count"] == 0
    assert rail["live"]["conservative_enclosure_available"] is False
    assert rail["live"]["unresolved_leaf"] == "TexturedParallelepiped"


def test_dynamic_envelopes_distinguish_frame_artifacts_from_missing_lifecycle_evidence():
    report = reconcile_preserved_live_geometry()
    dynamic = report["dynamic_envelopes"]
    gripper = dynamic["gripper"]
    can = dynamic["can"]
    assert gripper["conservative"] is True
    assert gripper["frame"] == "front"
    assert gripper["ee_to_front_translation_m"] == [0.0, 0.0, 0.1741]
    assert gripper["pinned_radius_m"] == 0.1
    assert gripper["frame_correct_required_radius_m"] == pytest.approx(0.065875008915307)
    assert gripper["wrong_frame_required_radius_m"] == pytest.approx(0.23651330651909497)
    assert gripper["runtime_literal_required_radius_m"] == pytest.approx(0.192581759418895)
    assert gripper["wrong_frame_required_radius_m"] > gripper["pinned_radius_m"]
    assert gripper["runtime_literal_required_radius_m"] > gripper["pinned_radius_m"]
    assert can["static_geometry_matches_pinned"] is True
    assert can["attachment_state"] == "unknown"
    assert can["release_state"] == "unknown"
    assert can["carried_pose_propagation_proven"] is False

    lifecycle = dynamic["carried_lifecycle"]["current_assessment"]
    assert lifecycle["state"] == "ambiguous"
    assert lifecycle["fail_closed"] is True
    assert lifecycle["carried_geometry_available"] is False
    assert lifecycle["contact_claim_allowed"] is False

    self_collision = dynamic["self_collision_pairs"]
    assert self_collision["counts"]["all_capsule_pairs"] == 171
    assert self_collision["counts"]["checked_capsule_pairs"] == 171
    assert self_collision["counts"]["adjacent_capsule_pairs"] == 39
    assert self_collision["counts"]["non_adjacent_capsule_pairs"] == 117
    assert self_collision["counts"]["gripper_arm_pairs"] == 19
    assert self_collision["counts"]["carried_robot_pairs"] == 20
    assert self_collision["counts"]["uncovered_collision_bodies"] == 42
    assert self_collision["coverage_proven"] is False
    assert self_collision["contract_coverage_validated"] is False
    assert self_collision["adjacency_auto_exempt"] is False
    assert self_collision["grasp_contact_auto_exempt"] is False
    assert dynamic["grasp_contact_exceptions"]["approved"] is False


def test_arbitrary_floor_orientation_is_rejected():
    cfg = load_live_reconciliation_config()
    catalog = load_scene_collision_catalog(cfg.pinned["scene_catalog"])
    geometry = _load_verified_json(cfg.evidence["geometry"])
    _, objects = _verify_export_contract(
        cfg, geometry, _load_verified_json(cfg.evidence["diagnostic"])
    )
    _, clean_blockers = _validate_floors(cfg, catalog, objects)
    assert clean_blockers == []

    tilted = copy.deepcopy(list(objects))
    floor = next(item for item in tilted if item.get("type") == "Floor")
    matrix = np.asarray(floor["T_world_owner"], dtype=np.float64).reshape(4, 4)
    angle = np.pi / 2.0
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    matrix[:3, :3] = rotation @ matrix[:3, :3]
    floor["T_world_owner"] = matrix.reshape(16).tolist()
    _, blockers = _validate_floors(cfg, catalog, tilted)
    assert any("horizontal" in item or "matched exactly" in item for item in blockers)


def test_safety_bound_is_strict_and_fails_closed_on_every_unknown_term():
    certified = evaluate_sweep_safety_bound(
        margin_m=0.1,
        point_speed_bound_m_s=1.0,
        dt_s=0.1,
        substeps=10,
        max_joint_displacement_bound_rad=0.01,
        joint_reach_bound_m=1.0,
        geometry_approximation_error_bound_m=0.01,
        registration_uncertainty_bound_m=0.01,
        numerical_error_bound_m=0.0,
        transform_error_bound_m=0.0,
    )
    assert certified.certified is True
    assert certified.required_margin_m == pytest.approx(0.03)
    assert certified.residual_m == pytest.approx(0.07)

    boundary = evaluate_sweep_safety_bound(
        margin_m=0.03,
        point_speed_bound_m_s=1.0,
        dt_s=0.1,
        substeps=10,
        max_joint_displacement_bound_rad=0.01,
        joint_reach_bound_m=1.0,
        geometry_approximation_error_bound_m=0.01,
        registration_uncertainty_bound_m=0.01,
        numerical_error_bound_m=0.0,
        transform_error_bound_m=0.0,
    )
    assert boundary.certified is False
    assert boundary.residual_m == pytest.approx(0.0)
    assert any("strictly dominate" in item for item in boundary.blockers)

    missing = evaluate_sweep_safety_bound(
        margin_m=1.0,
        point_speed_bound_m_s=1.0,
        dt_s=0.1,
        substeps=10,
        geometry_approximation_error_bound_m=None,
        registration_uncertainty_bound_m=0.01,
    )
    assert missing.certified is False
    assert any("joint displacement" in item for item in missing.blockers)
    assert any("geometry approximation" in item for item in missing.blockers)
    assert any("numerical error" in item for item in missing.blockers)
    assert any("transform error" in item for item in missing.blockers)
    assert missing.required_margin_m is None
    assert missing.residual_m is None

    joint_dominated = evaluate_sweep_safety_bound(
        margin_m=1.0,
        point_speed_bound_m_s=1.0,
        dt_s=1.0,
        substeps=1,
        max_joint_displacement_bound_rad=2.0,
        joint_reach_bound_m=0.5,
        geometry_approximation_error_bound_m=0.0,
        registration_uncertainty_bound_m=0.0,
        numerical_error_bound_m=0.0,
        transform_error_bound_m=0.0,
    )
    assert joint_dominated.joint_motion_allowance_m == pytest.approx(1.0)
    assert joint_dominated.certified is False  # exact boundary is never a pass


def test_support_errors_reduce_adaptive_substep_budget_and_can_consume_margin():
    base = {
        "speed_bound_m_s": 28.49316700274912,
        "dt_s": 0.032,
        "margin_m": 0.02,
        "minimum": 8,
        "maximum": 4096,
    }
    assert required_substeps(**base) == 46
    assert required_substeps(**base, geometry_error_bound_m=0.01) == 92
    assert (
        required_substeps(
            **base,
            geometry_error_bound_m=0.001,
            registration_uncertainty_bound_m=0.004952867331749651,
            numerical_error_bound_m=0.0005,
            transform_error_bound_m=0.0005,
        )
        == 70
    )
    with pytest.raises(Exception, match="consume the collision margin"):
        required_substeps(**base, geometry_error_bound_m=0.02)


def test_swept_checker_rejects_invalid_support_error_contracts():
    with pytest.raises(ValueError, match="geometry error bound"):
        SweptArmCheck(
            lambda state: (),
            required_links=("arm",),
            required_objects=frozenset(),
            scene_complete=True,
            dt_s=0.01,
            margin_m=0.01,
            adjacent_pairs=frozenset(),
            geometry_error_bound_m=-1.0,
        )


def test_preserved_sweep_analysis_reports_fail_closed_with_real_numbers():
    report = reconcile_preserved_live_geometry()
    sweep = report["swept_collision_validation"]
    bound = sweep["safety_bound"]
    assert sweep["adaptive_substeps"] == 46
    assert sweep["adaptive_travel_m"] == pytest.approx(0.019821333567129826)
    assert bound["point_motion_allowance_m"] == pytest.approx(0.019821333567129826)
    assert bound["registration_uncertainty_bound_m"] == pytest.approx(0.004952867331749651)
    assert sweep["independent_reference_max_one_pose_aabb_undercoverage_m"] == pytest.approx(
        0.0010469683238133465
    )
    assert bound["numerical_error_bound_m"] is None
    assert bound["transform_error_bound_m"] is None
    assert bound["required_margin_m"] is None
    assert bound["residual_m"] is None
    assert sweep["reference_is_global_bound"] is False
    assert bound["certified"] is False
    assert sweep["verdict"] == "FAIL_CLOSED"
    assert any("numerical error" in item for item in bound["blockers"])
    assert any("transform error" in item for item in bound["blockers"])
