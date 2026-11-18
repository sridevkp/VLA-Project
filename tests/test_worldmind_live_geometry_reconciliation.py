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
    assert len(cfg.evidence) == 6
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
    assert (
        by_name["gripper_fingers"]["classification"]
        == "observed_but_not_conservatively_represented"
    )
    assert by_name["CAN"]["classification"] == "observed_but_not_conservatively_represented"
    assert by_name["table"]["classification"] == "missing_from_live_export"
    assert by_name["PipeSection"]["classification"] == "observed_but_not_conservatively_represented"
    assert by_name["StraightStairsRail"]["classification"] == "missing_from_live_export"
    assert by_name["other_instantiated_obstacles"]["classification"] == "unsupported"
    assert (
        by_name["onboard_sensors"]["classification"] == "irrelevant_with_documented_justification"
    )
    assert report["classification_counts"]["matched_and_conservatively_represented"] == 1
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


def test_dynamic_envelopes_do_not_treat_static_can_as_carried_object_safety():
    report = reconcile_preserved_live_geometry()
    gripper = report["dynamic_envelopes"]["gripper"]
    can = report["dynamic_envelopes"]["can"]
    assert gripper["conservative"] is False
    assert gripper["observed_required_radius_m"] > 0.2
    assert gripper["observed_required_radius_m"] > gripper["pinned_radius_m"]
    assert can["static_geometry_matches_pinned"] is True
    assert can["attachment_state"] == "unknown"
    assert can["release_state"] == "unknown"
    assert can["carried_pose_propagation_proven"] is False
    assert report["dynamic_envelopes"]["self_collision_pairs"]["coverage_proven"] is False
    assert report["dynamic_envelopes"]["grasp_contact_exceptions"]["approved"] is False


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

    joint_dominated = evaluate_sweep_safety_bound(
        margin_m=1.0,
        point_speed_bound_m_s=1.0,
        dt_s=1.0,
        substeps=1,
        max_joint_displacement_bound_rad=2.0,
        joint_reach_bound_m=0.5,
        geometry_approximation_error_bound_m=0.0,
        registration_uncertainty_bound_m=0.0,
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
    assert sweep["reference_is_global_bound"] is False
    assert bound["certified"] is False
    assert sweep["verdict"] == "FAIL_CLOSED"
