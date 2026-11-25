"""Inert tests for deterministic self-collision pair coverage.

No test launches Webots, solves IK, moves the robot or reads a data split. All geometry identities
come from the pinned URDF or from synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

from vlarobot.world_model.self_collision import (
    SelfCollisionCoverage,
    build_self_collision_coverage,
    urdf_adjacent_link_pairs,
    urdf_collision_body_names,
    validate_contract_pair_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "src/vlarobot/backends/webots/tiago.urdf"
REQUIRED_LINKS = (
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


def test_pinned_urdf_enumerates_every_collision_body_and_link_adjacency():
    bodies = urdf_collision_body_names(URDF)
    assert len(bodies) == 61
    assert len(set(bodies)) == len(bodies)
    adjacency = urdf_adjacent_link_pairs(URDF, REQUIRED_LINKS)
    assert ("arm_2_link", "arm_3_link") in adjacency
    assert ("TIAGo front arm", "wrist_ft_tool_link") not in adjacency


def test_pinned_arm_coverage_checks_every_capsule_pair_and_reports_gaps():
    coverage = build_self_collision_coverage(
        URDF,
        REQUIRED_LINKS,
        max_point_speed_m_s=28.49316700274912,
        gripper_name="gripper_front",
        carried_object_id="CAN",
    )
    counts = coverage.report()["counts"]
    assert counts["capsules"] == 19
    assert counts["all_capsule_pairs"] == 171
    assert counts["checked_capsule_pairs"] == 171
    assert counts["adjacent_capsule_pairs"] == 39
    assert counts["non_adjacent_capsule_pairs"] == 117
    assert counts["same_link_capsule_pairs"] == 15
    assert counts["gripper_arm_pairs"] == 19
    assert counts["carried_robot_pairs"] == 20
    assert counts["uncovered_collision_bodies"] == 42
    assert coverage.coverage_proven is False
    assert validate_contract_pair_coverage(coverage, frozenset()) is False


def test_adjacency_and_grasp_contacts_are_never_auto_exempt():
    coverage = SelfCollisionCoverage(
        required_links=("a", "b"),
        capsule_names=("a:0", "a:1", "b:0"),
        adjacent_link_pairs=(("a", "b"),),
        adjacent_capsule_pairs=(("a:0", "b:0"), ("a:1", "b:0")),
        non_adjacent_capsule_pairs=(),
        same_link_capsule_pairs=(("a:0", "a:1"),),
        gripper_arm_pairs=(("a:0", "gripper"),),
        carried_robot_pairs=(),
        excluded_pairs=(("a:0", "b:0"),),
        uncovered_collision_bodies=(),
    )
    assert coverage.coverage_proven is False
    assert len(coverage.all_capsule_pairs) == 3
    assert len(coverage.checked_capsule_pairs) == 2
    excluded = frozenset(frozenset(pair) for pair in coverage.excluded_pairs)
    assert validate_contract_pair_coverage(coverage, excluded) is False
