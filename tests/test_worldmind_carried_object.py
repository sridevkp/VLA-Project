"""Inert tests for the bounded CAN lifecycle model.

No test launches Webots, actuates the robot or reads a data split. Missing evidence must always
fail closed and no contact claim is allowed without an owner-approved exception.
"""

from __future__ import annotations

import pytest

from vlarobot.world_model.carried_object import (
    CARRIED_OBJECT_STATES,
    CarriedGeometryBound,
    assess_carried_object_state,
    carried_lifecycle_report,
    validate_carried_transition,
)

_EVIDENCE = "1" * 64
_TRANSFORM = (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0)


def test_lifecycle_report_covers_every_bounded_state():
    report = carried_lifecycle_report()
    assert report["states"] == list(CARRIED_OBJECT_STATES)
    assert report["missing_attachment_policy"] == "fail_closed"
    assert report["contact_claim_policy"] == "requires_owner_reviewed_grasp_contact_exception"
    assert ["independent_pre_grasp", "attached"] in report["allowed_transitions"]


def test_missing_attachment_and_ambiguous_states_fail_closed():
    attached = assess_carried_object_state(
        "attached",
        static_geometry_verified=True,
        attachment_evidence_sha256=None,
        tool_to_object_transform=_TRANSFORM,
        grasp_contact_exception_approved=True,
    )
    assert attached.fail_closed is True
    assert attached.carried_geometry_available is False
    assert attached.contact_claim_allowed is False
    assert any("attachment evidence" in item for item in attached.blockers)

    ambiguous = assess_carried_object_state("ambiguous", static_geometry_verified=True)
    assert ambiguous.fail_closed is True
    assert ambiguous.static_obstacle_only is False
    assert ambiguous.transition_targets == ()


def test_complete_carried_evidence_allows_only_the_explicit_contact_claim():
    carried = assess_carried_object_state(
        "carried",
        static_geometry_verified=True,
        attachment_evidence_sha256=_EVIDENCE,
        tool_to_object_transform=_TRANSFORM,
        pose_propagation_verified=True,
        grasp_contact_exception_approved=True,
    )
    assert carried.fail_closed is False
    assert carried.carried_geometry_available is True
    assert carried.contact_claim_allowed is True

    no_exception = assess_carried_object_state(
        "carried",
        static_geometry_verified=True,
        attachment_evidence_sha256=_EVIDENCE,
        tool_to_object_transform=_TRANSFORM,
        pose_propagation_verified=True,
        grasp_contact_exception_approved=False,
    )
    assert no_exception.fail_closed is True
    assert no_exception.contact_claim_allowed is False

    post_release = assess_carried_object_state(
        "independent_post_release",
        static_geometry_verified=True,
        release_evidence_sha256=_EVIDENCE,
    )
    assert post_release.fail_closed is False
    assert post_release.static_obstacle_only is True
    assert post_release.contact_claim_allowed is False


def test_transition_validation_rejects_unevidenced_or_illegal_edges():
    attached = assess_carried_object_state(
        "attached",
        static_geometry_verified=True,
        attachment_evidence_sha256=_EVIDENCE,
        tool_to_object_transform=_TRANSFORM,
        grasp_contact_exception_approved=True,
    )
    carried = assess_carried_object_state(
        "carried",
        static_geometry_verified=True,
        attachment_evidence_sha256=_EVIDENCE,
        tool_to_object_transform=_TRANSFORM,
        pose_propagation_verified=True,
        grasp_contact_exception_approved=True,
    )
    pre_grasp = assess_carried_object_state("independent_pre_grasp", static_geometry_verified=True)
    assert validate_carried_transition(pre_grasp, attached) is True
    assert validate_carried_transition(attached, carried) is True
    assert validate_carried_transition(carried, pre_grasp) is False
    missing = assess_carried_object_state("ambiguous", static_geometry_verified=True)
    assert validate_carried_transition(missing, carried) is False


def test_carried_geometry_bound_is_positive_and_dimensionally_reported():
    bound = CarriedGeometryBound("CAN", 0.03175, 0.1222, "pinned CAN.proto")
    assert bound.report()["expected_dimensions_m"] == [0.0635, 0.0635, 0.1222]
    with pytest.raises(ValueError, match="positive"):
        CarriedGeometryBound("CAN", 0.0, 0.1222, "pinned CAN.proto")
