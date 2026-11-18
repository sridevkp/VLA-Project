"""Bounded carried-object lifecycle model for the pinned CAN cylinder.

The model is deliberately diagnostic: one static CAN pose does not prove grasp attachment,
release or tool-relative pose propagation. Missing attachment/release evidence and unsupported
or ambiguous states fail closed. Contact safety is never claimed unless a reviewed
grasp-contact exception is supplied explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

CarriedObjectState = Literal[
    "independent_pre_grasp",
    "attached",
    "carried",
    "released",
    "independent_post_release",
    "ambiguous",
]

CARRIED_OBJECT_STATES: tuple[CarriedObjectState, ...] = (
    "independent_pre_grasp",
    "attached",
    "carried",
    "released",
    "independent_post_release",
    "ambiguous",
)

_ALLOWED_TRANSITIONS: frozenset[tuple[CarriedObjectState, CarriedObjectState]] = frozenset(
    {
        ("independent_pre_grasp", "independent_pre_grasp"),
        ("independent_pre_grasp", "attached"),
        ("attached", "attached"),
        ("attached", "carried"),
        ("carried", "carried"),
        ("carried", "released"),
        ("released", "independent_post_release"),
        ("independent_post_release", "independent_post_release"),
    }
)


@dataclass(frozen=True)
class CarriedGeometryBound:
    """Pinned cylinder bound for the carried object."""

    object_id: str
    radius_m: float
    height_m: float
    source: str

    def __post_init__(self) -> None:
        if (
            not self.object_id
            or not self.source
            or isinstance(self.radius_m, bool)
            or isinstance(self.height_m, bool)
            or not math.isfinite(self.radius_m)
            or not math.isfinite(self.height_m)
            or self.radius_m <= 0.0
            or self.height_m <= 0.0
        ):
            raise ValueError("carried geometry bound must be finite, positive and sourced")

    def report(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "radius_m": self.radius_m,
            "height_m": self.height_m,
            "expected_dimensions_m": [2.0 * self.radius_m, 2.0 * self.radius_m, self.height_m],
            "source": self.source,
        }


@dataclass(frozen=True)
class CarriedObjectAssessment:
    """Deterministic result for one lifecycle state and its evidence."""

    state: CarriedObjectState
    static_obstacle_only: bool
    carried_geometry_available: bool
    contact_claim_allowed: bool
    transition_targets: tuple[CarriedObjectState, ...]
    blockers: tuple[str, ...]

    @property
    def fail_closed(self) -> bool:
        return bool(self.blockers)

    def report(self) -> dict[str, object]:
        return {
            "state": self.state,
            "static_obstacle_only": self.static_obstacle_only,
            "carried_geometry_available": self.carried_geometry_available,
            "contact_claim_allowed": self.contact_claim_allowed,
            "fail_closed": self.fail_closed,
            "transition_targets": list(self.transition_targets),
            "blockers": list(self.blockers),
        }


def _valid_transform(value: Sequence[float] | None) -> bool:
    if value is None or len(value) != 7:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ) and all(math.isfinite(float(item)) for item in value)


def assess_carried_object_state(
    state: CarriedObjectState,
    *,
    static_geometry_verified: bool,
    attachment_evidence_sha256: str | None = None,
    release_evidence_sha256: str | None = None,
    tool_to_object_transform: Sequence[float] | None = None,
    pose_propagation_verified: bool = False,
    grasp_contact_exception_approved: bool = False,
) -> CarriedObjectAssessment:
    """Fail closed unless the requested lifecycle state has all required evidence."""
    if state not in CARRIED_OBJECT_STATES:
        raise ValueError(f"unsupported carried-object state: {state}")
    blockers: list[str] = []
    if not static_geometry_verified:
        blockers.append("static carried-object geometry is not verified")

    targets = tuple(sorted(target for source, target in _ALLOWED_TRANSITIONS if source == state))
    if state == "ambiguous":
        blockers.append("attachment state is ambiguous")

    if state in {"attached", "carried"}:
        if not attachment_evidence_sha256:
            blockers.append("grasp attachment evidence is missing")
        if not _valid_transform(tool_to_object_transform):
            blockers.append("tool-to-object transform is missing or invalid")
        if state == "carried" and not pose_propagation_verified:
            blockers.append("carried-pose propagation is not verified")
        if not grasp_contact_exception_approved:
            blockers.append("grasp-contact exception is not owner-approved")
    elif state in {"released", "independent_post_release"}:
        if not release_evidence_sha256:
            blockers.append("release evidence is missing")
    elif state == "independent_pre_grasp" and attachment_evidence_sha256:
        blockers.append("pre-grasp state cannot carry attachment evidence")

    ready = not blockers
    return CarriedObjectAssessment(
        state=state,
        static_obstacle_only=state
        in {"independent_pre_grasp", "released", "independent_post_release"},
        carried_geometry_available=ready and state in {"attached", "carried"},
        contact_claim_allowed=ready
        and state in {"attached", "carried"}
        and grasp_contact_exception_approved,
        transition_targets=targets,
        blockers=tuple(blockers),
    )


def validate_carried_transition(
    before: CarriedObjectAssessment, after: CarriedObjectAssessment
) -> bool:
    """Return ``True`` only for a fully evidenced, legal lifecycle transition."""
    if before.fail_closed or after.fail_closed:
        return False
    if (before.state, after.state) not in _ALLOWED_TRANSITIONS:
        return False
    if after.state in {"attached", "carried"} and not after.carried_geometry_available:
        return False
    if after.state in {"attached", "carried"} and not after.contact_claim_allowed:
        return False
    return True


def carried_lifecycle_report() -> dict[str, object]:
    """Describe the lifecycle and the evidence each state requires."""
    requirements: dict[str, dict[str, object]] = {
        "independent_pre_grasp": {
            "requires": ["verified static geometry", "no attachment evidence"],
            "contact_claim": False,
        },
        "attached": {
            "requires": [
                "verified static geometry",
                "attachment evidence hash",
                "tool-to-object transform",
                "owner-approved grasp-contact exception",
            ],
            "contact_claim": True,
        },
        "carried": {
            "requires": [
                "verified static geometry",
                "attachment evidence hash",
                "tool-to-object transform",
                "verified pose propagation",
                "owner-approved grasp-contact exception",
            ],
            "contact_claim": True,
        },
        "released": {
            "requires": ["verified static geometry", "release evidence hash"],
            "contact_claim": False,
        },
        "independent_post_release": {
            "requires": ["verified static geometry", "release evidence hash"],
            "contact_claim": False,
        },
        "ambiguous": {"requires": ["fail closed"], "contact_claim": False},
    }
    payload: dict[str, object] = {
        "states": list(CARRIED_OBJECT_STATES),
        "allowed_transitions": [list(edge) for edge in sorted(_ALLOWED_TRANSITIONS)],
        "requirements_by_state": requirements,
        "missing_attachment_policy": "fail_closed",
        "ambiguous_state_policy": "fail_closed",
        "contact_claim_policy": "requires_owner_reviewed_grasp_contact_exception",
    }
    return payload


__all__ = [
    "CARRIED_OBJECT_STATES",
    "CarriedGeometryBound",
    "CarriedObjectAssessment",
    "assess_carried_object_state",
    "carried_lifecycle_report",
    "validate_carried_transition",
]
