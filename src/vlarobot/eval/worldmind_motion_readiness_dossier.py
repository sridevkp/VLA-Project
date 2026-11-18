"""Machine-readable, fail-closed motion-readiness dossier from inert evidence only.

This module never launches Webots, opens a socket, moves the robot, solves IK, scores an episode
or reads a data split. It recomputes independent reference cases, evaluates explicit gates and
reports `approved: false`; it cannot create or substitute for an owner safety certificate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from vlarobot.backends import worldmind_bindings as bindings
from vlarobot.eval.worldmind_live_geometry_reconciliation import (
    _load_verified_json,
    load_live_reconciliation_config,
    reconcile_preserved_live_geometry,
)
from vlarobot.world_model.arm_sweep import (
    ArmSweepContract,
    IntermediateSweptArmCheck,
    SweepGeometryError,
    evaluate_sweep_safety_bound,
    required_substeps,
)
from vlarobot.world_model.carried_object import assess_carried_object_state
from vlarobot.world_model.self_collision import (
    SelfCollisionCoverage,
    validate_contract_pair_coverage,
)

GateStatus = Literal["PASS", "FAIL", "UNVERIFIED"]
DOSSIER_SCHEMA_VERSION = 1

SWEEP_INEQUALITY = (
    "margin > max(point_speed*dt/substeps, joint_displacement*reach) "
    "+ geometry_approximation_error + registration_uncertainty "
    "+ numerical_error + transform_error"
)

OWNER_DECISIONS: tuple[dict[str, str], ...] = (
    {
        "id": "joint_displacement_bound",
        "decision": "Pin a finite independent maximum joint-displacement bound (rad).",
    },
    {
        "id": "global_geometry_bound",
        "decision": "Pin a finite global geometry-approximation error bound (m).",
    },
    {
        "id": "numerical_bound",
        "decision": "Pin a finite numerical-error bound (m).",
    },
    {
        "id": "transform_bound",
        "decision": "Pin a finite transform-error bound (m).",
    },
    {
        "id": "gripper_frame",
        "decision": "Review and approve the declared URDF front-frame gripper representation.",
    },
    {
        "id": "grasp_contact",
        "decision": "Review and approve any CAN grasp-contact exception; none exists today.",
    },
    {
        "id": "adjacent_pairs",
        "decision": "Review any adjacent-pair exclusion; adjacency is never auto-exempt.",
    },
    {
        "id": "runtime_table_observation",
        "decision": (
            "Obtain the exact read-only Webots expansion of the five nested Table SolidBox nodes."
        ),
    },
    {
        "id": "independent_safety_certificate",
        "decision": (
            "Record an independent safety-owner certificate; project code cannot self-approve."
        ),
    },
)

SMALLEST_REMAINING_EXPERIMENT = (
    "Read-only Webots expansion of the five nested Table SolidBox nodes with resolved PROTO "
    "hashes, evaluated translation/rotation/size/enableBoundingObject, resulting boundingObject "
    "Box.size, and world/base poses; it closes the table-transform gap without motion or IK."
)


@dataclass(frozen=True)
class Gate:
    """One machine-readable safety gate."""

    id: str
    status: GateStatus
    summary: str
    details: Mapping[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "summary": self.summary,
            "details": dict(self.details),
        }


def _mapping_from(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not a mapping")
    return cast(Mapping[str, Any], value)


def _sequence_from(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} is not a sequence")
    return cast(Sequence[Any], value)


def _close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if expected is None or actual is None:
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _close(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
            and len(actual) == len(expected)
            and all(_close(left, right) for left, right in zip(actual, expected, strict=True))
        )
    return actual == expected


def _case_result(
    case_id: str, kind: str, *, passed: bool, actual: Any, expected: Any
) -> dict[str, Any]:
    return {
        "id": case_id,
        "kind": kind,
        "pass": passed,
        "actual": actual,
        "expected": expected,
    }


def _swept_bound_case(
    case_id: str, payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    bound = evaluate_sweep_safety_bound(**dict(payload))
    actual = {
        "certified": bound.certified,
        "required_margin_m": bound.required_margin_m,
        "residual_m": bound.residual_m,
        "point_motion_allowance_m": bound.point_motion_allowance_m,
        "joint_motion_allowance_m": bound.joint_motion_allowance_m,
    }
    return _case_result(
        case_id,
        "swept_bound",
        passed=_close(actual, expected),
        actual=actual,
        expected=dict(expected),
    )


def _self_collision_case(
    case_id: str, payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    def strings(key: str) -> tuple[str, ...]:
        return tuple(str(item) for item in _sequence_from(payload.get(key, ()), key))

    def pairs(key: str) -> tuple[tuple[str, str], ...]:
        rows = _sequence_from(payload.get(key, ()), key)
        return tuple((str(row[0]), str(row[1])) for row in rows)

    coverage = SelfCollisionCoverage(
        required_links=strings("required_links"),
        capsule_names=strings("capsule_names"),
        adjacent_link_pairs=pairs("adjacent_link_pairs"),
        adjacent_capsule_pairs=pairs("adjacent_capsule_pairs"),
        non_adjacent_capsule_pairs=pairs("non_adjacent_capsule_pairs"),
        same_link_capsule_pairs=pairs("same_link_capsule_pairs"),
        gripper_arm_pairs=pairs("gripper_arm_pairs"),
        carried_robot_pairs=pairs("carried_robot_pairs"),
        excluded_pairs=pairs("excluded_pairs"),
        uncovered_collision_bodies=strings("uncovered_collision_bodies"),
    )
    excluded = frozenset(frozenset(pair) for pair in coverage.excluded_pairs)
    actual = {
        "coverage_proven": coverage.coverage_proven,
        "contract_coverage": validate_contract_pair_coverage(coverage, excluded),
        "all_capsule_pairs": len(coverage.all_capsule_pairs),
        "checked_capsule_pairs": len(coverage.checked_capsule_pairs),
        "gripper_arm_pairs": len(coverage.gripper_arm_pairs),
    }
    return _case_result(
        case_id,
        "self_collision",
        passed=_close(actual, expected),
        actual=actual,
        expected=dict(expected),
    )


def _carried_case(
    case_id: str, payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    assessment = assess_carried_object_state(**dict(payload))
    actual = {
        "fail_closed": assessment.fail_closed,
        "carried_geometry_available": assessment.carried_geometry_available,
        "contact_claim_allowed": assessment.contact_claim_allowed,
    }
    return _case_result(
        case_id,
        "carried_lifecycle",
        passed=_close(actual, expected),
        actual=actual,
        expected=dict(expected),
    )


def _gripper_case(
    case_id: str,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    dynamic = _mapping_from(report.get("dynamic_envelopes"), "dynamic_envelopes")
    gripper = _mapping_from(dynamic.get("gripper"), "gripper")
    pinned_radius = float(payload["pinned_radius_m"])
    frame = str(payload["frame"])
    frame_required = float(gripper["frame_correct_required_radius_m"])
    static_conservative = (
        gripper.get("conservative") is True
        and gripper.get("frame") == frame
        and frame_required <= pinned_radius + 1e-9
        and math.isclose(
            frame_required,
            float(payload["frame_correct_required_radius_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    radius_unchanged = math.isclose(
        float(gripper["pinned_radius_m"]), pinned_radius, rel_tol=0.0, abs_tol=1e-12
    )
    wrong_frame_required = float(gripper["wrong_frame_required_radius_m"])
    runtime_literal_required = float(gripper["runtime_literal_required_radius_m"])
    mismatch_is_frame_artifact = (
        math.isclose(
            wrong_frame_required,
            float(payload["wrong_frame_radius_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            runtime_literal_required,
            float(payload["runtime_literal_required_radius_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and wrong_frame_required > pinned_radius
        and runtime_literal_required > pinned_radius
        and static_conservative
    )
    actual = {
        "static_conservative": static_conservative,
        "radius_unchanged": radius_unchanged,
        "mismatch_is_frame_artifact": mismatch_is_frame_artifact,
    }
    return _case_result(
        case_id,
        "gripper_report",
        passed=_close(actual, expected),
        actual=actual,
        expected=dict(expected),
    )


def _table_case(
    case_id: str,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    components = _sequence_from(report.get("components"), "components")
    table = next(
        item for item in components if _mapping_from(item, "component").get("name") == "table"
    )
    live = _mapping_from(_mapping_from(table, "table component").get("live"), "table.live")
    source_expansion = _mapping_from(live.get("source_expansion"), "table.live.source_expansion")
    grounded = live.get("runtime_collision_tree_grounded") is True
    conservative_cross_check = (
        live.get("status") == payload["status"]
        and len(_sequence_from(source_expansion.get("boxes", ()), "source boxes"))
        == payload["box_count"]
        and grounded == payload["runtime_collision_tree_grounded"]
        and source_expansion.get("coarse_catalog_proxy_contains_all_boxes") is True
    )
    actual = {
        "conservative_source_cross_check": conservative_cross_check,
        "runtime_certified": grounded,
    }
    return _case_result(
        case_id,
        "table_report",
        passed=_close(actual, expected),
        actual=actual,
        expected=dict(expected),
    )


def _checker_case(case_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    contract = ArmSweepContract(
        required_links=tuple(
            str(item)
            for item in _sequence_from(payload.get("required_links", ()), "required_links")
        ),
        required_objects=frozenset(
            str(item)
            for item in _sequence_from(payload.get("required_objects", ()), "required_objects")
        ),
        adjacent_pairs=frozenset(),
        dt_s=float(payload["dt_s"]),
        margin_m=float(payload["margin_m"]),
        substeps=int(payload["substeps"]),
    )

    def geometry(state: Any) -> tuple[Any, ...]:
        del state
        raise SweepGeometryError("missing transform")

    result = IntermediateSweptArmCheck(cast(Any, geometry), contract)(
        cast(Any, None), cast(Any, None)
    )
    return _case_result(
        case_id,
        "checker_gate",
        passed=result is None,
        actual={"result": result},
        expected={"result": None},
    )


def evaluate_reference_cases(
    cases: Mapping[str, Any], report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Recompute every independent reference case; an unknown kind is a failed case."""
    if (
        cases.get("schema_version") != 1
        or cases.get("status") != "independent_inert_reference_cases"
    ):
        raise ValueError("unsupported independent reference-case fixture")
    entries = cases.get("cases")
    if not isinstance(entries, Sequence) or not entries:
        raise ValueError("independent reference-case fixture has no cases")
    results: list[dict[str, Any]] = []
    for item in entries:
        case = _mapping_from(item, "reference case")
        case_id = str(case["id"])
        kind = str(case["kind"])
        payload = _mapping_from(case["input"], f"{case_id}.input")
        expected = _mapping_from(case["expected"], f"{case_id}.expected")
        if kind == "swept_bound":
            result = _swept_bound_case(case_id, payload, expected)
        elif kind == "required_substeps":
            actual = required_substeps(**dict(payload))
            result = _case_result(
                case_id,
                kind,
                passed=_close(actual, expected["substeps"]),
                actual={"substeps": actual},
                expected=dict(expected),
            )
        elif kind == "self_collision":
            result = _self_collision_case(case_id, payload, expected)
        elif kind == "carried_lifecycle":
            result = _carried_case(case_id, payload, expected)
        elif kind == "gripper_report":
            result = _gripper_case(case_id, payload, expected, report)
        elif kind == "table_report":
            result = _table_case(case_id, payload, expected, report)
        elif kind == "checker_gate":
            result = _checker_case(case_id, payload)
        else:
            result = _case_result(
                case_id,
                kind,
                passed=False,
                actual={"error": "unsupported reference-case kind"},
                expected=dict(expected),
            )
        results.append(result)
    return results


def _status_for_residual(bound: Mapping[str, Any]) -> GateStatus:
    if bound.get("certified") is True and isinstance(bound.get("residual_m"), (int, float)):
        return "PASS" if float(bound["residual_m"]) > 0.0 else "FAIL"
    return "UNVERIFIED" if bound.get("residual_m") is None else "FAIL"


def _production_binding_gate(cfg: Any) -> Gate:
    contract_path = Path(str(cfg.pinned["arm_sweep_contract"]))
    contract = bindings.load_arm_sweep_config(contract_path)
    callback = bindings.arm_sweep_transition_check(contract_path)
    passed = (
        contract.status == "blocked"
        and contract.certificate is None
        and callback is None
        and not contract.adjacent_pairs
        and not contract.permitted_contact_objects
    )
    return Gate(
        id="production_binding",
        status="PASS" if passed else "FAIL",
        summary=(
            "Production sweep callback stays None with no auto-exempt contact pairs."
            if passed
            else "Production binding no longer fails closed."
        ),
        details={
            "status": contract.status,
            "certificate": contract.certificate is not None,
            "check_transition_is_none": callback is None,
            "adjacent_pairs": sorted(sorted(pair) for pair in contract.adjacent_pairs),
            "permitted_contact_objects": sorted(contract.permitted_contact_objects),
        },
    )


def build_motion_readiness_dossier(config_path: Path | None = None) -> dict[str, Any]:
    """Build the deterministic, non-approving dossier from pinned inert evidence."""
    cfg = load_live_reconciliation_config(config_path)
    report = cast(dict[str, Any], reconcile_preserved_live_geometry(config_path))
    cases = cast(Mapping[str, Any], _load_verified_json(cfg.evidence["reference_safety_cases"]))
    case_results = evaluate_reference_cases(cases, report)

    safety = _mapping_from(report.get("swept_collision_validation"), "swept_collision_validation")
    bound = _mapping_from(safety.get("safety_bound"), "safety_bound")
    dynamic = _mapping_from(report.get("dynamic_envelopes"), "dynamic_envelopes")
    gripper = _mapping_from(dynamic.get("gripper"), "gripper")
    lifecycle = _mapping_from(dynamic.get("carried_lifecycle"), "carried_lifecycle")
    current_lifecycle = _mapping_from(lifecycle.get("current_assessment"), "current_assessment")
    self_collision = _mapping_from(dynamic.get("self_collision_pairs"), "self_collision_pairs")
    accepted = _mapping_from(report.get("accepted_inspection"), "accepted_inspection")
    components = _sequence_from(report.get("components"), "components")
    table = next(
        item for item in components if _mapping_from(item, "component").get("name") == "table"
    )
    table_live = _mapping_from(_mapping_from(table, "table component").get("live"), "table.live")
    table_source = _mapping_from(table_live.get("source_expansion"), "table.live.source_expansion")

    gates: tuple[Gate, ...] = (
        Gate(
            id="frozen_evidence",
            status="PASS" if report.get("evidence_verified") is True else "FAIL",
            summary=(
                "All pinned artifacts were hash-verified before use."
                if report.get("evidence_verified") is True
                else "Evidence verification failed."
            ),
            details={
                "evidence_hashes": dict(
                    _mapping_from(report.get("evidence_hashes"), "evidence_hashes")
                ),
                "webots_rerun_performed": accepted.get("webots_rerun_performed"),
            },
        ),
        Gate(
            id="swept_residual",
            status=_status_for_residual(bound),
            summary=(
                "Swept inequality has a certificate only for strictly positive residual."
                if bound.get("certified") is True
                else "Required swept-residual bounds are unavailable; the transition fails closed."
            ),
            details={
                "inequality": SWEEP_INEQUALITY,
                "residual_m": bound.get("residual_m"),
                "required_margin_m": bound.get("required_margin_m"),
                "point_motion_allowance_m": bound.get("point_motion_allowance_m"),
                "joint_motion_allowance_m": bound.get("joint_motion_allowance_m"),
                "geometry_approximation_error_bound_m": bound.get(
                    "geometry_approximation_error_bound_m"
                ),
                "registration_uncertainty_bound_m": bound.get("registration_uncertainty_bound_m"),
                "numerical_error_bound_m": bound.get("numerical_error_bound_m"),
                "transform_error_bound_m": bound.get("transform_error_bound_m"),
                "blockers": list(_sequence_from(bound.get("blockers", ()), "blockers")),
            },
        ),
        Gate(
            id="gripper_envelope",
            status=(
                "PASS"
                if gripper.get("conservative") is True
                and gripper.get("frame") == "front"
                and math.isclose(float(gripper["pinned_radius_m"]), 0.1, rel_tol=0.0, abs_tol=1e-12)
                else "UNVERIFIED"
            ),
            summary=(
                "Frame-corrected static gripper evidence fits the unchanged pinned radius."
                if gripper.get("conservative") is True
                else "Gripper envelope is not conservative in its declared frame."
            ),
            details=dict(gripper),
        ),
        Gate(
            id="can_lifecycle",
            status="UNVERIFIED" if current_lifecycle.get("fail_closed") is True else "PASS",
            summary=(
                "CAN attachment, release and carried pose are unobserved."
                if current_lifecycle.get("fail_closed") is True
                else "CAN lifecycle evidence is complete."
            ),
            details=dict(current_lifecycle),
        ),
        Gate(
            id="self_collision",
            status=(
                "FAIL"
                if self_collision.get("adjacency_auto_exempt") is True
                or self_collision.get("grasp_contact_auto_exempt") is True
                or self_collision.get("contract_excluded_pairs")
                else "UNVERIFIED"
                if self_collision.get("coverage_proven") is not True
                or self_collision.get("contract_coverage_validated") is not True
                else "PASS"
            ),
            summary=(
                "Every pinned capsule pair is checked; full collision-body coverage is unproven."
                if self_collision.get("coverage_proven") is not True
                else "Self-collision pair coverage is complete."
            ),
            details={
                "counts": dict(
                    _mapping_from(self_collision.get("counts"), "self_collision.counts")
                ),
                "coverage_proven": self_collision.get("coverage_proven"),
                "contract_coverage_validated": self_collision.get("contract_coverage_validated"),
                "uncovered_collision_bodies": list(
                    _sequence_from(
                        self_collision.get("uncovered_collision_bodies", ()),
                        "uncovered_collision_bodies",
                    )
                ),
                "excluded_pairs": list(
                    _sequence_from(self_collision.get("excluded_pairs", ()), "excluded_pairs")
                ),
            },
        ),
        Gate(
            id="table_transform",
            status=(
                "UNVERIFIED"
                if table_live.get("status") == "source_derived_unverified"
                and table_live.get("runtime_collision_tree_grounded") is not True
                else "PASS"
                if table_live.get("runtime_collision_tree_grounded") is True
                else "FAIL"
            ),
            summary=(
                "Table expansion is source-derived and has no runtime collision transform."
                if table_live.get("status") == "source_derived_unverified"
                else "Table collision-transform gate changed."
            ),
            details={
                "status": table_live.get("status"),
                "box_count": len(
                    _sequence_from(table_source.get("boxes", ()), "table source boxes")
                ),
                "coarse_proxy_contains_all_boxes": table_source.get(
                    "coarse_catalog_proxy_contains_all_boxes"
                ),
                "missing_runtime_observation": table_live.get("missing_runtime_observation"),
            },
        ),
        Gate(
            id="reference_cases",
            status="PASS" if all(item["pass"] is True for item in case_results) else "FAIL",
            summary=(
                "All independent inert reference cases reproduce their expected outcomes."
                if all(item["pass"] is True for item in case_results)
                else "One or more independent reference cases failed."
            ),
            details={"case_count": len(case_results), "cases": case_results},
        ),
        _production_binding_gate(cfg),
        Gate(
            id="motion_authorization",
            status="FAIL",
            summary=(
                "No independent safety-owner certificate exists; project code cannot approve "
                "motion."
            ),
            details={"approved": False, "owner_approvals_required": list(OWNER_DECISIONS)},
        ),
    )
    gate_rows = [gate.report() for gate in gates]
    statuses = [gate.status for gate in gates]
    overall = "FAIL" if "FAIL" in statuses else "UNVERIFIED" if "UNVERIFIED" in statuses else "PASS"
    return {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "artifact": "worldmind_motion_readiness_dossier_v1",
        "status": "diagnostic_not_certified",
        "approved": False,
        "motion_allowed": False,
        "motion_readiness": "NOT_READY" if overall != "PASS" else "READY",
        "overall_gate": overall,
        "gates": gate_rows,
        "gate_counts": {
            status: statuses.count(status) for status in ("PASS", "FAIL", "UNVERIFIED")
        },
        "swept_inequality": SWEEP_INEQUALITY,
        "reference_case_count": len(case_results),
        "evidence_hashes": dict(_mapping_from(report.get("evidence_hashes"), "evidence_hashes")),
        "unsupported_cases": {
            "table": table_live.get("missing_runtime_observation"),
            "forwarded_geometry": ["PipeSection", "StraightStairsRail"],
            "carried_object": "attachment, release and pose propagation are unobserved",
            "self_collision": "full URDF collision-body coverage is unproven",
        },
        "owner_decisions_required": list(OWNER_DECISIONS),
        "smallest_remaining_experiment": SMALLEST_REMAINING_EXPERIMENT,
        "blockers": list(_sequence_from(report.get("blockers", ()), "blockers")),
    }


__all__ = [
    "DOSSIER_SCHEMA_VERSION",
    "SWEEP_INEQUALITY",
    "Gate",
    "build_motion_readiness_dossier",
    "evaluate_reference_cases",
]
