"""Proposed Eq. 3.8 component definitions — ENGINEERING CHOICES, NOT REPORT QUOTATIONS.

The submitted Phase I report gives only the verbal meanings of Jg, Jl, Jr and Ju and the form
``J = wg*Jg + wl*Jl + wr*Jr + wu*Ju``. It does NOT supply formulas, units, normalisation
constants, weighting values, ranking direction or a calibrated risk threshold. Everything in this
module is an explicit proposal for owner review.

Nothing here is registered as a WorldMind component binding, evaluator, weight or threshold. The
production registry stays empty for these identities; a `ready` config referencing them fails
closed. These functions are inert: they read only the numeric inputs passed to them and never
access a backend, campaign, TEST/TRAIN split or wall clock.

Decision-time amendment (2026-11-19, owner approval required): ``Jr`` is a *soft* tool-envelope
segment/AABB clearance proxy computed from current observed object geometry and the predicted tool
path; it is not certified full-arm clearance and never replaces the hard safety veto. ``Ju`` is a
deterministic command-following residual computed from the current observation, candidate chunk and
predicted trajectory; no future observed state, correction residual, fabricated uncertainty or
ensemble is used. Both changes amend the scientific contract and are not activated here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from vlarobot.core.types import (
    ActionChunk,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    JointTarget,
    Pose3D,
    Vec3,
    WorldState,
)
from vlarobot.world_model.analytic import segment_box_clearance
from vlarobot.world_model.rollout import Trajectory, quaternion_product

COMPONENT_NAMES = ("Jg", "Jl", "Jr", "Ju")
PROPOSAL_STATUS = "engineering_choice_not_report_quotation"
AMENDMENT_STATUS = "engineering_choice_owner_approval_required_decision_time_amendment"
SAFETY_PRECEDENCE = (
    "Safety is a hard veto (SafetyMonitor.validate + RiskGate + unverified-arm-sweep). "
    "A candidate rejected by a hard check is never ranked; no utility term may offset a veto."
)
MISSING_DATA_BEHAVIOUR = (
    "Fail closed with MissingObjectiveInput. Never impute zero error, a synthetic goal, an "
    "unmeasured clearance, a wall-clock duration or an invented uncertainty."
)
CAN_CONTACT_POLICY = (
    "Intentional CAN contact during grasping receives no exemption in Jr_soft. The CAN remains an "
    "observed object box; tool-envelope overlap or negative clearance contributes the maximum soft "
    "deficit. This is a ranking cost only: hard contact/grasp policy and safety vetoes are "
    "unchanged."
)
JU_ALIGNMENT_POLICY = (
    "Ju requires exactly len(actions)+1 predicted states and matching current/predicted object "
    "identities. JointTarget residuals use sorted joint names for command and prediction. Gripper "
    "fields are accounted for EEDelta, EEPoseTarget and GripperCmd. Empty or unsupported "
    "comparisons fail closed with MissingObjectiveInput. Ju measures predicted command-following "
    "consistency, "
    "not calibrated uncertainty."
)
WEIGHT_SELECTION = (
    "Weights (wg, wl, wr, wu) and ranking direction are selected by the owner on the frozen "
    "VALIDATION evidence through a registered evaluator and recorded in an approved calibration "
    "candidate. The selection statistic is not an independent evaluation of post-calibration "
    "performance."
)


class MissingObjectiveInput(ValueError):
    """A required physical input is absent; no objective value may be invented."""


class HardVetoPrecedesUtility(ValueError):
    """A hard safety veto was supplied; utility aggregation is not permitted."""


@dataclass(frozen=True)
class ObjectiveProposal:
    """One explicitly non-authoritative definition awaiting owner decision."""

    name: str
    formula: str
    units: str
    normalization: str
    ranking_direction: str
    required_inputs: tuple[str, ...]
    missing_data: str
    safety_precedence: str
    status: str

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "units": self.units,
            "normalization": self.normalization,
            "ranking_direction": self.ranking_direction,
            "required_inputs": list(self.required_inputs),
            "missing_data": self.missing_data,
            "safety_precedence": self.safety_precedence,
            "status": self.status,
        }


PROPOSED_OBJECTIVES: Mapping[str, ObjectiveProposal] = {
    "Jg": ObjectiveProposal(
        "Jg",
        "sqrt((||p_ee - p_goal||_2 / s_p)^2 + (angle(q_ee, q_goal) / s_r)^2)",
        "dimensionless (m and rad divided by declared scales)",
        "s_p in m > 0; s_r in rad > 0; both owner-selected on VALIDATION",
        "minimize",
        ("ee pose in robot base frame", "goal pose in robot base frame"),
        MISSING_DATA_BEHAVIOUR,
        SAFETY_PRECEDENCE,
        PROPOSAL_STATUS,
    ),
    "Jl": ObjectiveProposal(
        "Jl",
        "(sum_k ||p_{k+1} - p_k||_2 / s_l) + lambda_t * (N_steps * dt_s / s_t)",
        "dimensionless (m and s divided by declared scales)",
        "s_l in m > 0; s_t in s > 0; lambda_t dimensionless; owner-selected",
        "minimize",
        ("tool positions in robot base frame", "control tick dt_s in s"),
        MISSING_DATA_BEHAVIOUR + " Wall-clock time is never substituted for control time.",
        SAFETY_PRECEDENCE,
        PROPOSAL_STATUS,
    ),
    "Jr": ObjectiveProposal(
        "Jr",
        "Jr_soft = mean_{k=1..K} clip((m - tilde_d_k) / m, 0, 1); tilde_d_k < 0 maps to 1; "
        "tilde_d_k = min_o segmentBoxClearance(p_{k-1}, p_k, AABB_o(current), r_tool); "
        "UNCERTIFIED proxy",
        "dimensionless ratio in [0, 1]",
        "m in m > 0 and r_tool in m > 0 are owner-selected; tilde_d_k is an UNCERTIFIED "
        "tool-envelope segment/AABB clearance proxy (m), not certified full-arm clearance",
        "minimize",
        (
            "current observed object poses/dimensions with scene_complete=true",
            "predicted tool positions in robot base frame",
            "tool-envelope radius r_tool in m",
            "declared clearance margin m in m",
        ),
        MISSING_DATA_BEHAVIOUR
        + " Incomplete scene, absent object geometry, truncated prediction or missing r_tool/m "
        "raises MissingObjectiveInput. This soft score is not a safety certificate. "
        + CAN_CONTACT_POLICY,
        SAFETY_PRECEDENCE
        + " Hard vetoes (SafetyMonitor.validate, RiskGate, unverified-arm-sweep) are evaluated "
        "separately and still veto independently of Jr.",
        AMENDMENT_STATUS,
    ),
    "Ju": ObjectiveProposal(
        "Ju",
        "Ju_consistency = mean_{i in C} r_i; r_i = ||p_hat_{i+1} - p_cmd_i||/s_p + "
        "angle(q_hat_{i+1}, q_cmd_i)/s_r for EE commands; ||q_hat_{i+1} - q_cmd_i||/s_q for "
        "JointTarget; |g_hat_{i+1} - g_cmd_i|/s_g for gripper components",
        "dimensionless deterministic command-following residual (NOT calibrated uncertainty)",
        "s_p in m > 0; s_r in rad > 0; s_q in rad > 0; s_g in m > 0; owner-selected on VALIDATION",
        "minimize",
        (
            "current observed WorldState (identity alignment only)",
            "ActionChunk command fields",
            "predicted Trajectory states",
            "scales s_p, s_r, s_q and s_g",
        ),
        MISSING_DATA_BEHAVIOUR
        + " No future observed state, ObjectWorldModel.correct residual, fabricated uncertainty "
        "or ensemble is accepted. " + JU_ALIGNMENT_POLICY,
        SAFETY_PRECEDENCE,
        AMENDMENT_STATUS,
    ),
}


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissingObjectiveInput(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MissingObjectiveInput(f"{name} must be a finite positive number")
    return number


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise MissingObjectiveInput(f"{name} must be a nonempty sequence")
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise MissingObjectiveInput(f"{name} must be finite and one-dimensional")
    return array


def _xyz(position: Vec3, name: str) -> np.ndarray:
    return _vector(tuple(position.as_array()), name)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MissingObjectiveInput(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise MissingObjectiveInput(f"{name} must be a finite real number")
    return number


def _unit_quaternion(values: Sequence[float], name: str) -> np.ndarray:
    quaternion = _vector(tuple(values), name)
    if quaternion.shape != (4,) or not math.isclose(
        float(quaternion @ quaternion), 1.0, abs_tol=1e-6
    ):
        raise MissingObjectiveInput(f"{name} must be a unit xyzw quaternion")
    return quaternion


def _orientation_error(a: Sequence[float], b: Sequence[float], name: str) -> float:
    q_a = _unit_quaternion(a, f"{name}.a")
    q_b = _unit_quaternion(b, f"{name}.b")
    dot = min(1.0, max(-1.0, abs(float(q_a @ q_b))))
    return 2.0 * math.acos(dot)


def _object_aabbs(current: WorldState) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(current, WorldState) or not current.scene.objects:
        raise MissingObjectiveInput("Jr requires a current scene with at least one observed object")
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for index, obj in enumerate(current.scene.objects):
        center = _xyz(obj.pose.position, f"current.scene.objects[{index}].pose.position")
        dimensions = _xyz(obj.dimensions, f"current.scene.objects[{index}].dimensions")
        quaternion = _vector(
            tuple(obj.pose.orientation), f"current.scene.objects[{index}].pose.orientation"
        )
        if quaternion.shape != (4,) or not math.isclose(
            float(quaternion @ quaternion), 1.0, abs_tol=1e-3
        ):
            raise MissingObjectiveInput(f"object {index} has an invalid orientation")
        if np.any(dimensions <= 0.0):
            raise MissingObjectiveInput(f"object {index} dimensions must be positive")
        x, y, z, w = quaternion
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        half = np.abs(rotation) @ (dimensions / 2.0)
        boxes.append((center - half, center + half))
    return boxes


def tool_envelope_clearances(
    current: WorldState,
    chunk: ActionChunk,
    trajectory: Trajectory,
    *,
    tool_radius_m: float,
    scene_complete: bool,
) -> tuple[float, ...]:
    """Decision-time, UNCERTIFIED tool-envelope clearance per predicted segment.

    Uses only the current observed object geometry and the predicted tool path. It is an exact
    segment/AABB distance minus a declared tool radius; full-arm, self-collision and contact
    certification are explicitly outside this proxy. The CAN is never exempted: intentional CAN
    contact contributes a negative clearance and therefore the maximum soft deficit.
    """
    radius = _positive(tool_radius_m, "tool_radius_m")
    if not isinstance(current, WorldState):
        raise MissingObjectiveInput("Jr requires a current observed WorldState")
    if not isinstance(chunk, ActionChunk) or not chunk.actions:
        raise MissingObjectiveInput("Jr requires a nonempty candidate ActionChunk")
    if not isinstance(trajectory, Trajectory):
        raise MissingObjectiveInput("Jr requires a predicted Trajectory")
    if len(trajectory.states) != len(chunk.actions) + 1:
        raise MissingObjectiveInput("Jr requires one predicted state per candidate action")
    if not isinstance(scene_complete, bool) or not scene_complete:
        raise MissingObjectiveInput(
            "Jr requires scene_complete=true; incomplete scenes fail closed"
        )
    boxes = _object_aabbs(current)
    points = [
        _xyz(
            state.robot.ee_pose.position,
            f"trajectory.states[{index}].robot.ee_pose.position",
        )
        for index, state in enumerate(trajectory.states)
    ]
    clearances: list[float] = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
        try:
            value = min(
                segment_box_clearance(start, end, lower, upper, radius) for lower, upper in boxes
            )
        except (TypeError, ValueError) as exc:
            raise MissingObjectiveInput(f"Jr segment {index} has invalid geometry") from exc
        number = float(value)
        if not math.isfinite(number):
            raise MissingObjectiveInput(f"Jr segment {index} has a nonfinite clearance")
        clearances.append(number)
    return tuple(clearances)


def soft_geometric_risk(
    current: WorldState,
    chunk: ActionChunk,
    trajectory: Trajectory,
    *,
    tool_radius_m: float,
    margin_m: float,
    scene_complete: bool,
) -> float:
    """Proposed decision-time Jr: bounded deficit of an uncertified tool-clearance proxy.

    The hard safety veto is not represented here; it remains a separate gate that excludes the
    candidate from ranking before this score is ever used.
    """
    clearances = tool_envelope_clearances(
        current,
        chunk,
        trajectory,
        tool_radius_m=tool_radius_m,
        scene_complete=scene_complete,
    )
    return predicted_risk(clearances, margin_m=margin_m)


def goal_error(
    ee_pose: Pose3D,
    goal_pose: Pose3D,
    *,
    position_scale_m: float,
    orientation_scale_rad: float,
) -> float:
    """Proposed Jg: normalised base-frame position and orientation error. Minimize."""
    s_p = _positive(position_scale_m, "position_scale_m")
    s_r = _positive(orientation_scale_rad, "orientation_scale_rad")
    if not isinstance(ee_pose, Pose3D) or not isinstance(goal_pose, Pose3D):
        raise MissingObjectiveInput("Jg requires explicit ee and goal Pose3D values")
    delta = _xyz(ee_pose.position, "ee_pose.position") - _xyz(
        goal_pose.position, "goal_pose.position"
    )
    q_a = _vector(tuple(ee_pose.orientation), "ee_pose.orientation")
    q_b = _vector(tuple(goal_pose.orientation), "goal_pose.orientation")
    if q_a.shape != (4,) or q_b.shape != (4,):
        raise MissingObjectiveInput("Jg requires xyzw quaternions")
    if not math.isclose(float(q_a @ q_a), 1.0, abs_tol=1e-8) or not math.isclose(
        float(q_b @ q_b), 1.0, abs_tol=1e-8
    ):
        raise MissingObjectiveInput("Jg requires unit xyzw quaternions")
    dot = min(1.0, max(-1.0, float(q_a @ q_b)))
    angle = 2.0 * math.acos(dot)
    value = math.hypot(float(np.linalg.norm(delta)) / s_p, angle / s_r)
    if not math.isfinite(value):
        raise MissingObjectiveInput("Jg produced a non-finite value")
    return value


def trajectory_cost(
    positions: Sequence[Vec3],
    *,
    dt_s: float,
    path_scale_m: float,
    time_scale_s: float,
    time_weight: float = 0.0,
) -> float:
    """Proposed Jl: normalised tool path length plus optional control-time cost. Minimize."""
    s_l = _positive(path_scale_m, "path_scale_m")
    s_t = _positive(time_scale_s, "time_scale_s")
    tick = _positive(dt_s, "dt_s")
    if isinstance(time_weight, bool) or not isinstance(time_weight, (int, float)):
        raise MissingObjectiveInput("time_weight must be a finite nonnegative number")
    lam = float(time_weight)
    if not math.isfinite(lam) or lam < 0.0:
        raise MissingObjectiveInput("time_weight must be a finite nonnegative number")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        raise MissingObjectiveInput("Jl requires a sequence of tool positions")
    if len(positions) < 2:
        raise MissingObjectiveInput("Jl requires at least two tool positions")
    points = np.stack([_xyz(item, f"positions[{index}]") for index, item in enumerate(positions)])
    path_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    duration_s = (len(positions) - 1) * tick
    value = path_m / s_l + lam * duration_s / s_t
    if not math.isfinite(value):
        raise MissingObjectiveInput("Jl produced a non-finite value")
    return value


def predicted_risk(clearances_m: Sequence[float], *, margin_m: float) -> float:
    """Bounded mean deficit of supplied clearances. For Jr these are UNCERTIFIED proxies."""
    margin = _positive(margin_m, "margin_m")
    values = _vector(tuple(clearances_m), "clearances_m")
    deficit = np.clip((margin - values) / margin, 0.0, 1.0)
    value = float(np.mean(deficit))
    if not math.isfinite(value):
        raise MissingObjectiveInput("Jr produced a non-finite value")
    return value


def prediction_consistency(
    current: WorldState,
    chunk: ActionChunk,
    trajectory: Trajectory,
    *,
    position_scale_m: float,
    orientation_scale_rad: float,
    joint_scale_rad: float,
    gripper_scale_m: float,
) -> float:
    """Proposed decision-time Ju: deterministic command-following residual. Minimize.

    The only observation used is the current state (identity alignment). Later observed states,
    corrections, uncertainty estimates and ensembles are not inputs and cannot leak into a score.
    Action/state alignment, sorted joint names, gripper accounting and fail-closed behaviour follow
    JU_ALIGNMENT_POLICY. This is command-following consistency, not calibrated uncertainty.
    """
    s_p = _positive(position_scale_m, "position_scale_m")
    s_r = _positive(orientation_scale_rad, "orientation_scale_rad")
    s_q = _positive(joint_scale_rad, "joint_scale_rad")
    s_g = _positive(gripper_scale_m, "gripper_scale_m")
    if not isinstance(current, WorldState):
        raise MissingObjectiveInput("Ju requires a current observed WorldState")
    if not isinstance(chunk, ActionChunk) or not chunk.actions:
        raise MissingObjectiveInput("Ju requires a nonempty candidate ActionChunk")
    if not isinstance(trajectory, Trajectory):
        raise MissingObjectiveInput("Ju requires a predicted Trajectory")
    if len(trajectory.states) != len(chunk.actions) + 1:
        raise MissingObjectiveInput("Ju requires one predicted state per candidate action")
    if not all(isinstance(state, WorldState) for state in trajectory.states):
        raise MissingObjectiveInput("Ju requires predicted WorldState entries")
    current_ids = sorted(obj.object_id for obj in current.scene.objects)
    predicted_ids = sorted(obj.object_id for obj in trajectory.states[0].scene.objects)
    if current_ids != predicted_ids:
        raise MissingObjectiveInput("Ju trajectory is not aligned with the current observation")

    terms: list[float] = []
    for index, action in enumerate(chunk.actions):
        before = trajectory.states[index].robot
        after = trajectory.states[index + 1].robot
        if isinstance(action, EEDelta):
            before_position = _xyz(before.ee_pose.position, f"Ju step {index} start position")
            delta = _vector(tuple(action.dpos.as_array()), f"Ju step {index} dpos")
            actual_position = _xyz(after.ee_pose.position, f"Ju step {index} predicted position")
            terms.append(float(np.linalg.norm(before_position + delta - actual_position)) / s_p)
            rotation = _vector(tuple(action.drot.as_array()), f"Ju step {index} drot")
            angle = float(np.linalg.norm(rotation))
            factor = math.sin(angle / 2.0) / angle if angle else 0.5
            delta_q = np.r_[rotation * factor, math.cos(angle / 2.0)]
            expected_q = quaternion_product(
                delta_q,
                _unit_quaternion(before.ee_pose.orientation, f"Ju step {index} start orientation"),
            )
            terms.append(
                _orientation_error(
                    expected_q, after.ee_pose.orientation, f"Ju step {index} orientation"
                )
                / s_r
            )
            terms.append(
                abs(
                    _finite(action.gripper, f"Ju step {index} commanded gripper")
                    - _finite(after.gripper_opening, f"Ju step {index} predicted gripper")
                )
                / s_g
            )
        elif isinstance(action, EEPoseTarget):
            target_position = _xyz(action.pose.position, f"Ju step {index} target position")
            actual_position = _xyz(after.ee_pose.position, f"Ju step {index} predicted position")
            terms.append(float(np.linalg.norm(target_position - actual_position)) / s_p)
            terms.append(
                _orientation_error(
                    action.pose.orientation,
                    after.ee_pose.orientation,
                    f"Ju step {index} orientation",
                )
                / s_r
            )
            if action.gripper is not None:
                terms.append(
                    abs(
                        _finite(action.gripper, f"Ju step {index} commanded gripper")
                        - _finite(after.gripper_opening, f"Ju step {index} predicted gripper")
                    )
                    / s_g
                )
        elif isinstance(action, JointTarget):
            if not action.q:
                raise MissingObjectiveInput(f"Ju step {index} JointTarget has no commanded joints")
            names = sorted(action.q)
            expected_q = np.asarray(
                [_finite(action.q[name], f"Ju step {index} command {name}") for name in names]
            )
            missing = [name for name in names if name not in after.joint_pos]
            if missing:
                raise MissingObjectiveInput(f"Ju step {index} prediction lacks joints {missing}")
            actual_q = np.asarray(
                [
                    _finite(after.joint_pos[name], f"Ju step {index} prediction {name}")
                    for name in names
                ]
            )
            terms.append(float(np.linalg.norm(expected_q - actual_q)) / s_q)
        elif isinstance(action, GripperCmd):
            terms.append(
                abs(
                    _finite(action.opening, f"Ju step {index} commanded gripper")
                    - _finite(after.gripper_opening, f"Ju step {index} predicted gripper")
                )
                / s_g
            )
        else:
            raise MissingObjectiveInput(f"Ju step {index} has no comparable command field")

    if not terms:
        raise MissingObjectiveInput("Ju requires at least one comparable prediction")
    value = float(np.mean(np.asarray(terms, dtype=np.float64)))
    if not math.isfinite(value):
        raise MissingObjectiveInput("Ju produced a non-finite value")
    return value


def combine(
    components: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    maximize: bool,
    hard_veto: bool = False,
) -> float:
    """Aggregate four explicit component values; hard vetoes always precede utility."""
    if hard_veto:
        raise HardVetoPrecedesUtility("hard-vetoed candidates are not rankable by utility")
    if set(components) != set(COMPONENT_NAMES) or set(weights) != set(COMPONENT_NAMES):
        raise MissingObjectiveInput("combine requires exactly Jg, Jl, Jr and Ju")
    if not isinstance(maximize, bool):
        raise MissingObjectiveInput("ranking direction must be an explicit boolean")
    values: list[float] = []
    coefficients: list[float] = []
    for name in COMPONENT_NAMES:
        value, weight = components[name], weights[name]
        if isinstance(value, bool) or isinstance(weight, bool):
            raise MissingObjectiveInput("component values and weights must be numbers")
        value_f, weight_f = float(value), float(weight)
        if not math.isfinite(value_f) or not math.isfinite(weight_f):
            raise MissingObjectiveInput("component values and weights must be finite")
        if weight_f < 0.0:
            raise MissingObjectiveInput("weights must be nonnegative")
        values.append(value_f)
        coefficients.append(weight_f)
    if not any(coefficients):
        raise MissingObjectiveInput("at least one weight must be positive")
    score = float(np.dot(np.asarray(values), np.asarray(coefficients)))
    if not math.isfinite(score):
        raise MissingObjectiveInput("combined objective is not finite")
    return score if maximize else -score


def proposal_records() -> tuple[dict[str, Any], ...]:
    """Return the full proposal contract for owner review; no file or registry is modified."""
    return (
        *(
            dict(
                PROPOSED_OBJECTIVES[name].as_record(),
                weight_selection=WEIGHT_SELECTION,
                authoritative=False,
            )
            for name in COMPONENT_NAMES
        ),
    )
