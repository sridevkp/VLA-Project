"""Proposed Eq. 3.8 component definitions — ENGINEERING CHOICES, NOT REPORT QUOTATIONS.

The submitted Phase I report gives only the verbal meanings of Jg, Jl, Jr and Ju and the form
``J = wg*Jg + wl*Jl + wr*Jr + wu*Ju``. It does NOT supply formulas, units, normalisation
constants, weighting values, ranking direction or a calibrated risk threshold. Everything in this
module is an explicit proposal for owner review.

Nothing here is registered as a WorldMind component binding, evaluator, weight or threshold. The
production registry stays empty for these identities; a `ready` config referencing them fails
closed. These functions are inert: they read only the numeric inputs passed to them and never
access a backend, campaign, TEST/TRAIN split or wall clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vlarobot.core.types import Pose3D, Vec3, WorldState
from vlarobot.world_model.analytic import ObjectWorldModel

COMPONENT_NAMES = ("Jg", "Jl", "Jr", "Ju")
PROPOSAL_STATUS = "engineering_choice_not_report_quotation"
SAFETY_PRECEDENCE = (
    "Safety is a hard veto (SafetyMonitor.validate + RiskGate + unverified-arm-sweep). "
    "A candidate rejected by a hard check is never ranked; no utility term may offset a veto."
)
MISSING_DATA_BEHAVIOUR = (
    "Fail closed with MissingObjectiveInput. Never impute zero error, a synthetic goal, an "
    "unmeasured clearance, a wall-clock duration or an invented uncertainty."
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
        "mean_k clip((m - d_min,k) / m, 0, 1); d_min,k < 0 maps to 1",
        "dimensionless ratio in [0, 1]",
        "m in m > 0 is a declared clearance margin; d_min,k is certified swept-arm clearance",
        "minimize",
        ("certified full-arm minimum clearance per predicted step", "clearance margin m"),
        MISSING_DATA_BEHAVIOUR
        + " Unverified geometry yields MissingObjectiveInput, never zero risk.",
        SAFETY_PRECEDENCE + " Collision hard checks still veto independently of Jr.",
        PROPOSAL_STATUS,
    ),
    "Ju": ObjectiveProposal(
        "Ju",
        "mean_k (||e_p,k|| / s_p + e_theta,k / s_r)",
        "dimensionless prediction-discrepancy proxy (NOT calibrated uncertainty)",
        "s_p in m > 0; s_r in rad > 0; owner-selected on VALIDATION",
        "minimize",
        ("aligned predicted and corrected WorldState trajectories", "at least two paired steps"),
        MISSING_DATA_BEHAVIOUR + " Missing alignment is an error, not zero uncertainty.",
        SAFETY_PRECEDENCE,
        PROPOSAL_STATUS,
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
    """Proposed Jr: bounded mean clearance deficit from a certified full-arm sweep. Minimize."""
    margin = _positive(margin_m, "margin_m")
    values = _vector(tuple(clearances_m), "clearances_m")
    deficit = np.clip((margin - values) / margin, 0.0, 1.0)
    value = float(np.mean(deficit))
    if not math.isfinite(value):
        raise MissingObjectiveInput("Jr produced a non-finite value")
    return value


def prediction_penalty(
    predicted: Sequence[WorldState],
    corrected: Sequence[WorldState],
    *,
    position_scale_m: float,
    orientation_scale_rad: float,
) -> float:
    """Proposed Ju: normalised, aligned prediction-vs-correction discrepancy proxy. Minimize."""
    s_p = _positive(position_scale_m, "position_scale_m")
    s_r = _positive(orientation_scale_rad, "orientation_scale_rad")
    if (
        not isinstance(predicted, Sequence)
        or not isinstance(corrected, Sequence)
        or len(predicted) != len(corrected)
        or len(predicted) < 2
    ):
        raise MissingObjectiveInput("Ju requires aligned predicted and corrected trajectories")
    correction = ObjectWorldModel()
    terms: list[float] = []
    for index, (before, after) in enumerate(zip(predicted, corrected, strict=True)):
        if not isinstance(before, WorldState) or not isinstance(after, WorldState):
            raise MissingObjectiveInput(f"Ju step {index} is not a WorldState")
        ee_delta = _xyz(before.robot.ee_pose.position, "predicted ee") - _xyz(
            after.robot.ee_pose.position, "corrected ee"
        )
        terms.append(float(np.linalg.norm(ee_delta)) / s_p)
        if index == 0:
            continue
        error = correction.correct(before, after)
        if not math.isfinite(error.position_error) or not math.isfinite(error.orientation_error):
            raise MissingObjectiveInput("Ju received a nonfinite object correction")
        terms.append(error.position_error / s_p)
        terms.append(error.orientation_error / s_r)
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
