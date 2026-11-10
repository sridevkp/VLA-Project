"""Webots-free evaluation of FK models against live GT Solid telemetry (ADR-0020).

The live ``validate-kinematics`` command drives a Webots session and collects
one :class:`JointSample` per pose: the measured absolute joint angles plus the
``kinematics_telemetry`` diagnostics record (world poses of every torso/arm
``endPoint`` Solid, both finger Solids, and the Robot-node base frame). This
module turns those samples into per-model, per-link FK errors and the ADR-0020
accept/reject decision — without importing Webots, so the math is unit-testable
on fake telemetry.

Semantics:

- FK output is expressed in the **Robot node frame** (the frame
  ``frames.world_to_base`` uses); GT Solid poses are transformed into the same
  base frame per sample. The URDF chain root is displaced from the Robot node by
  the rigid upstream residual quantified in ADR-0019, so its per-link errors
  are expected to carry that frame bias — the sim chain fixes it by
  construction, and the two reports show exactly that difference.
- The acceptance criterion applies to the **sim** model only: finger-midpoint
  error (midpoint of the two finger pad centres) mean < 2 mm and max < 5 mm
  over the sweep. The URDF model stays as the informational baseline.
- Link errors are *held out* from any fitted parameter: nothing in this module
  is tuned on the sweep before scoring. Only the failure-analysis refinement
  (below) fits, and it cross-validates strictly **by pose** (even poses fitted,
  odd poses held out) — never by run or episode.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    FINGER_TIP_OFFSET_M,
    TABLE_CLEARANCE_REQUIRED_M,
    TABLE_FOOTPRINT_CENTRE_XY_M,
    TABLE_FOOTPRINT_SIZE_XY_M,
    TABLE_TOP_HEIGHT_M,
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.sim_kinematics import (
    chain_element_index,
)

#: ADR-0020 acceptance criterion for the sim-derived model: finger-midpoint
#: mean and max error over the sweep, base frame (m).
FINGER_MEAN_ACCEPT_M = 0.002
FINGER_MAX_ACCEPT_M = 0.005
#: Joint-target sampling margin inside the recorded motor limits (rad).
POSE_MARGIN_RAD = 0.15
#: Relaxed target tolerance used to decide a sweep pose has settled (rad).
SETTLE_TOLERANCE_RAD = 0.02
#: Minimum tracked-link distance above the table top inside the footprint (m).
TABLE_PLANE_CLEARANCE_M = TABLE_CLEARANCE_REQUIRED_M

#: The two chains scored by the sweep: the repo URDF baseline and the sim model.
ModelName = Literal["urdf", "sim"]


@dataclass(frozen=True)
class JointSample:
    """One sweep pose: measured absolute angles plus its GT telemetry record."""

    joints: dict[str, float]
    telemetry: dict[str, object]


@dataclass(frozen=True)
class LinkStats:
    """FK-vs-GT position error of one chain element over the sweep (base frame)."""

    link: str
    n: int
    mean_error_m: float
    max_error_m: float


@dataclass(frozen=True)
class ModelReport:
    """Per-link and tool-level errors of one kinematic model."""

    model: str
    per_link: tuple[LinkStats, ...]
    finger_mean_m: float | None
    finger_max_m: float | None
    accepted: bool


@dataclass(frozen=True)
class RefinementCandidate:
    """One named-parameter least-squares refinement of the residual."""

    parameters: str
    fit: str
    fit_rmse_m: float
    holdout_rmse_m: float


@dataclass(frozen=True)
class RefinementReport:
    """Failure analysis: where the residual lives and what could explain it."""

    model: str
    worst_link: str
    worst_mean_m: float
    residual_variation_m: float
    candidates: tuple[RefinementCandidate, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SweepReport:
    """The whole validation outcome: both models plus the failure analysis."""

    models: dict[str, ModelReport]
    refinement: RefinementReport | None
    accepted_model: str | None


# ---------------------------------------------------------------------------
# Frame and payload helpers.
# ---------------------------------------------------------------------------


def _world_pose(payload: object) -> tuple[np.ndarray, np.ndarray] | None:
    """``(R_world_local, p_world)`` of one recorded payload, or None if incomplete."""
    if not isinstance(payload, dict):
        return None
    position = payload.get("world_xyz_m")
    rotation = payload.get("world_orientation_matrix_row_major")
    if not isinstance(position, list) or len(position) != 3:
        return None
    if not isinstance(rotation, list) or len(rotation) != 9:
        return None
    values = [float(value) for value in (*rotation, *position)]
    if not all(math.isfinite(value) for value in values):
        return None
    return np.asarray(rotation, dtype=float).reshape(3, 3), np.asarray(position, dtype=float)


def base_frame_transform(telemetry: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """``(R_base_world, p_base)`` from the telemetry ``base_frame`` record."""
    base = _world_pose(telemetry.get("base_frame"))
    if base is None:
        raise ValueError("telemetry lacks a complete base_frame world pose.")
    rotation, position = base
    return rotation.T, -rotation.T @ position


def pose_in_base(
    payload: object, base_transform: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray] | None:
    """A recorded world pose expressed in the Robot-node base frame."""
    world = _world_pose(payload)
    if world is None:
        return None
    rotation_world, position_world = world
    rotation_base, position_base = base_transform
    return rotation_base @ rotation_world, rotation_base @ position_world + position_base


def finger_midpoint_gt(
    telemetry: dict[str, object], *, finger_tip_offset_m: float = FINGER_TIP_OFFSET_M
) -> np.ndarray | None:
    """Midpoint of the two finger pad centres in the base frame, or None.

    Both finger endPoint Solids are required: a one-sided midpoint would rotate
    the tool frame about the CAN axis and could not be compared with FK.
    """
    fingers = telemetry.get("finger_endpoint_solids")
    if not isinstance(fingers, dict) or len(fingers) != 2:
        return None
    base_transform = base_frame_transform(telemetry)
    pads: list[np.ndarray] = []
    for payload in fingers.values():
        posed = pose_in_base(payload, base_transform)
        if posed is None:
            return None
        rotation, position = posed
        pads.append(position + rotation @ np.array([0.0, 0.0, -abs(finger_tip_offset_m)]))
    return np.mean(np.stack(pads), axis=0)


def _q_vector(kin: Kinematics, joints: dict[str, float], torso_value: float) -> np.ndarray:
    q = np.zeros(len(kin.chain.links), dtype=float)
    q[kin.idx[TORSO_JOINT]] = torso_value
    for name in ARM_JOINTS:
        q[kin.idx[name]] = float(joints.get(name, 0.0))
    return q


def _element_names(kin: Kinematics) -> dict[str, str]:
    """Chain element that lands in each joint's child Solid frame.

    The sim chain reaches the child Solid frame after the joint's rotation and
    its fixed ``K`` element, so the comparison element is ``sim_after_<joint>``;
    the URDF chain's joint element already ends in the child link frame.
    """
    if kin.model == "sim":
        # The torso child Solid frame is the baked FixedLink named after the
        # joint; a hinge's child Solid frame is reached only after its rotation
        # link and the fixed ``K`` element that follows it.
        names = {TORSO_JOINT: TORSO_JOINT}
        names.update({name: _sim_after_name(name) for name in ARM_JOINTS})
        return names
    return {name: name for name in (TORSO_JOINT, *ARM_JOINTS) if name in kin.idx}


def _sim_after_name(name: str) -> str:
    """``sim_after_<joint>`` for an arm joint (same rule as the chain builder)."""
    return f"sim_after_{name}"


def _tool_index(kin: Kinematics) -> int:
    if kin.model == "sim":
        return chain_element_index(kin.chain, "sim_tool_offset")
    return len(kin.chain.links) - 1


def _fk(
    kin: Kinematics, joints: dict[str, float], torso_value: float
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """``(child-frame positions per joint, tool position, tool rotation)``."""
    q = _q_vector(kin, joints, torso_value)
    matrices = kin.chain.forward_kinematics([float(value) for value in q], full_kinematics=True)
    positions = {
        name: np.asarray(matrices[kin.idx[element]][:3, 3], dtype=float)
        for name, element in _element_names(kin).items()
    }
    tool_index = _tool_index(kin)
    tool = np.asarray(matrices[tool_index][:3, 3], dtype=float)
    rotation = np.asarray(matrices[tool_index][:3, :3], dtype=float)
    return positions, tool, rotation


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------


def _torso_value(kin: Kinematics) -> float:
    return kin.torso_reference


def evaluate_model(
    samples: Sequence[JointSample],
    model: ModelName,
    *,
    finger_tip_offset_m: float = FINGER_TIP_OFFSET_M,
) -> ModelReport:
    """Per-link and finger-midpoint errors of one model over the sweep."""
    if not samples:
        raise ValueError("At least one sample is required for validation.")
    kin = Kinematics(model_override=model)
    torso_value = _torso_value(kin)
    per_link_errors: dict[str, list[float]] = {name: [] for name in _element_names(kin)}
    finger_errors: list[float] = []
    for sample in samples:
        telemetry = sample.telemetry
        base_transform = base_frame_transform(telemetry)
        fk_positions, fk_tool, _ = _fk(kin, sample.joints, torso_value)
        gt_solids = telemetry.get("joint_endpoint_solids")
        if not isinstance(gt_solids, dict):
            raise ValueError("telemetry lacks joint_endpoint_solids.")
        for name, errors in per_link_errors.items():
            posed = pose_in_base(gt_solids.get(name), base_transform)
            if posed is None:
                raise ValueError(f"telemetry lacks a complete {name} child Solid pose.")
            errors.append(float(np.linalg.norm(fk_positions[name] - posed[1])))
        midpoint = finger_midpoint_gt(telemetry, finger_tip_offset_m=finger_tip_offset_m)
        if midpoint is not None:
            finger_errors.append(float(np.linalg.norm(fk_tool - midpoint)))
    per_link = tuple(
        LinkStats(
            link=name,
            n=len(errors),
            mean_error_m=float(np.mean(errors)),
            max_error_m=float(np.max(errors)),
        )
        for name, errors in per_link_errors.items()
    )
    finger_mean = float(np.mean(finger_errors)) if finger_errors else None
    finger_max = float(np.max(finger_errors)) if finger_errors else None
    accepted = bool(
        model == "sim"
        and finger_mean is not None
        and finger_max is not None
        and finger_mean < FINGER_MEAN_ACCEPT_M
        and finger_max < FINGER_MAX_ACCEPT_M
    )
    return ModelReport(
        model=model,
        per_link=per_link,
        finger_mean_m=finger_mean,
        finger_max_m=finger_max,
        accepted=accepted,
    )


def evaluate_sweep(
    samples: Sequence[JointSample],
    *,
    models: Sequence[ModelName] = ("urdf", "sim"),
    finger_tip_offset_m: float = FINGER_TIP_OFFSET_M,
) -> SweepReport:
    """Score every model on the same poses and apply the ADR-0020 criterion."""
    reports: dict[str, ModelReport] = {
        model: evaluate_model(samples, model, finger_tip_offset_m=finger_tip_offset_m)
        for model in models
    }
    accepted_model = next((model for model, report in reports.items() if report.accepted), None)
    refinement = None
    if "sim" in reports and not reports["sim"].accepted:
        refinement = propose_refinement(
            samples, model="sim", finger_tip_offset_m=finger_tip_offset_m
        )
    return SweepReport(models=reports, refinement=refinement, accepted_model=accepted_model)


# ---------------------------------------------------------------------------
# Failure analysis: named-parameter refinement with pose-based cross-validation.
# ---------------------------------------------------------------------------


def _rmse(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def propose_refinement(
    samples: Sequence[JointSample],
    *,
    model: ModelName = "sim",
    finger_tip_offset_m: float = FINGER_TIP_OFFSET_M,
) -> RefinementReport:
    """Least-squares refinement of the residual with explicit identifiability notes.

    Only called when the acceptance criterion fails. Residual vectors are
    ``GT_midpoint - FK_tool`` in the base frame, held out strictly by pose:
    even poses are fitted, odd poses score the held-out error. Two named
    parameter families are fitted — a constant tool-frame offset and a constant
    base-frame offset — and the notes state when the data cannot distinguish
    them, or when no rigid offset explains the residual at all.
    """
    kin = Kinematics(model_override=model)
    torso_value = _torso_value(kin)
    residuals: list[np.ndarray] = []
    tool_rotations: list[np.ndarray] = []
    link_errors: dict[str, list[float]] = {}
    for sample in samples:
        base_transform = base_frame_transform(sample.telemetry)
        fk_positions, fk_tool, _ = _fk(kin, sample.joints, torso_value)
        qt = _q_vector(kin, sample.joints, torso_value)
        matrices = kin.chain.forward_kinematics(
            [float(value) for value in qt], full_kinematics=True
        )
        tool_rotation = np.asarray(matrices[_tool_index(kin)][:3, :3], dtype=float)
        midpoint = finger_midpoint_gt(sample.telemetry, finger_tip_offset_m=finger_tip_offset_m)
        if midpoint is None:
            continue
        residuals.append(midpoint - fk_tool)
        tool_rotations.append(tool_rotation)
        gt_solids = sample.telemetry.get("joint_endpoint_solids")
        if isinstance(gt_solids, dict):
            for name in _element_names(kin):
                posed = pose_in_base(gt_solids.get(name), base_transform)
                if posed is not None:
                    link_errors.setdefault(name, []).append(
                        float(np.linalg.norm(fk_positions[name] - posed[1]))
                    )
    if not residuals:
        raise ValueError("No usable samples for the refinement analysis.")
    worst_link = max(link_errors, key=lambda name: float(np.mean(link_errors[name])))
    worst_mean = float(np.mean(link_errors[worst_link]))
    residual_array = np.stack(residuals)
    # Diagnostic only: spread of the residual *vectors* about their mean. A
    # constant offset anywhere in the chain keeps this near zero; a pose-
    # dependent model error does not.
    variation = _rmse(residual_array - np.mean(residual_array, axis=0))
    fit_idx = list(range(0, len(residual_array), 2))
    holdout_idx = list(range(1, len(residual_array), 2))
    candidates: list[RefinementCandidate] = []
    notes: list[str] = [
        "cross-validated by pose: even poses fitted, odd poses held out (never by run).",
    ]
    if not holdout_idx:
        notes.append("insufficient poses for a holdout split; refinement is fit-only.")
        holdout_idx = fit_idx
    fit_residuals = residual_array[fit_idx]
    holdout_residuals = residual_array[holdout_idx]
    base_offset = np.mean(fit_residuals, axis=0)
    candidates.append(
        RefinementCandidate(
            parameters="base_frame_offset_xyz_m",
            fit="constant residual vector in the base frame",
            fit_rmse_m=_rmse(fit_residuals - base_offset),
            holdout_rmse_m=_rmse(holdout_residuals - base_offset),
        )
    )
    fit_tool_residuals = np.stack(
        [tool_rotations[index].T @ residual_array[index] for index in fit_idx]
    )
    holdout_tool_residuals = np.stack(
        [tool_rotations[index].T @ residual_array[index] for index in holdout_idx]
    )
    tool_offset = np.mean(fit_tool_residuals, axis=0)
    candidates.append(
        RefinementCandidate(
            parameters="tool_offset_xyz_m",
            fit="constant residual vector in the arm_7/tool frame",
            fit_rmse_m=_rmse(fit_tool_residuals - tool_offset),
            holdout_rmse_m=_rmse(holdout_tool_residuals - tool_offset),
        )
    )
    baseline_holdout = _rmse(holdout_residuals)
    best = min(candidates, key=lambda candidate: candidate.holdout_rmse_m)
    if baseline_holdout > 0.0 and best.holdout_rmse_m < 0.5 * baseline_holdout:
        notes.append(
            f"the {best.parameters} family reduces the held-out residual from "
            f"{baseline_holdout * 1e3:.2f} mm to {best.holdout_rmse_m * 1e3:.2f} mm."
        )
    else:
        notes.append(
            "no rigid-offset family explains the held-out residual "
            f"({best.holdout_rmse_m * 1e3:.2f} mm of {baseline_holdout * 1e3:.2f} mm): "
            "the error is pose-dependent, i.e. axis/scale/structure, not a frame offset."
        )
    if candidates[0].holdout_rmse_m > 0.0 and abs(
        candidates[0].holdout_rmse_m - candidates[1].holdout_rmse_m
    ) < 0.1 * max(candidates[0].holdout_rmse_m, candidates[1].holdout_rmse_m):
        notes.append(
            "base-frame and tool-frame offsets fit the pose data equally well: the two "
            "parameter families are not separable from poses alone (identifiability note)."
        )
    if baseline_holdout > 0.0 and worst_mean < 0.2 * baseline_holdout:
        notes.append(
            f"every tracked link frame agrees to <= {worst_mean * 1e3:.2f} mm while the "
            f"tool-level residual is {baseline_holdout * 1e3:.2f} mm: the residual is "
            "introduced between the arm_7 child Solid and the finger pad frame (tool offset "
            "or finger geometry), not in the arm chain."
        )
    else:
        notes.append(
            f"the {worst_link} link frame is off by {worst_mean * 1e3:.2f} mm mean "
            f"(tool residual {baseline_holdout * 1e3:.2f} mm), so the error is already present "
            "at or before that segment; compare the per-link rows to find the first affected "
            "joint."
        )
    return RefinementReport(
        model=model,
        worst_link=worst_link,
        worst_mean_m=worst_mean,
        residual_variation_m=variation,
        candidates=tuple(candidates),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Pose sampling and safety pre-check (also Webots-free).
# ---------------------------------------------------------------------------


def pose_limits(
    snapshot: dict[str, object] | None,
) -> dict[str, tuple[float, float]]:
    """Per-joint absolute motor limits: the snapshot's when present, else config.

    The snapshot records the Webots motor limits at export time, which is the
    authority for the sweep; the config fallback only keeps the command usable
    when the snapshot predates the limits fields (it then samples a generic
    +-2 rad window, clearly reported).
    """
    limits: dict[str, tuple[float, float]] = {}
    entries = snapshot.get("joints") if isinstance(snapshot, dict) else None
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or name not in ARM_JOINTS:
            continue
        bounds = entry.get("limits")
        if not isinstance(bounds, dict):
            continue
        low = bounds.get("min_position")
        high = bounds.get("max_position")
        if (
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and float(high) > float(low)
        ):
            limits[name] = (float(low), float(high))
    for name in ARM_JOINTS:
        limits.setdefault(name, (-2.0, 2.0))
    return limits


def diverse_pose_targets(
    *,
    n: int,
    seed: int,
    limits: dict[str, tuple[float, float]],
    margin_rad: float = POSE_MARGIN_RAD,
) -> list[dict[str, float]]:
    """``n`` deterministic, stratified arm-joint targets inside the limits.

    Each joint's sampled values are spread over evenly spaced quantiles (then
    jittered) and the joint order is shuffled per pose, so the sweep covers the
    workspace instead of clustering. The torso is never sampled: the sweep holds
    it fixed at the settled export value (M1 sim model is baked there).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(limits) < len(ARM_JOINTS):
        raise ValueError("pose_limits must cover every arm joint.")
    rng = np.random.default_rng(seed)
    joint_order = list(ARM_JOINTS)
    positions = {name: rng.permutation(n) for name in joint_order}
    targets: list[dict[str, float]] = []
    for index in range(n):
        target: dict[str, float] = {}
        for name in joint_order:
            low, high = limits[name]
            span = max(high - low - 2.0 * margin_rad, 0.0)
            quantile = (positions[name][index] + 0.5) / n
            jitter = float(rng.uniform(-0.5, 0.5)) * span / n
            value = low + margin_rad + quantile * span + jitter
            target[name] = float(min(max(value, low + margin_rad), high - margin_rad))
        targets.append(target)
    return targets


def sample_is_safe(
    telemetry: dict[str, object],
    *,
    clearance_m: float = TABLE_PLANE_CLEARANCE_M,
    table_top_height_m: float = TABLE_TOP_HEIGHT_M,
    footprint_centre_xy_m: tuple[float, float] = TABLE_FOOTPRINT_CENTRE_XY_M,
    footprint_size_xy_m: tuple[float, float] = TABLE_FOOTPRINT_SIZE_XY_M,
) -> tuple[bool, str]:
    """Obstacle check for one sweep pose, by design from GT Solid poses.

    Every tracked link (arm child Solids and both finger Solids) whose XY lies
    inside the table footprint must stay at least ``clearance_m`` above the
    table top; the pose is aborted otherwise. The check is deliberately
    conservative about *which* link it evaluates: the lowest tracked point
    decides.
    """
    arm_tracked: dict[str, object] = {}
    solids = telemetry.get("joint_endpoint_solids")
    if isinstance(solids, dict):
        arm_tracked.update(
            {name: payload for name, payload in solids.items() if name in ARM_JOINTS}
        )
    if not arm_tracked:
        return False, "no tracked link poses in telemetry"
    tracked: dict[str, object] = dict(arm_tracked)
    fingers = telemetry.get("finger_endpoint_solids")
    if isinstance(fingers, dict):
        tracked.update(fingers)
    half_x = footprint_size_xy_m[0] / 2.0
    half_y = footprint_size_xy_m[1] / 2.0
    centre_x, centre_y = footprint_centre_xy_m
    plane = table_top_height_m + clearance_m
    for name, payload in tracked.items():
        world = _world_pose(payload)
        if world is None:
            # Finger endPoint Solids can be unreadable in some exports; arm links must be complete.
            if name in arm_tracked:
                return False, f"{name} has no complete world pose"
            continue
        _, position = world
        inside = abs(position[0] - centre_x) <= half_x and abs(position[1] - centre_y) <= half_y
        if inside and position[2] < plane:
            return (
                False,
                f"{name} at z={position[2]:.3f} m is below {plane:.3f} m inside the "
                "table footprint",
            )
    return True, "all tracked links clear the table plane"


# ---------------------------------------------------------------------------
# Text rendering for the CLI.
# ---------------------------------------------------------------------------


def format_report(report: SweepReport) -> list[str]:
    """Human-readable lines for the validation outcome (CLI prints them)."""
    lines: list[str] = []
    for model, model_report in report.models.items():
        lines.append(
            f"model={model} finger_mean_mm={_mm(model_report.finger_mean_m)} "
            f"finger_max_mm={_mm(model_report.finger_max_m)} accepted={model_report.accepted}"
        )
        for stats in model_report.per_link:
            lines.append(
                f"  link={stats.link} n={stats.n} "
                f"mean_mm={stats.mean_error_m * 1e3:.3f} max_mm={stats.max_error_m * 1e3:.3f}"
            )
    if report.accepted_model is None:
        lines.append("result=REJECT no model met the ADR-0020 acceptance criterion")
    else:
        lines.append(f"result=ACCEPT model={report.accepted_model}")
    refinement = report.refinement
    if refinement is not None:
        lines.append(
            f"refinement model={refinement.model} worst_link={refinement.worst_link} "
            f"worst_mean_mm={refinement.worst_mean_m * 1e3:.3f} "
            f"residual_variation_mm={refinement.residual_variation_m * 1e3:.3f}"
        )
        for candidate in refinement.candidates:
            lines.append(
                f"  candidate={candidate.parameters} fit={candidate.fit!r} "
                f"fit_rmse_mm={candidate.fit_rmse_m * 1e3:.3f} "
                f"holdout_rmse_mm={candidate.holdout_rmse_m * 1e3:.3f}"
            )
        for note in refinement.notes:
            lines.append(f"  note: {note}")
    return lines


def _mm(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1e3:.3f}"
