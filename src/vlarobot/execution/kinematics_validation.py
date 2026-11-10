"""Webots-free evaluation and redesigned feasible FK validation sweep (ADR-0020).

The live ``validate-kinematics`` command drives a Webots session and collects
scored samples: measured absolute joint angles plus ``kinematics_telemetry``.
This module provides pose design, stationarity settle helpers, scoring, and the
accept/reject decision without importing Webots.

Sweep redesign (feasible, no Webots required to evaluate): 12 poses total —
6 spread + 6 near the PREGRASP/DESCEND working volume (tool-vertical bias in
joint space). Poses are reached by joint sub-moves bounded to
``0.4 * velocity_limit_rad_s * horizon_s`` per joint so the safety layer does
not reject; each sub-move settles by stationarity (|delta| < threshold for N
consecutive steps) with a cap of 60 steps per sub-move. A pose that cannot
settle or is unsafe fails fast with a clear reason and the sweep continues.

Acceptance (12 poses): mean < 3 mm, max < 6 mm on ``arm_7`` endpoint and
``wrist`` in the Robot/base frame (finger midpoint only when the tool frame is
verified). Results are reported separately for the whole set and the near-task
poses. The tool frame is labeled FALLBACK/UNVERIFIED unless explicitly verified.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

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
from vlarobot.backends.webots.sim_kinematics import chain_element_index

# ADR-0020 redesign acceptance (12 poses): arm_7 endpoint and wrist, base frame.
LINK_MEAN_ACCEPT_M = 0.003
LINK_MAX_ACCEPT_M = 0.006
REQUIRED_LINKS: tuple[str, ...] = ("arm_7_joint", "arm_6_joint")  # wrist=end of arm_6 child frame
SETTLE_DELTA_RAD = 0.0005
SETTLE_STEPS = 3
SETTLE_SUBMOVE_CAP = 60
SUBMOVE_SAFETY_FRACTION = 0.4
DEFAULT_VELOCITY_LIMIT_RAD_S = 1.0
DEFAULT_HORIZON_S = 1.0
POSE_MARGIN_RAD = 0.10

ModelName = Literal["urdf", "sim"]


@dataclass(frozen=True)
class JointSample:
    joints: dict[str, float]
    telemetry: dict[str, object]


@dataclass(frozen=True)
class PoseSpec:
    index: int
    group: str  # "spread" | "near_task"
    target: dict[str, float]


@dataclass(frozen=True)
class LinkStats:
    link: str
    n: int
    mean_error_m: float
    max_error_m: float


@dataclass(frozen=True)
class GroupDecision:
    group: str
    n: int
    mean_m: float | None
    max_m: float | None
    accepted: bool


@dataclass(frozen=True)
class ModelReport:
    model: str
    per_link: tuple[LinkStats, ...]
    scored_links: tuple[str, ...]
    finger_mean_m: float | None
    finger_max_m: float | None
    tool_verified: bool
    whole: GroupDecision
    near_task: GroupDecision
    accepted: bool


@dataclass(frozen=True)
class RefinementCandidate:
    parameters: str
    fit: str
    fit_rmse_m: float
    holdout_rmse_m: float


@dataclass(frozen=True)
class RefinementReport:
    model: str
    worst_link: str
    worst_mean_m: float
    residual_variation_m: float
    candidates: tuple[RefinementCandidate, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SweepReport:
    models: dict[str, ModelReport]
    refinement: RefinementReport | None
    accepted_model: str | None
    tool_status: str


# ---------------------------------------------------------------------------
# Frame/payload helpers
# ---------------------------------------------------------------------------


def _world_pose(payload: object) -> tuple[np.ndarray, np.ndarray] | None:
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
    base = _world_pose(telemetry.get("base_frame"))
    if base is None:
        raise ValueError("telemetry lacks a complete base_frame world pose.")
    rotation, position = base
    return rotation.T, -rotation.T @ position


def pose_in_base(
    payload: object, base_transform: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray] | None:
    world = _world_pose(payload)
    if world is None:
        return None
    rotation_world, position_world = world
    rotation_base, position_base = base_transform
    return rotation_base @ rotation_world, rotation_base @ position_world + position_base


def finger_midpoint_gt(
    telemetry: dict[str, object], *, finger_tip_offset_m: float = FINGER_TIP_OFFSET_M
) -> np.ndarray | None:
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
    if kin.model == "sim":
        names = {TORSO_JOINT: TORSO_JOINT}
        names.update({name: f"sim_after_{name}" for name in ARM_JOINTS})
        return names
    return {name: name for name in (TORSO_JOINT, *ARM_JOINTS) if name in kin.idx}


def _tool_index(kin: Kinematics) -> int:
    if kin.model == "sim":
        return chain_element_index(kin.chain, "sim_tool_offset")
    return len(kin.chain.links) - 1


def _fk(
    kin: Kinematics, joints: dict[str, float], torso_value: float
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
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
# Pose design and budget
# ---------------------------------------------------------------------------


def pose_limits(snapshot: dict[str, object] | None) -> dict[str, tuple[float, float]]:
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


def _clamp_target(
    target: dict[str, float], limits: dict[str, tuple[float, float]], margin: float
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ARM_JOINTS:
        low, high = limits[name]
        lo, hi = low + margin, high - margin
        if lo > hi:
            lo = hi = (low + high) / 2.0
        out[name] = float(min(max(float(target.get(name, (low + high) / 2.0)), lo), hi))
    return out


def feasible_pose_plan(*, seed: int, limits: dict[str, tuple[float, float]]) -> list[PoseSpec]:
    """12 poses: 6 spread + 6 near-task (PREGRASP/DESCEND working volume, tool-vertical bias)."""
    rng = np.random.default_rng(seed)
    specs: list[PoseSpec] = []
    # 6 spread stratified poses
    n_spread = 6
    positions = {name: rng.permutation(n_spread) for name in ARM_JOINTS}
    for index in range(n_spread):
        target: dict[str, float] = {}
        for name in ARM_JOINTS:
            low, high = limits[name]
            span = max(high - low - 2.0 * POSE_MARGIN_RAD, 0.0)
            quantile = (positions[name][index] + 0.5) / n_spread
            value = low + POSE_MARGIN_RAD + quantile * span
            target[name] = float(min(max(value, low + POSE_MARGIN_RAD), high - POSE_MARGIN_RAD))
        specs.append(PoseSpec(index=index, group="spread", target=target))
    # 6 near-task poses: deterministic joint-space configurations biased to forward reach and
    # downward/tool-vertical wrist orientations in the working volume (heuristic, joint-space).
    near_templates = [
        {
            "arm_1_joint": 0.35,
            "arm_2_joint": 0.20,
            "arm_3_joint": -1.20,
            "arm_4_joint": 1.45,
            "arm_5_joint": 0.00,
            "arm_6_joint": 0.85,
            "arm_7_joint": 0.00,
        },
        {
            "arm_1_joint": 0.55,
            "arm_2_joint": 0.10,
            "arm_3_joint": -1.40,
            "arm_4_joint": 1.55,
            "arm_5_joint": 0.20,
            "arm_6_joint": 0.95,
            "arm_7_joint": 0.10,
        },
        {
            "arm_1_joint": 0.25,
            "arm_2_joint": 0.30,
            "arm_3_joint": -1.00,
            "arm_4_joint": 1.35,
            "arm_5_joint": -0.20,
            "arm_6_joint": 0.75,
            "arm_7_joint": -0.10,
        },
        {
            "arm_1_joint": 0.45,
            "arm_2_joint": -0.05,
            "arm_3_joint": -1.30,
            "arm_4_joint": 1.60,
            "arm_5_joint": 0.00,
            "arm_6_joint": 1.05,
            "arm_7_joint": 0.00,
        },
        {
            "arm_1_joint": 0.65,
            "arm_2_joint": 0.15,
            "arm_3_joint": -1.10,
            "arm_4_joint": 1.40,
            "arm_5_joint": 0.30,
            "arm_6_joint": 0.90,
            "arm_7_joint": 0.20,
        },
        {
            "arm_1_joint": 0.30,
            "arm_2_joint": 0.05,
            "arm_3_joint": -1.50,
            "arm_4_joint": 1.50,
            "arm_5_joint": -0.10,
            "arm_6_joint": 1.00,
            "arm_7_joint": -0.05,
        },
    ]
    for i, tmpl in enumerate(near_templates):
        specs.append(
            PoseSpec(
                index=n_spread + i,
                group="near_task",
                target=_clamp_target(tmpl, limits, POSE_MARGIN_RAD),
            )
        )
    return specs


def submove_bound(velocity_limit_rad_s: float, horizon_s: float) -> float:
    return (
        SUBMOVE_SAFETY_FRACTION
        * max(1e-6, float(velocity_limit_rad_s))
        * max(1e-6, float(horizon_s))
    )


def plan_submoves(
    current: dict[str, float], target: dict[str, float], bound: float
) -> list[dict[str, float]]:
    max_err = max(abs(float(target[n]) - float(current[n])) for n in ARM_JOINTS)
    segments = max(1, int(math.ceil(max_err / max(1e-9, bound))))
    steps: list[dict[str, float]] = []
    for seg in range(1, segments + 1):
        alpha = seg / segments
        steps.append(
            {
                n: float(current[n]) + alpha * (float(target[n]) - float(current[n]))
                for n in ARM_JOINTS
            }
        )
    return steps


def estimate_step_budget(
    poses: Sequence[PoseSpec],
    start: dict[str, float],
    bound: float,
    ms_per_step: float | None,
) -> dict[str, object]:
    current = dict(start)
    total_submoves = 0
    for spec in poses:
        subs = plan_submoves(current, spec.target, bound)
        total_submoves += len(subs)
        current = dict(spec.target)
    # cap steps per sub-move = SETTLE_SUBMOVE_CAP
    max_steps = total_submoves * SETTLE_SUBMOVE_CAP
    est = {
        "poses": len(poses),
        "total_submoves": total_submoves,
        "steps_per_submove_cap": SETTLE_SUBMOVE_CAP,
        "max_total_steps": max_steps,
        "ms_per_step": ms_per_step,
    }
    if ms_per_step is not None and ms_per_step > 0:
        est["estimated_s"] = max_steps * ms_per_step / 1000.0
        est["estimated_min"] = est["estimated_s"] / 60.0
    return est


# ---------------------------------------------------------------------------
# Safety pre-check
# ---------------------------------------------------------------------------


def sample_is_safe(
    telemetry: dict[str, object],
    *,
    clearance_m: float = TABLE_CLEARANCE_REQUIRED_M,
    table_top_height_m: float = TABLE_TOP_HEIGHT_M,
    footprint_centre_xy_m: tuple[float, float] = TABLE_FOOTPRINT_CENTRE_XY_M,
    footprint_size_xy_m: tuple[float, float] = TABLE_FOOTPRINT_SIZE_XY_M,
) -> tuple[bool, str]:
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
            if name in arm_tracked:
                return False, f"{name} has no complete world pose"
            continue
        _, position = world
        inside = abs(position[0] - centre_x) <= half_x and abs(position[1] - centre_y) <= half_y
        if inside and position[2] < plane:
            return (
                False,
                f"{name} at z={position[2]:.3f} m is below {plane:.3f} m "
                "inside the table footprint",
            )
    return True, "all tracked links clear the table plane"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _link_errors_for_samples(
    kin: Kinematics, samples: Sequence[JointSample], links: Sequence[str]
) -> tuple[dict[str, list[float]], list[float]]:
    torso_value = kin.torso_reference
    per_link: dict[str, list[float]] = {name: [] for name in links}
    finger_errors: list[float] = []
    for sample in samples:
        telemetry = sample.telemetry
        base_transform = base_frame_transform(telemetry)
        fk_positions, fk_tool, _ = _fk(kin, sample.joints, torso_value)
        gt_solids = telemetry.get("joint_endpoint_solids")
        if not isinstance(gt_solids, dict):
            raise ValueError("telemetry lacks joint_endpoint_solids.")
        for name in links:
            posed = pose_in_base(gt_solids.get(name), base_transform)
            if posed is None or name not in fk_positions:
                raise ValueError(f"telemetry lacks a complete {name} child Solid pose.")
            per_link[name].append(float(np.linalg.norm(fk_positions[name] - posed[1])))
        midpoint = finger_midpoint_gt(telemetry)
        if midpoint is not None:
            finger_errors.append(float(np.linalg.norm(fk_tool - midpoint)))
    return per_link, finger_errors


def _decision(errors: list[float], group: str) -> GroupDecision:
    if not errors:
        return GroupDecision(group=group, n=0, mean_m=None, max_m=None, accepted=False)
    mean_v = float(np.mean(errors))
    max_v = float(np.max(errors))
    return GroupDecision(
        group=group,
        n=len(errors),
        mean_m=mean_v,
        max_m=max_v,
        accepted=(mean_v < LINK_MEAN_ACCEPT_M and max_v < LINK_MAX_ACCEPT_M),
    )


def evaluate_model(
    samples: Sequence[tuple[JointSample, str]],
    model: ModelName,
    *,
    tool_verified: bool = False,
) -> ModelReport:
    if not samples:
        raise ValueError("At least one sample is required for validation.")
    kin = Kinematics(model_override=model)
    torso_value = kin.torso_reference
    scored_links = tuple(name for name in REQUIRED_LINKS if name in _element_names(kin))
    if not scored_links:
        raise ValueError("no required scoring links available in chain.")
    per_link_errors: dict[str, list[float]] = {name: [] for name in _element_names(kin)}
    scored_all: list[float] = []
    scored_near: list[float] = []
    finger_errors: list[float] = []
    for sample, group in samples:
        telemetry = sample.telemetry
        base_transform = base_frame_transform(telemetry)
        fk_positions, fk_tool, _ = _fk(kin, sample.joints, torso_value)
        gt_solids = telemetry.get("joint_endpoint_solids")
        if not isinstance(gt_solids, dict):
            raise ValueError("telemetry lacks joint_endpoint_solids.")
        pose_scored_vals: list[float] = []
        for name, errors in per_link_errors.items():
            posed = pose_in_base(gt_solids.get(name), base_transform)
            if posed is None:
                raise ValueError(f"telemetry lacks a complete {name} child Solid pose.")
            err = float(np.linalg.norm(fk_positions[name] - posed[1]))
            errors.append(err)
            if name in scored_links:
                pose_scored_vals.append(err)
        if pose_scored_vals:
            v = float(max(pose_scored_vals))
            scored_all.append(v)
            if group == "near_task":
                scored_near.append(v)
        midpoint = finger_midpoint_gt(telemetry)
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
    whole = _decision(scored_all, "whole")
    near = _decision(scored_near, "near_task")
    accepted = bool(model == "sim" and whole.accepted and near.accepted)
    return ModelReport(
        model=model,
        per_link=per_link,
        scored_links=scored_links,
        finger_mean_m=finger_mean,
        finger_max_m=finger_max,
        tool_verified=tool_verified,
        whole=whole,
        near_task=near,
        accepted=accepted,
    )


def evaluate_sweep(
    samples: Sequence[tuple[JointSample, str]],
    *,
    models: Sequence[ModelName] = ("urdf", "sim"),
    tool_verified: bool = False,
) -> SweepReport:
    reports: dict[str, ModelReport] = {
        model: evaluate_model(samples, model, tool_verified=tool_verified) for model in models
    }
    accepted_model = next((model for model, report in reports.items() if report.accepted), None)
    refinement = None
    if "sim" in reports and not reports["sim"].accepted:
        try:
            refinement = propose_refinement(samples, model="sim")
        except Exception:
            refinement = None
    return SweepReport(
        models=cast(dict[str, ModelReport], reports),
        refinement=refinement,
        accepted_model=accepted_model,
        tool_status="VERIFIED" if tool_verified else "FALLBACK/UNVERIFIED",
    )


# ---------------------------------------------------------------------------
# Refinement (best-effort, pose holdout)
# ---------------------------------------------------------------------------


def _rmse(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def propose_refinement(
    samples: Sequence[tuple[JointSample, str]], *, model: ModelName = "sim"
) -> RefinementReport:
    kin = Kinematics(model_override=model)
    torso_value = kin.torso_reference
    residuals: list[np.ndarray] = []
    tool_rotations: list[np.ndarray] = []
    link_errors: dict[str, list[float]] = {}
    for sample, _group in samples:
        base_transform = base_frame_transform(sample.telemetry)
        fk_positions, fk_tool, _ = _fk(kin, sample.joints, torso_value)
        q = _q_vector(kin, sample.joints, torso_value)
        matrices = kin.chain.forward_kinematics([float(v) for v in q], full_kinematics=True)
        tool_rotation = np.asarray(matrices[_tool_index(kin)][:3, :3], dtype=float)
        midpoint = finger_midpoint_gt(sample.telemetry)
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
    worst_link = (
        max(link_errors, key=lambda name: float(np.mean(link_errors[name])))
        if link_errors
        else "unknown"
    )
    worst_mean = float(np.mean(link_errors[worst_link])) if link_errors else 0.0
    residual_array = np.stack(residuals)
    variation = _rmse(residual_array - np.mean(residual_array, axis=0))
    fit_idx = list(range(0, len(residual_array), 2))
    holdout_idx = list(range(1, len(residual_array), 2)) or fit_idx
    fit_res = residual_array[fit_idx]
    hold_res = residual_array[holdout_idx]
    base_offset = np.mean(fit_res, axis=0)
    candidates = [
        RefinementCandidate(
            "base_frame_offset_xyz_m",
            "constant residual vector in the base frame",
            _rmse(fit_res - base_offset),
            _rmse(hold_res - base_offset),
        ),
    ]
    fit_tool = np.stack([tool_rotations[i].T @ residual_array[i] for i in fit_idx])
    hold_tool = np.stack([tool_rotations[i].T @ residual_array[i] for i in holdout_idx])
    tool_offset = np.mean(fit_tool, axis=0)
    candidates.append(
        RefinementCandidate(
            "tool_offset_xyz_m",
            "constant residual vector in the arm_7/tool frame",
            _rmse(fit_tool - tool_offset),
            _rmse(hold_tool - tool_offset),
        )
    )
    notes = [
        "cross-validated by pose: even poses fitted, odd poses held out (never by run).",
        f"worst_link={worst_link} mean_mm={worst_mean * 1e3:.3f} "
        f"residual_variation_mm={variation * 1e3:.3f}.",
    ]
    return RefinementReport(
        model=model,
        worst_link=worst_link,
        worst_mean_m=worst_mean,
        residual_variation_m=variation,
        candidates=tuple(candidates),
        notes=tuple(notes),
    )


def format_report(report: SweepReport) -> list[str]:
    lines = [f"tool_frame={report.tool_status} (finger midpoint scored only when VERIFIED)"]
    for model, model_report in report.models.items():
        lines.append(
            f"model={model} scored_links={','.join(model_report.scored_links)} "
            f"whole_mean_mm={_mm(model_report.whole.mean_m)} "
            f"whole_max_mm={_mm(model_report.whole.max_m)} "
            f"whole_accepted={model_report.whole.accepted} (n={model_report.whole.n}) "
            f"near_mean_mm={_mm(model_report.near_task.mean_m)} "
            f"near_max_mm={_mm(model_report.near_task.max_m)} "
            f"near_accepted={model_report.near_task.accepted} "
            f"(n={model_report.near_task.n}) "
            f"accepted={model_report.accepted}"
        )
        if model_report.finger_mean_m is not None:
            lines.append(
                f"  finger_midpoint_mean_mm={_mm(model_report.finger_mean_m)} "
                f"finger_midpoint_max_mm={_mm(model_report.finger_max_m)} "
                "(informational; "
                f"{'counted' if model_report.tool_verified else 'NOT counted'}"
                ", tool FALLBACK/UNVERIFIED)"
            )
        for stats in model_report.per_link:
            lines.append(
                f"  link={stats.link} n={stats.n} "
                f"mean_mm={stats.mean_error_m * 1e3:.3f} max_mm={stats.max_error_m * 1e3:.3f}"
            )
    if report.accepted_model is None:
        lines.append("result=REJECT no model met the redesigned acceptance criterion")
    else:
        lines.append(f"result=ACCEPT model={report.accepted_model}")
    if report.refinement is not None:
        r = report.refinement
        lines.append(
            f"refinement worst_link={r.worst_link} worst_mean_mm={r.worst_mean_m * 1e3:.3f}"
        )
        for c in r.candidates:
            lines.append(
                f"  candidate={c.parameters} fit_rmse_mm={c.fit_rmse_m * 1e3:.3f} "
                f"holdout_rmse_mm={c.holdout_rmse_m * 1e3:.3f}"
            )
        for note in r.notes:
            lines.append(f"  note: {note}")
    return lines


def _mm(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1e3:.3f}"
