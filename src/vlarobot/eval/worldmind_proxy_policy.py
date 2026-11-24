"""Approved November 24 WorldMind objective and simulation proxy safety scope."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from vlarobot.core.types import ActionChunk, ObjectState, Vec3, WorldState
from vlarobot.eval.worldmind_objectives import (
    MissingObjectiveInput,
    predicted_risk,
    tool_envelope_clearances,
)
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.analytic import segment_box_clearance
from vlarobot.world_model.rollout import Trajectory, rotate

PROXY_MARGIN_M = 0.03
# Target stroke and collision boxes from both GRIPPER_BO groups in R2025a TiagoGripper.proto.
# Bound all corners about the finger-pad tool point, including maximum slider translation.
TOOL_RADIUS_M = max(
    math.hypot(
        abs(x) + sx / 2 + 0.045,
        abs(y) + sy / 2,
        abs(z + 0.1741) + sz / 2,
    )
    for x, y, z, sx, sy, sz in (
        (0.004, 0, -0.1741, 0.0076, 0.05091, 0.08982),
        (0.0049, 0.0153, -0.1208, 0.0055, 0.0295, 0.018),
        (-0.0085, 0.0156, -0.1048, 0.03213, 0.02959, 0.01487),
    )
)
GRID = {"equal": (1.0, 1.0, 1.0, 0.0), "goal_x2": (2.0, 1.0, 1.0, 0.0)}
OPTIONAL_GRID = {"risk_x2": (1.0, 1.0, 2.0, 0.0)}
HARD_CHECKS = (
    "joint_limit_margin",
    "ik_feasibility_and_residual",
    "table_clearance",
    "object_clearance",
    "placement_tolerance",
)
LEGACY_SWEEP_REASONS = frozenset({"unverified-arm-sweep", "arm-sweep-collision"})


@dataclass(frozen=True)
class ProxyEvidence:
    """Named decision-time measurements. Missing source measurements cannot pass."""

    joint_limits_pass: bool
    ik_pass: bool
    # One signed clearance per predicted transition in each channel.
    table_clearance_m: tuple[float, ...]
    object_clearance_m: tuple[float, ...]


def observe_proxy_clearance(
    current: WorldState,
    chunk: ActionChunk,
    prediction: Trajectory,
    *,
    table_boxes: Sequence[ObjectState],
    expected_object_ids: frozenset[str],
    joint_limits_pass: bool,
    ik_pass: bool,
) -> ProxyEvidence:
    """Compute clearance from separately identified tables and observed object boxes."""
    object_ids = [obj.object_id for obj in current.scene.objects]
    table_ids = [obj.object_id for obj in table_boxes]
    if (
        not table_ids
        or len(table_ids) != len(set(table_ids))
        or set(table_ids).intersection(object_ids)
        or len(object_ids) != len(expected_object_ids)
        or set(object_ids) != expected_object_ids
    ):
        raise MissingObjectiveInput("proxy requires the exact modeled object identities and tables")
    blocking = set(prediction.hard_reasons) - LEGACY_SWEEP_REASONS
    actuation_safe = blocking.isdisjoint({"safety-monitor-veto", "ik-joint-safety-veto"})
    # Check exact oriented table boxes so base roll/pitch cannot inflate the tabletop AABB.
    table_paths = []
    for table in table_boxes:
        x, y, z, w = table.pose.orientation
        points = [
            rotate(
                (-x, -y, -z, w),
                state.robot.ee_pose.position.as_array() - table.pose.position.as_array(),
            )
            for state in prediction.states
        ]
        half = table.dimensions.as_array() / 2
        table_paths.append(
            tuple(
                segment_box_clearance(start, end, -half, half, TOOL_RADIUS_M)
                for start, end in zip(points[:-1], points[1:], strict=True)
            )
        )
    return ProxyEvidence(
        joint_limits_pass and actuation_safe,
        ik_pass and blocking.isdisjoint({"ik-unreachable", "ik-joint-safety-veto"}),
        tuple(min(clearances) for clearances in zip(*table_paths, strict=True)),
        tool_envelope_clearances(
            current,
            chunk,
            prediction,
            tool_radius_m=TOOL_RADIUS_M,
            scene_complete=True,
        ),
    )


def assess_proxy(
    current: WorldState,
    chunk: ActionChunk,
    prediction: Trajectory,
    evidence: ProxyEvidence,
    *,
    placement_state_index: int,
    placement_tool_goal_base: Vec3,
    placement_tolerance_m: float,
) -> CandidateAssessment:
    """Score aligned predictions and evaluate only the owner-declared proxy hard checks."""
    count = len(chunk.actions)
    states = prediction.states
    if (
        count != 50
        or len(states) != count + 1
        or prediction.dt_s != chunk.dt
        or states[0].robot.ee_pose != current.robot.ee_pose
        or not 1 <= placement_state_index <= count
        or not math.isfinite(placement_tolerance_m)
        or placement_tolerance_m <= 0
    ):
        raise MissingObjectiveInput("proxy requires complete aligned H=50 placement evidence")
    path = [state.robot.ee_pose.position.as_array() for state in states]
    goal = placement_tool_goal_base.as_array()
    jg = math.dist(path[placement_state_index], goal)
    jl = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:], strict=True))
    if not math.isfinite(jg) or not math.isfinite(jl):
        raise MissingObjectiveInput("nonfinite tool path or goal")
    table, objects = evidence.table_clearance_m, evidence.object_clearance_m
    if any(
        len(clearances) != count or not all(math.isfinite(c) for c in clearances)
        for clearances in (table, objects)
    ):
        raise MissingObjectiveInput("each clearance channel requires H=50 finite values")
    flags = (
        evidence.joint_limits_pass is True,
        evidence.ik_pass is True,
        min(table) > 0,
        min(objects) > 0,
        jg <= placement_tolerance_m,
    )
    return CandidateAssessment(
        tuple(zip(HARD_CHECKS, flags, strict=True)),
        (jg, jl, predicted_risk(objects, margin_m=PROXY_MARGIN_M), 0.0),
    )


def selections_by_policy(
    decisions: Iterable[Sequence[CandidateAssessment]],
    *,
    include_optional: bool = False,
) -> dict[str, tuple[int | None, ...]]:
    """Compare induced arm-5 selections on identical ordered decision pools."""
    grid = GRID | OPTIONAL_GRID if include_optional else GRID
    pools = [list(pool) for pool in decisions]
    if not pools:
        raise ValueError("at least one comparable decision pool required")
    return {
        name: tuple(
            RiskGate(weights=weights, maximize=False, required_constraints=HARD_CHECKS)
            .select(pool)
            .selected_index
            for pool in pools
        )
        for name, weights in grid.items()
    }


def select_policy(
    episode_ids: Sequence[str],
    outcomes: Mapping[str, Mapping[str, bool]],
) -> tuple[str, dict[str, float]]:
    """Evaluate approved observed success fraction with complete VALIDATION outcomes."""
    from vlarobot.eval.runner import build_episode_plan

    expected = {entry.scene.name for entry in build_episode_plan("validation", 0)}
    if len(episode_ids) != 18 or set(episode_ids) != expected:
        raise ValueError("expected the frozen 18 VALIDATION episode identities")
    if set(outcomes) not in (set(GRID), set(GRID | OPTIONAL_GRID)):
        raise ValueError("incomplete or unexpected policy grid")
    scores = {}
    for policy, episodes in outcomes.items():
        if set(episodes) != expected or any(type(value) is not bool for value in episodes.values()):
            raise ValueError(f"incomplete or unknown VALIDATION outcomes for {policy}")
        scores[policy] = sum(episodes.values()) / 18
    return min(scores, key=lambda policy: (-scores[policy], policy)), scores
