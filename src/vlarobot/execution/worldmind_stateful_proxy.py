"""Phase-aware arm-5 placement, prediction and ranking for held-waypoint chunks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from vlarobot.core.types import Action, ActionChunk, Decision, JointTarget, WorldState
from vlarobot.eval.worldmind_config import FROZEN_CONTROL_DT_S, FROZEN_HORIZON_STEPS
from vlarobot.eval.worldmind_objectives import predicted_risk
from vlarobot.eval.worldmind_proxy_policy import GRID, HARD_CHECKS, PROXY_MARGIN_M
from vlarobot.execution.worldmind_proxy_binding import _matched_proxy_evidence  # noqa: PLC2701
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.rollout import Trajectory

_PHASES = frozenset(
    (
        "INIT",
        "PREGRASP",
        "DESCEND",
        "CLOSE",
        "LIFT",
        "TRANSIT",
        "PLACE_DESCEND",
        "RELEASE",
        "RETREAT",
        "HOME",
    )
)


@dataclass(frozen=True)
class Waypoint:
    phase: str
    terminal: bool
    settle_rad: float | None
    target_xyz_m: tuple[float, float, float] | None

    @property
    def placement_required(self) -> bool:
        return self.phase == "PLACE_DESCEND" and self.terminal


@dataclass(frozen=True)
class PlacementBinding:
    status: str
    reason: str
    state_index: int | None
    error_m: float | None
    accepted: bool | None


def predicted_completion(
    action: Action, states: Sequence[WorldState], waypoint: Waypoint
) -> int | None:
    """First future state satisfying the live expert's strict joint settle predicate."""
    if not isinstance(action, JointTarget) or waypoint.settle_rad is None:
        return None
    return next(
        (
            index
            for index, state in enumerate(states[1:], 1)
            if all(
                abs(state.robot.joint_pos.get(name, math.inf) - target) < waypoint.settle_rad
                for name, target in action.q.items()
            )
        ),
        None,
    )


def bind_placement(
    waypoint: Waypoint,
    completion: int | None,
    states: Sequence[WorldState],
    goal: Sequence[float],
    tolerance_m: float,
) -> PlacementBinding:
    if not waypoint.placement_required:
        return PlacementBinding(
            "not_applicable", "chunk_has_no_placement_waypoint", None, None, None
        )
    if completion is None or not 0 < completion < len(states):
        return PlacementBinding(
            "unavailable", "required_placement_not_predicted", None, None, False
        )
    distance = math.dist(states[completion].robot.ee_pose.position.as_array(), goal)
    if not math.isfinite(distance):
        return PlacementBinding("invalid", "nonfinite_placement_state", completion, None, False)
    return PlacementBinding(
        "measured", "terminal_place_descend", completion, distance, distance <= tolerance_m
    )


def rank_phase_pool(
    current: WorldState,
    pool: Sequence[ActionChunk],
    waypoint: Waypoint,
    predict: Callable[[WorldState, ActionChunk], Trajectory],
    monitor: Any,
    geometry: Mapping[str, Any],
    goal: Sequence[float],
    placement_tolerance_m: float,
    ik_accepts: Callable[[Action], bool],
) -> tuple[Decision, list[dict[str, Any]]]:
    """Run production geometric checks and Eq. 3.8 with only same-phase applicable terms."""
    if waypoint.phase not in _PHASES:
        raise ValueError("unrecognized predicted waypoint phase")
    placement_required = waypoint.placement_required
    required = HARD_CHECKS if placement_required else HARD_CHECKS[:-1]
    gate = RiskGate(
        weights=GRID["equal"],
        maximize=False,
        required_constraints=required,
        inactive_components=() if placement_required else (0,),
    )
    assessments, records = [], []
    for chunk in pool:
        if (
            len(chunk.actions) != FROZEN_HORIZON_STEPS
            or chunk.dt != FROZEN_CONTROL_DT_S
            or any(action != chunk.actions[0] for action in chunk.actions)
        ):
            raise ValueError("phase prediction requires an aligned H=50 held-command chunk")
        action = chunk.actions[0]
        trajectory = predict(current, chunk)
        complete = (
            len(trajectory.states) == len(chunk.actions) + 1
            and trajectory.dt_s == chunk.dt
            and trajectory.states[0].robot.ee_pose == current.robot.ee_pose
        )
        completion = predicted_completion(action, trajectory.states, waypoint) if complete else None
        placement = bind_placement(
            waypoint, completion, trajectory.states, goal, placement_tolerance_m
        )
        consumed = completion or len(chunk.actions)
        checks: dict[str, bool | None] = {"prediction_complete": complete}
        components = (placement.error_m, None, None, 0.0)
        if complete:
            prefix = replace(chunk, actions=chunk.actions[:consumed])
            states = trajectory.states[: consumed + 1]
            evidence = _matched_proxy_evidence(
                current,
                prefix,
                replace(trajectory, states=states),
                monitor,
                geometry,
                active_phase=waypoint.phase,
                phase_terminal=waypoint.terminal and completion is not None,
            )
            table, objects = evidence.geometric_hard_pass
            checks.update(
                zip(
                    HARD_CHECKS[:4],
                    (
                        evidence.joint_limits_pass,
                        evidence.ik_pass and ik_accepts(action),
                        table,
                        objects,
                    ),
                    strict=True,
                )
            )
            path = [state.robot.ee_pose.position.as_array() for state in states]
            components = (
                placement.error_m,
                sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:], strict=True)),
                predicted_risk(evidence.object_clearance_m, margin_m=PROXY_MARGIN_M),
                0.0,
            )
        if placement_required:
            checks["placement_tolerance"] = placement.accepted
        assessments.append(CandidateAssessment(tuple(checks.items()), components))
        records.append(
            {
                "checks": checks,
                "components": components,
                "placement": asdict(placement),
                "predicted_phase": waypoint.phase,
                "predicted_states": len(trajectory.states),
                "active_prefix_ticks": consumed,
                "predicted_completion": completion,
                "admitted": all(
                    checks.get(name) is True for name in (*required, "prediction_complete")
                ),
                "prediction_reasons": trajectory.hard_reasons,
            }
        )
    return gate.select(assessments), records
