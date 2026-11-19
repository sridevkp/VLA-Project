"""Parameterized factories for the owner-approved Eq. 3.8 component implementations.

This module supplies **no numeric value**. Each factory requires an explicit, validated parameter
object (every field mandatory, no defaults) and returns the pipeline-signature callable
``(WorldState, chunk, Trajectory | None) -> float``. The wrapped formulas are the existing approved
implementations in ``vlarobot.eval.worldmind_objectives``; their ``MissingObjectiveInput`` is never
caught or converted to a default value.

The factories are inert: no backend, Webots process, IK solve, inference, scoring, calibration
candidate or split access is performed here. They are registered in the existing
``WorldMindBindings.component_factories`` group, while ``components`` itself stays empty and the
production config stays ``blocked`` until the owner supplies reviewed parameters and evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vlarobot.core.types import Pose3D, Vec3, WorldState
from vlarobot.eval.worldmind_objectives import (
    MissingObjectiveInput,
    goal_error,
    prediction_consistency,
    soft_geometric_risk,
    trajectory_cost,
)
from vlarobot.world_model.rollout import Trajectory

ComponentCallable = Callable[[WorldState, Any, Trajectory | None], float]
EePoseProvider = Callable[[WorldState, Any, Trajectory], Pose3D]
GoalPoseProvider = Callable[[WorldState, Any, Trajectory], Pose3D]
ToolPositionsProvider = Callable[[WorldState, Any, Trajectory], Sequence[Vec3]]


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissingObjectiveInput(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MissingObjectiveInput(f"{name} must be a finite positive number")
    return number


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissingObjectiveInput(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise MissingObjectiveInput(f"{name} must be a finite nonnegative number")
    return number


def _provider(value: Any, name: str) -> Callable[..., Any]:
    if not callable(value):
        raise MissingObjectiveInput(f"{name} must be an explicit callable provider")
    return value


def _trajectory(trajectory: Trajectory | None, name: str) -> Trajectory:
    """Reject an absent prediction instead of inventing one (fail closed)."""
    if not isinstance(trajectory, Trajectory) or not trajectory.states:
        raise MissingObjectiveInput(f"{name} requires a nonempty predicted Trajectory")
    return trajectory


@dataclass(frozen=True)
class GoalComponentParameters:
    """All Jg inputs are owner-supplied; the ee/goal extractors are explicit, never inferred."""

    position_scale_m: float
    orientation_scale_rad: float
    ee_pose_of: EePoseProvider
    goal_pose_of: GoalPoseProvider

    def __post_init__(self) -> None:
        _positive(self.position_scale_m, "position_scale_m")
        _positive(self.orientation_scale_rad, "orientation_scale_rad")
        _provider(self.ee_pose_of, "ee_pose_of")
        _provider(self.goal_pose_of, "goal_pose_of")


@dataclass(frozen=True)
class LengthComponentParameters:
    """All Jl inputs are owner-supplied; the tool-path extractor is explicit, never inferred."""

    dt_s: float
    path_scale_m: float
    time_scale_s: float
    time_weight: float
    tool_positions_of: ToolPositionsProvider

    def __post_init__(self) -> None:
        _positive(self.dt_s, "dt_s")
        _positive(self.path_scale_m, "path_scale_m")
        _positive(self.time_scale_s, "time_scale_s")
        _nonnegative(self.time_weight, "time_weight")
        _provider(self.tool_positions_of, "tool_positions_of")


@dataclass(frozen=True)
class RiskComponentParameters:
    """Jr soft proxy inputs; no full-arm certificate is implied and no margin is defaulted."""

    tool_radius_m: float
    margin_m: float
    scene_complete: bool

    def __post_init__(self) -> None:
        _positive(self.tool_radius_m, "tool_radius_m")
        _positive(self.margin_m, "margin_m")
        if self.scene_complete is not True:
            raise MissingObjectiveInput("Jr requires scene_complete=true; fail closed otherwise")


@dataclass(frozen=True)
class ConsistencyComponentParameters:
    """Ju command-following scales; all four are owner-supplied and validated."""

    position_scale_m: float
    orientation_scale_rad: float
    joint_scale_rad: float
    gripper_scale_m: float

    def __post_init__(self) -> None:
        _positive(self.position_scale_m, "position_scale_m")
        _positive(self.orientation_scale_rad, "orientation_scale_rad")
        _positive(self.joint_scale_rad, "joint_scale_rad")
        _positive(self.gripper_scale_m, "gripper_scale_m")


def goal_component_factory(parameters: GoalComponentParameters) -> ComponentCallable:
    """Bind the approved Jg formula to validated parameters; return a pipeline component."""
    if not isinstance(parameters, GoalComponentParameters):
        raise MissingObjectiveInput("goal_component_factory requires validated parameters")

    def component(state: WorldState, chunk: Any, trajectory: Trajectory | None) -> float:
        predicted = _trajectory(trajectory, "Jg")
        ee_pose = parameters.ee_pose_of(state, chunk, predicted)
        goal_pose = parameters.goal_pose_of(state, chunk, predicted)
        return goal_error(
            ee_pose,
            goal_pose,
            position_scale_m=parameters.position_scale_m,
            orientation_scale_rad=parameters.orientation_scale_rad,
        )

    return component


def length_component_factory(parameters: LengthComponentParameters) -> ComponentCallable:
    """Bind the approved Jl formula to validated parameters; return a pipeline component."""
    if not isinstance(parameters, LengthComponentParameters):
        raise MissingObjectiveInput("length_component_factory requires validated parameters")

    def component(state: WorldState, chunk: Any, trajectory: Trajectory | None) -> float:
        predicted = _trajectory(trajectory, "Jl")
        positions = parameters.tool_positions_of(state, chunk, predicted)
        return trajectory_cost(
            positions,
            dt_s=parameters.dt_s,
            path_scale_m=parameters.path_scale_m,
            time_scale_s=parameters.time_scale_s,
            time_weight=parameters.time_weight,
        )

    return component


def risk_component_factory(parameters: RiskComponentParameters) -> ComponentCallable:
    """Bind the approved Jr soft proxy to validated parameters; return a pipeline component."""
    if not isinstance(parameters, RiskComponentParameters):
        raise MissingObjectiveInput("risk_component_factory requires validated parameters")

    def component(state: WorldState, chunk: Any, trajectory: Trajectory | None) -> float:
        return soft_geometric_risk(
            state,
            chunk,
            trajectory,  # type: ignore[arg-type]
            tool_radius_m=parameters.tool_radius_m,
            margin_m=parameters.margin_m,
            scene_complete=parameters.scene_complete,
        )

    return component


def consistency_component_factory(
    parameters: ConsistencyComponentParameters,
) -> ComponentCallable:
    """Bind the approved Ju formula to validated parameters; return a pipeline component."""
    if not isinstance(parameters, ConsistencyComponentParameters):
        raise MissingObjectiveInput("consistency_component_factory requires validated parameters")

    def component(state: WorldState, chunk: Any, trajectory: Trajectory | None) -> float:
        return prediction_consistency(
            state,
            chunk,
            trajectory,  # type: ignore[arg-type]
            position_scale_m=parameters.position_scale_m,
            orientation_scale_rad=parameters.orientation_scale_rad,
            joint_scale_rad=parameters.joint_scale_rad,
            gripper_scale_m=parameters.gripper_scale_m,
        )

    return component


# Stable identities for the FACTORIES, not approved component binding identities. The owner still
# selects which identity each config component binds to; nothing here is registered in `components`.
GOAL_FACTORY_BINDING = "goal_error_normalized_v1_factory"
LENGTH_FACTORY_BINDING = "trajectory_cost_normalized_v1_factory"
RISK_FACTORY_BINDING = "soft_geometric_risk_v1_factory"
CONSISTENCY_FACTORY_BINDING = "prediction_consistency_v1_factory"

__all__ = [
    "COMPONENT_FACTORY_BINDINGS",
    "CONSISTENCY_FACTORY_BINDING",
    "ConsistencyComponentParameters",
    "GOAL_FACTORY_BINDING",
    "GoalComponentParameters",
    "LENGTH_FACTORY_BINDING",
    "LengthComponentParameters",
    "RISK_FACTORY_BINDING",
    "RiskComponentParameters",
    "consistency_component_factory",
    "goal_component_factory",
    "length_component_factory",
    "risk_component_factory",
]

COMPONENT_FACTORY_BINDINGS: dict[str, Callable[[Any], ComponentCallable]] = {
    GOAL_FACTORY_BINDING: goal_component_factory,
    LENGTH_FACTORY_BINDING: length_component_factory,
    RISK_FACTORY_BINDING: risk_component_factory,
    CONSISTENCY_FACTORY_BINDING: consistency_component_factory,
}
