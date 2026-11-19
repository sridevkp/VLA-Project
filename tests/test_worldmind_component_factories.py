"""Focused tests for the parameterized Eq. 3.8 component factories (no Webots, no scoring).

Factories are validated at instantiation, delegate to the existing approved implementations, and
never convert ``MissingObjectiveInput`` into a default. Registration does not populate the
``components`` registry or make production ready.
"""

from __future__ import annotations

from dataclasses import MISSING
from types import SimpleNamespace

import pytest

from vlarobot.backends import worldmind_bindings as backend_bindings
from vlarobot.backends.worldmind_component_factories import (
    COMPONENT_FACTORY_BINDINGS,
    CONSISTENCY_FACTORY_BINDING,
    GOAL_FACTORY_BINDING,
    LENGTH_FACTORY_BINDING,
    RISK_FACTORY_BINDING,
    ConsistencyComponentParameters,
    GoalComponentParameters,
    LengthComponentParameters,
    RiskComponentParameters,
    consistency_component_factory,
    goal_component_factory,
    length_component_factory,
    risk_component_factory,
)
from vlarobot.core.types import (
    ActionChunk,
    EEDelta,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_config import ComponentSpec, validate_worldmind_bindings
from vlarobot.eval.worldmind_objectives import (
    MissingObjectiveInput,
    goal_error,
    prediction_consistency,
    soft_geometric_risk,
    trajectory_cost,
)
from vlarobot.world_model.rollout import Trajectory

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _state(x: float, *, object_id: str = "CAN", object_x: float = 0.5) -> WorldState:
    robot = RobotState(
        joint_pos={"arm_1_joint": 0.0},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(x, 0.0, 0.5), IDENTITY),
        gripper_opening=0.045,
    )
    scene = SceneState(
        [
            ObjectState(
                object_id,
                "can",
                Pose3D(Vec3(object_x, 0.0, 0.5), IDENTITY),
                Vec3(0.08, 0.08, 0.12),
                1.0,
            )
        ]
    )
    return WorldState(scene, robot)


def _trajectory(*states: WorldState) -> Trajectory:
    return Trajectory(states, 0.032, (), (), ())


def _delta_chunk() -> ActionChunk:
    return ActionChunk([EEDelta(Vec3(0.2, 0.01, 0.0), Vec3(0.0, 0.0, 0.0), 0.045)], 0.032, "test")


def test_parameter_objects_require_every_value_and_validate_at_instantiation() -> None:
    for dataclass_type in (
        GoalComponentParameters,
        LengthComponentParameters,
        RiskComponentParameters,
        ConsistencyComponentParameters,
    ):
        for name, field in dataclass_type.__dataclass_fields__.items():
            assert field.default is MISSING, f"{dataclass_type.__name__}.{name}"

    with pytest.raises(MissingObjectiveInput, match="position_scale_m"):
        GoalComponentParameters(0.0, 0.5, lambda *_: None, lambda *_: None)
    with pytest.raises(MissingObjectiveInput, match="ee_pose_of"):
        GoalComponentParameters(0.1, 0.5, None, lambda *_: None)  # type: ignore[arg-type]
    with pytest.raises(MissingObjectiveInput, match="dt_s"):
        LengthComponentParameters(0.0, 0.1, 1.0, 0.0, lambda *_: None)
    with pytest.raises(MissingObjectiveInput, match="time_weight"):
        LengthComponentParameters(0.032, 0.1, 1.0, -1.0, lambda *_: None)
    with pytest.raises(MissingObjectiveInput, match="tool_radius_m"):
        RiskComponentParameters(0.0, 0.02, True)
    with pytest.raises(MissingObjectiveInput, match="scene_complete"):
        RiskComponentParameters(0.01, 0.02, False)
    with pytest.raises(MissingObjectiveInput, match="joint_scale_rad"):
        ConsistencyComponentParameters(0.1, 0.5, 0.0, 0.05)


@pytest.mark.parametrize(
    ("factory", "params"),
    [
        (
            goal_component_factory,
            GoalComponentParameters(0.1, 0.5, lambda *_: None, lambda *_: None),
        ),
        (
            length_component_factory,
            LengthComponentParameters(0.032, 0.1, 1.0, 0.0, lambda *_: None),
        ),
        (risk_component_factory, RiskComponentParameters(0.01, 0.02, True)),
        (
            consistency_component_factory,
            ConsistencyComponentParameters(0.1, 0.5, 0.1, 0.05),
        ),
    ],
)
def test_factories_require_validated_parameter_objects(factory, params) -> None:
    assert callable(factory(params))
    with pytest.raises(MissingObjectiveInput, match="validated parameters"):
        factory({})  # type: ignore[arg-type]


def test_goal_component_delegates_and_fails_closed_without_prediction() -> None:
    current = _state(0.0)
    goal = Pose3D(Vec3(0.5, 0.0, 0.5), IDENTITY)
    predicted = _state(0.2)
    trajectory = _trajectory(current, predicted)
    params = GoalComponentParameters(
        0.1,
        0.5,
        lambda _s, _c, traj: traj.states[-1].robot.ee_pose,
        lambda *_: goal,
    )
    component = goal_component_factory(params)
    assert component(current, _delta_chunk(), trajectory) == pytest.approx(
        goal_error(predicted.robot.ee_pose, goal, position_scale_m=0.1, orientation_scale_rad=0.5)
    )
    with pytest.raises(MissingObjectiveInput, match="nonempty predicted Trajectory"):
        component(current, _delta_chunk(), None)


def test_length_component_delegates_and_fails_closed_without_prediction() -> None:
    current = _state(0.0)
    trajectory = _trajectory(current, _state(0.2))
    params = LengthComponentParameters(
        0.032,
        0.1,
        1.0,
        0.5,
        lambda _s, _c, traj: [state.robot.ee_pose.position for state in traj.states],
    )
    component = length_component_factory(params)
    positions = [state.robot.ee_pose.position for state in trajectory.states]
    assert component(current, _delta_chunk(), trajectory) == pytest.approx(
        trajectory_cost(positions, dt_s=0.032, path_scale_m=0.1, time_scale_s=1.0, time_weight=0.5)
    )
    with pytest.raises(MissingObjectiveInput, match="nonempty predicted Trajectory"):
        component(current, _delta_chunk(), None)


def test_risk_component_propagates_missing_objective_input() -> None:
    current = _state(0.0)
    chunk = _delta_chunk()
    trajectory = _trajectory(current, _state(0.19))
    component = risk_component_factory(RiskComponentParameters(0.01, 0.02, True))
    assert component(current, chunk, trajectory) == pytest.approx(
        soft_geometric_risk(
            current, chunk, trajectory, tool_radius_m=0.01, margin_m=0.02, scene_complete=True
        )
    )
    empty_scene = WorldState(SceneState([]), current.robot)
    with pytest.raises(MissingObjectiveInput):
        component(empty_scene, chunk, trajectory)


def test_consistency_component_propagates_missing_objective_input() -> None:
    current = _state(0.0)
    chunk = _delta_chunk()
    trajectory = _trajectory(current, _state(0.19))
    component = consistency_component_factory(ConsistencyComponentParameters(0.1, 0.5, 0.1, 0.05))
    assert component(current, chunk, trajectory) == pytest.approx(
        prediction_consistency(
            current,
            chunk,
            trajectory,
            position_scale_m=0.1,
            orientation_scale_rad=0.5,
            joint_scale_rad=0.1,
            gripper_scale_m=0.05,
        )
    )
    with pytest.raises(MissingObjectiveInput, match="per candidate action"):
        component(current, chunk, _trajectory(current))


def test_production_registers_factories_but_components_stay_empty() -> None:
    bindings = backend_bindings.production_worldmind_bindings()
    assert set(bindings.component_factories) == set(COMPONENT_FACTORY_BINDINGS)
    assert {
        GOAL_FACTORY_BINDING,
        LENGTH_FACTORY_BINDING,
        RISK_FACTORY_BINDING,
        CONSISTENCY_FACTORY_BINDING,
    }.issubset(bindings.component_factories)
    assert all(callable(factory) for factory in bindings.component_factories.values())
    assert bindings.components == {}
    assert bindings.calibration_evaluators == {}
    assert bindings.waypoint_sets == {}

    # A ready config naming a component still fails closed: factories are not components.
    config = SimpleNamespace(
        components={"Jg": ComponentSpec("approved-jg-v1", "dimensionless")},
        assessment=SimpleNamespace(hard_checks=()),
        calibration=SimpleNamespace(evaluator_binding="calibration"),
        surrogate=SimpleNamespace(binding="surrogate", inputs={"forward": "forward"}),
        safety=SimpleNamespace(binding="safety", inputs={"monitor": "monitor"}),
        verification=SimpleNamespace(binding="verification"),
        adapter=SimpleNamespace(
            pose_solver_binding="solver",
            waypoints_binding="waypoints",
            alternate_waypoints_binding=None,
        ),
    )
    with pytest.raises(ValueError, match="unregistered WorldMind bindings") as err:
        validate_worldmind_bindings(config, bindings)  # type: ignore[arg-type]
    assert "components.Jg=approved-jg-v1" in str(err.value)
