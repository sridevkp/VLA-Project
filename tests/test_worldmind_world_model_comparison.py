"""Offline fixture coverage. No calibration artifacts or simulator evaluation are created."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.worldmind_bindings import production_worldmind_bindings
from vlarobot.backends.worldmind_component_factories import (
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
    Action,
    ApplyResult,
    Pose3D,
    RobotState,
    ScenarioSpec,
    TaskSpec,
    Vec3,
)
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.execution.worldmind_pipeline import WorldMindEpisodePolicy, _world_state  # noqa: PLC2701
from vlarobot.execution.worldmind_world_model_comparison import (
    FixtureRanking,
    build_offline_world_model_comparison,
)
from vlarobot.risk.gate import RiskGate
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/worldmind_diagnostic.approved.yaml"
JOINT = "arm_7_joint"


@pytest.fixture
def config():
    return load_worldmind_diagnostic_config(CONFIG_PATH)


class _CountingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.applies: list[Action] = []

    def apply(self, action: Action) -> ApplyResult:
        self.applies.append(action)
        return super().apply(action)


@pytest.mark.parametrize("veto", [False, True])
def test_fixture_uses_approved_factories_and_frozen_pool_without_riskgate_select(
    config, monkeypatch: pytest.MonkeyPatch, veto: bool
) -> None:
    events: list[str] = []

    class Monitor(SafetyMonitor):
        veto_live = False

        def validate(self, action: Action, state: RobotState) -> ApplyResult:
            events.append("validate")
            if self.veto_live:
                return ApplyResult(False, "fixture motor veto")
            return super().validate(action, state)

    class Predictor(AnalyticSurrogate):
        def predict_trajectory(self, state, chunk, **kwargs):
            events.append("predict")
            result = super().predict_trajectory(state, chunk, **kwargs)
            assert "unverified-arm-sweep" in result.hard_reasons
            if events.count("predict") == 32:
                monitor.veto_live = veto
            return result

    def forbidden_gate(*args, **kwargs):
        pytest.fail("RiskGate.select must not be invoked for this comparison arm")

    monkeypatch.setattr(RiskGate, "select", forbidden_gate)
    original = production_worldmind_bindings()
    safety = original.safety[config.safety.binding](config, original.inputs["safety"])
    monitor = Monitor(safety.limits)
    backend = _CountingBackend()
    scenario = ScenarioSpec(name="offline-world-model-fixture")
    observation = backend.reset(scenario, seed=0)
    state = _world_state(observation, backend.get_ground_truth(), config, target=config.task.target)

    def forward(q):
        return Pose3D(Vec3(0.5 + q[JOINT], 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))

    predictor = Predictor(
        VelocityLimitedKinematics({JOINT: 1.0}, forward),
        inverse=lambda pose, robot: None,
        gripper_speed_m_s=0.03,
        gravity_m_s2=9.81,
        support_z_m={config.task.target: 0.4},
        validate=lambda action, robot: safety.validate(action, robot).accepted,
    )
    waypoints = (ScriptedWaypoint("FIXTURE", 50, joints={JOINT: 0.02}),)
    bindings = replace(
        original,
        surrogates={config.surrogate.binding: lambda config, inputs: predictor},
        safety={config.safety.binding: lambda config, inputs: monitor},
        hard_checks={name: forbidden_gate for name in config.assessment.hard_checks},
        waypoint_factories={
            config.adapter.waypoints_binding: lambda config, state, task: waypoints
        },
    )
    task = TaskSpec(
        instruction=config.task.instruction,
        target=config.task.target,
        destination=config.task.destination,
    )
    # Compare against the existing production generator without executing its episode policy.
    production = SimpleNamespace(config=config, bindings=bindings)
    expected_generator = WorldMindEpisodePolicy._generator(production, waypoints, 0)
    expected = expected_generator.propose(task, state, None)
    events.clear()

    # Purely illustrative inputs. These are not calibrated weights, scales or margins.
    ranking = FixtureRanking(
        components={
            "Jg": goal_component_factory(
                GoalComponentParameters(
                    1.0,
                    1.0,
                    lambda state, chunk, trajectory: trajectory.states[-1].robot.ee_pose,
                    lambda state, chunk, trajectory: state.scene.objects[0].pose,
                )
            ),
            "Jl": length_component_factory(
                LengthComponentParameters(
                    0.032,
                    1.0,
                    1.0,
                    1.0,
                    lambda state, chunk, trajectory: [
                        node.robot.ee_pose.position for node in trajectory.states
                    ],
                )
            ),
            "Jr": risk_component_factory(RiskComponentParameters(0.01, 0.01, True)),
            "Ju": consistency_component_factory(ConsistencyComponentParameters(1.0, 1.0, 1.0, 1.0)),
        },
        weights=(1.0, 1.0, 1.0, 1.0),
        maximize=False,
    )
    result = build_offline_world_model_comparison(config, bindings, fixture_ranking=ranking)
    assert result.fixture is not None
    decision = result.fixture.execute_fixture_tick(
        backend, observation, episode_id=scenario.name, seed=0, decision_index=1
    )

    assert decision.generation.proposer_source == "scripted"
    assert decision.generation.decision_id.endswith("-d01")
    assert decision.generation.draw_seed == expected_generator.draw_log[0]["seed"]
    assert decision.generation.draw_seeds == expected_generator.draw_log[0]["draw_seeds"]
    assert decision.candidate_ids == tuple(f"candidate-{i:02d}" for i in range(32))
    assert all(
        len(chunk.actions) == 50 and chunk.source == "scripted" for chunk in decision.candidates
    )
    assert [asdict(chunk) for chunk in decision.candidates] == [asdict(chunk) for chunk in expected]
    assert decision.selection.index is not None
    assert decision.selected_candidate_id == decision.candidate_ids[decision.selection.index]
    assert decision.selection.prediction_count == events.count("predict") == 32
    assert decision.selection.veto_count == 0
    assert np.isfinite(decision.selection.score)
    assert events[-1] == "validate"
    assert decision.apply_result.accepted is not veto
    assert backend.applies == ([] if veto else [decision.selection.chunk.actions[0]])
    assert "prediction_pairing" not in asdict(decision)
    assert decision.provenance["arm"] == "vla_world_model"
    assert decision.provenance["execution_scope"] == "fixture_only"
    assert decision.provenance["scientific_execution_approved"] == "false"
    assert decision.provenance["calibration_status"] == "fixture_inputs_only"
    assert "calibration_id" not in decision.provenance
    with pytest.raises(TypeError, match="FakeBackend"):
        result.fixture.execute_fixture_tick(object(), observation, episode_id=scenario.name, seed=0)


def test_missing_calibration_keeps_approved_definitions_non_executable(config, tmp_path):
    bindings = production_worldmind_bindings()
    result = build_offline_world_model_comparison(config, bindings)
    assert result.fixture is None
    assert result.blockers == ("approved calibration missing",)
    assert result.provenance["execution_scope"] == "non_executable"
    assert result.provenance["objective_definitions_sha256"] == (
        "457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14"
    )
    with pytest.raises((FileNotFoundError, ValueError), match="missing|not found"):
        build_offline_world_model_comparison(
            config, bindings, calibration_candidate=tmp_path / "missing.json"
        )


def test_missing_scientific_definitions_fail_before_fixture_composition(config, tmp_path):
    absent = replace(config, report=replace(config.report, definitions=tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        build_offline_world_model_comparison(absent, production_worldmind_bindings())
