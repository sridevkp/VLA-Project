"""Offline fixture coverage. No calibration artifacts or simulator evaluation are created."""

from __future__ import annotations

import hashlib
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
from vlarobot.eval.worldmind_offline_evidence import evidence_digest, read_evidence
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.execution.worldmind_offline_acquisition import acquire_fixture_decision
from vlarobot.execution.worldmind_pipeline import WorldMindEpisodePolicy, _world_state  # noqa: PLC2701
from vlarobot.execution.worldmind_world_model_comparison import (
    FixtureRanking,
    build_offline_world_model_comparison,
)
from vlarobot.risk.gate import CandidateAssessment, RiskGate
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


@pytest.mark.parametrize(
    ("veto", "fault"),
    [
        (True, None),
        (False, None),
        (False, "apply"),
        (False, "step"),
        (False, "observation"),
        (False, "missing_step_reply"),
        (False, "initial"),
        (False, "prediction"),
        (False, "generation"),
        (False, "assessment"),
        (False, "no_rank"),
    ],
)
def test_fixture_uses_approved_factories_and_frozen_pool_without_riskgate_select(
    config, monkeypatch: pytest.MonkeyPatch, tmp_path, veto: bool, fault: str | None
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
    observation.robot.base_pose_world = Pose3D(Vec3(0.0, 0.0, 0.2), (0.6, 0.0, 0.0, 0.8))
    observation.robot.gripper_holding = None
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

    episode_id = "validation-clear-i00-s0"
    observation = backend.reset(ScenarioSpec(name=episode_id), seed=0)
    observation.robot.base_pose_world = Pose3D(Vec3(0.0, 0.0, 0.2), (0.6, 0.0, 0.0, 0.8))
    observation.robot.gripper_holding = None
    events.clear()
    backend.applies.clear()
    monitor.veto_live = False

    def fail(*args, **kwargs):
        raise RuntimeError("injected fixture fault")

    if fault in {"apply", "step"}:
        monkeypatch.setattr(backend, fault, fail)
    elif fault == "observation":
        normal_ground_truth = backend.get_ground_truth

        def fail_after_step():
            if backend.applies:
                fail()
            return normal_ground_truth()

        monkeypatch.setattr(backend, "get_ground_truth", fail_after_step)
    elif fault == "missing_step_reply":
        monkeypatch.setattr(backend, "step", lambda n: None)
    elif fault == "initial":
        observation.robot.base_pose_world = None
    elif fault == "prediction":
        normal_prediction = result.fixture.planner.predict

        def injected_prediction(current, chunk):
            if events.count("predict") == 1:
                fail()
            return normal_prediction(current, chunk)

        monkeypatch.setattr(result.fixture.planner, "predict", injected_prediction)
    elif fault == "generation":
        monkeypatch.setitem(
            result.fixture.bindings.waypoint_factories, config.adapter.waypoints_binding, fail
        )
    elif fault == "assessment":
        monkeypatch.setattr(result.fixture.planner, "assess", fail)
    elif fault == "no_rank":
        monkeypatch.setattr(
            result.fixture.planner,
            "assess",
            lambda *args: CandidateAssessment((), (float("nan"),) * 4),
        )

    goal = state.scene.objects[0].pose
    hashes = {
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "plan": "013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2",
        "definitions": result.provenance["objective_definitions_sha256"],
    }
    path = tmp_path / "capture.msgpack"
    with pytest.raises(ValueError, match="requires the unapproved"):
        acquire_fixture_decision(
            path,
            replace(result.fixture, provenance={**result.provenance, "arm": "vla_risk_veto"}),
            backend,
            observation,
            episode_id=episode_id,
            seed=0,
            goal_base=goal,
            source_hashes=hashes,
        )
    assert not path.exists()
    acquired = acquire_fixture_decision(
        path,
        result.fixture,
        backend,
        observation,
        episode_id=episode_id,
        seed=0,
        goal_base=goal,
        source_hashes=hashes,
    )
    archive = read_evidence(path, expected_digest=acquired.evidence.digest)
    assert evidence_digest(archive.records) == evidence_digest(acquired.evidence.records)
    saved = archive.records
    assert saved["arm"] == "vla_world_model"
    assert saved["provenance"] == dict(result.provenance)
    assert saved["policy"] == "adr0083_fixture_eq38_motor_only_execute_first"
    assert saved["recovery"] == []
    assert saved["draw_seeds"] == list(decision.generation.draw_seeds)
    assert not archive.ranking_evidence_complete

    if fault in {"initial", "generation", "prediction", "assessment", "no_rank"}:
        assert not backend.applies
        assert saved["execution"] == []
        assert saved["episode_outcome"]["status"] == "failed"
        assert saved["pairing"]["valid_steps"] == 0
        if fault == "initial":
            assert saved["input"] is None
            assert "invalid_input" in saved["failures"]
        elif fault == "generation":
            assert not archive.pool_complete
            assert "failed_generation" in saved["failures"]
        elif fault == "prediction":
            assert archive.pool_complete
            assert len(saved["candidates"][0]["payload"]["states"]) == 51
            assert saved["candidates"][1]["rollout"] == "failed_rollout"
        else:
            assert archive.pool_complete and saved["ranking"] is None
            assert len(saved["candidates"][0]["payload"]["states"]) == 51
            if fault == "no_rank":
                assert saved["planner_selection"]["reason"] == "no finite prediction score"
                assert acquired.selection.index is None
        return

    assert archive.pool_complete
    captured_state = saved["input"]["state"]
    assert captured_state.robot == observation.robot
    assert captured_state.robot.gripper_holding is None
    assert saved["input"]["goal"] == goal
    assert [c["id"] for c in saved["candidates"]] == list(decision.candidate_ids)
    assert all(c["payload"]["states"][0] == captured_state for c in saved["candidates"])
    assert all(len(c["payload"]["states"]) == 51 for c in saved["candidates"])
    assert acquired.selection.index == decision.selection.index
    assert acquired.selection.score == decision.selection.score
    assert saved["selected"] == decision.selected_candidate_id
    assert [c["payload"]["chunk"] for c in saved["candidates"]] == list(decision.candidates)
    execution = saved["execution"][0]
    assert execution["action"] == acquired.selection.chunk.actions[0]
    assert execution["motor"].accepted is not veto
    if fault == "apply":
        assert (execution["apply"], execution["step"]) == ("unknown", "not_executed")
    elif fault in {"step", "missing_step_reply"}:
        assert (execution["apply"], execution["step"]) == ("accepted", "unknown")
    elif fault == "observation":
        assert (execution["apply"], execution["step"]) == ("accepted", "confirmed")
    elif veto:
        assert (execution["apply"], execution["step"]) == ("not_attempted", "not_executed")
    else:
        assert (execution["apply"], execution["step"]) == ("accepted", "confirmed")
    assert saved["pairing"]["valid_steps"] == (1 if fault is None and not veto else 0)
    assert (execution["after"] is None) == (veto or fault is not None)


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
