"""Inert FakeBackend tests for the production WorldMind composition.

No Webots process, real IK, neural inference, training, actuation, scored campaign or TEST
split is used. The report/calibration bindings are synthetic fixtures; production still fails
closed until reviewed authoritative values are registered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.fake.backend import CAN_WORLD_XYZ
from vlarobot.cli import app
from vlarobot.core.types import (
    Action,
    ApplyResult,
    GroundTruth,
    JointTarget,
    Observation,
    Pose3D,
    ScenarioSpec,
    SceneState,
    Vec3,
    VerifyResult,
    WorldState,
)
from vlarobot.eval import worldmind_config
from vlarobot.eval.runner import EvalEpisode, build_episode_plan
from vlarobot.eval.worldmind_config import (
    WorldMindBindings,
    WorldMindConfig,
    load_worldmind_config,
    plan_worldmind_validation,
    validate_worldmind_bindings,
)
from vlarobot.eval.worldmind_evidence import WorldMindEvidenceWriter
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import (
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.execution.worldmind_pipeline import run_worldmind_evaluation
from vlarobot.planning.verification import PlacementVerifier
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate

ARM_JOINTS = tuple(f"arm_{index}_joint" for index in range(1, 8))
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
TABLE51_NAMES = (
    "task_success_rate",
    "prediction_error",
    "collision_rate",
    "path_length",
    "execution_time",
    "planning_latency",
    "risk_rejection_rate",
    "reliability",
)


@dataclass
class StaticPoseSolver:
    """Explicit reviewed-style solver stub; it performs no IK and reports zero residual."""

    joints: dict[str, float]
    calls: list[Pose3D] = field(default_factory=list)
    position_error_m: float = 0.0
    orientation_error_rad: float = 0.0

    def solve(self, pose: Pose3D, state: Any) -> StaticPoseSolver:
        del state
        self.calls.append(pose)
        return self


@dataclass
class SharedState:
    """Mutable observation sinks shared by injected binding factories."""

    events: list[str] = field(default_factory=list)
    veto_execution: bool = False


class RecordingMonitor:
    """Delegate to the real SafetyMonitor and record live validation order."""

    def __init__(self, shared: SharedState) -> None:
        self.shared = shared
        self.inner = SafetyMonitor(
            SafetyLimits(
                joint_position={name: (-2.0, 2.0) for name in ARM_JOINTS},
                joint_velocity={name: 1.0 for name in ARM_JOINTS},
            )
        )

    def validate(self, action: Action, robot: Any) -> ApplyResult:
        self.shared.events.append("validate")
        if self.shared.veto_execution:
            return ApplyResult(False, "injected live safety veto")
        return self.inner.validate(action, robot)


@dataclass
class ScriptedVerifier:
    """Injected verifier with explicit, recorded outcomes for recovery tests."""

    results: list[VerifyResult]

    def check(self, task: Any, predicted: Any, observed: Any) -> VerifyResult:
        del task, predicted, observed
        if not self.results:
            raise AssertionError("scripted verifier was called more often than configured")
        return self.results.pop(0)

    def next_action(self, result: VerifyResult) -> str:
        return "complete" if result.success else "retry_once"


class SpyBackend:
    """Delegate to FakeBackend while recording the live control boundary."""

    def __init__(self, inner: FakeBackend, shared: SharedState | None = None) -> None:
        self.inner = inner
        self.shared = shared
        self.events: list[str] = []

    def _record(self, event: str) -> None:
        self.events.append(event)
        if self.shared is not None:
            self.shared.events.append(f"backend.{event}")

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        self._record("reset")
        return self.inner.reset(scenario, seed)

    def apply(self, action: Action) -> ApplyResult:
        self._record("apply")
        return self.inner.apply(action)

    def step(self, n_steps: int = 1) -> Observation:
        self._record("step")
        return self.inner.step(n_steps)

    def get_ground_truth(self) -> GroundTruth:
        return self.inner.get_ground_truth()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class WorldMindFixture:
    """Synthetic config, reviewed-style bindings and observation sinks for one test."""

    root: Path
    config: WorldMindConfig
    bindings: WorldMindBindings
    shared: SharedState
    solver: StaticPoseSolver
    report_sha: str
    config_path: Path


def _write_fixture_files(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    pdf = root / "synthetic-report.bin"
    pdf.write_bytes(b"WorldMind inert fixture: synthetic report bytes, not Phase I.\n")
    report_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(worldmind_config, "AUTHORITATIVE_REPORT_SHA256", report_sha)

    definitions = root / "definitions.json"
    definitions.write_text(
        json.dumps(
            {
                "report_sha256": report_sha,
                "components": {
                    name: f"synthetic {name} definition" for name in ("Jg", "Jl", "Jr", "Ju")
                },
                "table51": {
                    name: {"definition": f"synthetic {name} definition", "unit": "fixture"}
                    for name in TABLE51_NAMES
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    calibration = root / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_id": "synthetic-calibration-v1",
                "split": "validation",
                "approved": True,
                "calibration_id": "synthetic-calibration-v1",
                "source": "synthetic inert fixture; not approved VALIDATION evidence",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    safety_path = root / "safety-config.yaml"
    safety_path.write_text("{}\n", encoding="utf-8")
    return pdf, report_sha


def _make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hard_value: bool = True,
    verifier_mode: str = "placement",
    arm_veto: bool = False,
    alternate: bool = False,
) -> WorldMindFixture:
    root = tmp_path / "fixture"
    root.mkdir()
    pdf, report_sha = _write_fixture_files(root, monkeypatch)
    shared = SharedState()
    solver = StaticPoseSolver({name: 0.0 for name in ARM_JOINTS})
    primary_pose = Pose3D(Vec3(0.5, 0.0, 0.5), IDENTITY_QUAT)
    alternate_pose = Pose3D(Vec3(0.6, 0.0, 0.5), IDENTITY_QUAT)

    raw_config: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready",
        "split": "validation",
        "base_seed": 0,
        "seeds": [0, 1, 2],
        "episodes": 18,
        "backend": {
            "kind": "fake",
            "declaration": "fake://deterministic; state=oracle",
            "state_source": "oracle",
        },
        "task": {
            "instruction": "synthetic inert fixture",
            "target": "CAN",
            "destination": "destination",
        },
        "pipeline": {
            "proposer": "scripted_chunk_adapter",
            "arm": "full_system",
            "candidate_count": 32,
            "horizon_steps": 50,
            "control_dt_s": 0.032,
            "decision_seed_base": 0,
        },
        "bounds": {"max_decisions": 2, "max_steps": 2, "timings_enabled": True},
        "report": {
            "pdf": str(pdf),
            "definitions": str(root / "definitions.json"),
            "sha256": report_sha,
        },
        "components": {
            name: {"binding": name.lower(), "unit": "fixture"} for name in ("Jg", "Jl", "Jr", "Ju")
        },
        "assessment": {
            "weights": {"Jg": 1.0, "Jl": 0.0, "Jr": 0.0, "Ju": 0.0},
            "maximize": True,
            "required_constraints": ["synthetic_hard"],
            "hard_checks": ["synthetic_hard"],
        },
        "calibration": {
            "evidence_id": "synthetic-calibration-v1",
            "evidence_path": str(root / "calibration.json"),
        },
        "surrogate": {
            "binding": "surrogate_factory",
            "inputs": {"validate": "fixture_validate", "tracking": "fixture_tracking"},
        },
        "safety": {
            "binding": "safety_factory",
            "config_path": str(root / "safety-config.yaml"),
            "inputs": {"monitor": "fixture_monitor"},
        },
        "verification": {
            "binding": "placement_verifier",
            "destination_world": list(CAN_WORLD_XYZ),
            "rest_center_world": list(CAN_WORLD_XYZ),
            "object_dimensions_m": [0.08, 0.08, 0.12],
            "up_axis_base": [0.0, 0.0, 1.0],
            "distance_tolerance_m": 0.01,
            "height_tolerance_m": 0.01,
            "tilt_limit_rad": 0.1,
            "home_tolerance_rad": 0.01,
            "home_joints": {name: 0.0 for name in ARM_JOINTS},
        },
        "adapter": {
            "waypoints_binding": "primary_waypoints",
            "alternate_waypoints_binding": "alternate_waypoints" if alternate else None,
            "pose_solver_binding": "static_solver",
            "ik_position_tolerance_m": 0.01,
            "ik_orientation_tolerance_rad": 0.01,
            "jitter": {
                "joint_std_rad": 0.001,
                "joint_limit_rad": 0.005,
                "max_attempts": 10,
            },
        },
    }
    config_path = root / "worldmind.yaml"
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    config = load_worldmind_config(config_path)

    def component_factory(name: str) -> Any:
        def component(state: Any, chunk: Any, trajectory: Any) -> float:
            del state, chunk, trajectory
            return 1.0 if name == "Jg" else 0.0

        return component

    def hard_check(state: Any, chunk: Any, trajectory: Any) -> bool:
        del state, chunk, trajectory
        if arm_veto:
            shared.veto_execution = True
        return hard_value

    def safety_factory(loaded: WorldMindConfig, inputs: Any) -> RecordingMonitor:
        del loaded, inputs
        return RecordingMonitor(shared)

    def surrogate_factory(loaded: WorldMindConfig, inputs: Any) -> AnalyticSurrogate:
        del inputs
        tracking = VelocityLimitedKinematics(
            {name: 1.0 for name in ARM_JOINTS},
            lambda joints: Pose3D(Vec3(0.5, 0.0, 0.5), IDENTITY_QUAT),
        )
        return AnalyticSurrogate(
            tracking,
            inverse=lambda pose, robot: JointTarget(dict(robot.joint_pos)),
            gripper_speed_m_s=0.045,
            gravity_m_s2=9.81,
            support_z_m={"CAN": loaded.verification.destination_world[2]},
            validate=lambda action, robot: True,
            check_transition=lambda before, after: True,
        )

    def verification_factory(
        loaded: WorldMindConfig,
        destination_base: Pose3D,
        rest_center_base: Pose3D,
        inputs: Any,
    ) -> Any:
        del inputs
        if verifier_mode == "fail_then_success":
            return ScriptedVerifier(
                [
                    VerifyResult(False, "synthetic verification miss"),
                    VerifyResult(True, "synthetic verification pass"),
                ]
            )
        if verifier_mode != "placement":
            raise ValueError(f"unknown verifier mode: {verifier_mode}")
        return PlacementVerifier(
            destination_base=destination_base,
            rest_center_base=rest_center_base,
            up_axis_base=loaded.verification.up_axis_base,
            home_joints=loaded.verification.home_joints,
            distance_tolerance_m=loaded.verification.distance_tolerance_m,
            height_tolerance_m=loaded.verification.height_tolerance_m,
            tilt_limit_rad=loaded.verification.tilt_limit_rad,
            home_tolerance_rad=loaded.verification.home_tolerance_rad,
        )

    solver.position_error_m = 0.0
    solver.orientation_error_rad = 0.0
    bindings = WorldMindBindings(
        components={name.lower(): component_factory(name) for name in ("Jg", "Jl", "Jr", "Ju")},
        hard_checks={"synthetic_hard": hard_check},
        surrogates={"surrogate_factory": surrogate_factory},
        safety={"safety_factory": safety_factory},
        verification={"placement_verifier": verification_factory},
        pose_solvers={"static_solver": solver},
        waypoint_sets={
            "primary_waypoints": [
                ScriptedWaypoint("primary", 49, pose=primary_pose),
                ScriptedWaypoint("release", 1, gripper_opening=0.045),
            ],
            "alternate_waypoints": [
                ScriptedWaypoint("alternate", 49, pose=alternate_pose),
                ScriptedWaypoint("release", 1, gripper_opening=0.045),
            ],
        },
        inputs={
            "surrogate": {"validate": "fixture_validate", "tracking": "fixture_tracking"},
            "safety": {"monitor": "fixture_monitor"},
            "verification": {},
        },
    )
    validate_worldmind_bindings(config, bindings)
    return WorldMindFixture(root, config, bindings, shared, solver, report_sha, config_path)


def _one_episode_plan() -> tuple[EvalEpisode, ...]:
    return (
        EvalEpisode(
            index=0,
            scene=ScenarioSpec(name="validation-clear-i00-s0", metadata={"family": "clear"}),
            seed=0,
        ),
    )


def _run_one(
    fixture: WorldMindFixture,
    tmp_path: Path,
    *,
    name: str = "run",
    config: WorldMindConfig | None = None,
    bindings: WorldMindBindings | None = None,
    backend: SpyBackend | None = None,
    plan: tuple[EvalEpisode, ...] | None = None,
) -> tuple[Any, dict[str, Path], dict[str, Any], list[dict[str, Any]], SpyBackend]:
    out = tmp_path / name
    backend = backend or SpyBackend(FakeBackend(), fixture.shared)
    report, sidecars = run_worldmind_evaluation(
        plan or _one_episode_plan(),
        backend,
        config or fixture.config,
        out,
        bindings or fixture.bindings,
    )
    decisions = [
        json.loads(line)
        for line in sidecars["decisions"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    pairings = [
        json.loads(line)
        for line in sidecars["prediction_pairing"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    return report, sidecars, {"decisions": decisions, "pairings": pairings}, backend


def test_scripted_adapter_is_seeded_bounded_and_scripted_only(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    config = fixture.config
    adapter = ScriptedChunkAdapter(
        fixture.bindings.waypoint_sets["primary_waypoints"],  # type: ignore[arg-type]
        solver=fixture.solver,
        monitor=RecordingMonitor(fixture.shared),
        horizon_steps=config.pipeline.horizon_steps,
        control_dt_s=config.pipeline.control_dt_s,
        jitter=JointJitter(0.001, 0.005, 10),
        position_tolerance_m=config.adapter.ik_position_tolerance_m,
        orientation_tolerance_rad=config.adapter.ik_orientation_tolerance_rad,
    )
    backend = FakeBackend()
    observation = backend.reset(_one_episode_plan()[0].scene, 0)
    state = WorldState(SceneState([]), observation.robot)
    task = type("Task", (), {"instruction": "x", "target": "CAN", "destination": "d"})()
    draw = adapter.draw()
    first = draw(task, state, object(), 7)
    second = draw(task, state, object(), 7)
    third = draw(task, state, object(), 8)
    assert len(first.actions) == 50
    assert first.dt == 0.032
    assert first.source == "scripted"
    assert first.actions[0].q == second.actions[0].q  # type: ignore[attr-defined]
    assert first.actions[0].q != third.actions[0].q  # type: ignore[attr-defined]
    assert all(abs(value) <= 0.005 for value in first.actions[0].q.values())  # type: ignore[attr-defined]


def test_jitter_never_clips_or_silently_accepts_out_of_bound_draws(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    config = fixture.config
    adapter = ScriptedChunkAdapter(
        fixture.bindings.waypoint_sets["primary_waypoints"],  # type: ignore[arg-type]
        solver=fixture.solver,
        monitor=RecordingMonitor(fixture.shared),
        horizon_steps=config.pipeline.horizon_steps,
        control_dt_s=config.pipeline.control_dt_s,
        jitter=JointJitter(1.0e9, 1.0e-9, 1),
        position_tolerance_m=config.adapter.ik_position_tolerance_m,
        orientation_tolerance_rad=config.adapter.ik_orientation_tolerance_rad,
    )
    backend = FakeBackend()
    observation = backend.reset(_one_episode_plan()[0].scene, 0)
    state = WorldState(SceneState([]), observation.robot)
    task = type("Task", (), {"instruction": "x", "target": "CAN", "destination": "d"})()
    with pytest.raises(ValueError, match="rejection budget exhausted"):
        adapter.draw()(task, state, object(), 0)


def test_full_composition_records_k32_nplus1_gate_execution_pairing_and_recording(
    tmp_path, monkeypatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    report, sidecars, records, backend = _run_one(fixture, tmp_path)
    decision = records["decisions"][0]
    pairing = records["pairings"][0]

    assert report.summary["n_episodes"] == 1
    assert report.summary["backend"] == "fake"
    assert report.summary["state_source"] == "oracle"
    assert (sidecars["decisions"].parent / "episode-00.meta.json").exists() is False
    assert report.episodes_csv.is_file()
    assert report.summary_json.is_file()
    assert set(p.name for p in sidecars.values()) == {
        "decisions.jsonl",
        "prediction_pairing.jsonl",
        "errors.jsonl",
        "risk_rejection.json",
        "phase_latency.json",
        "verification_recovery.json",
        "provenance.json",
    }
    assert sidecars["errors"].read_text(encoding="utf-8").strip() == ""
    assert decision["candidate_count"] == 32
    assert decision["proposer_source"] == "scripted"
    assert len(decision["draw_seeds"]) == 32
    assert len(decision["candidates"]) == 32
    assert sum(c["sampling_seed"] is not None for c in decision["candidates"]) == 32
    assert decision["selected_candidate_id"] == "candidate-00"
    assert decision["execution"]["pre_apply_accepted"] is True
    assert decision["execution"]["apply_accepted"] is True
    assert decision["execution"]["backend_stepped"] is True
    assert decision["verification"]["next_action"] == "complete"
    assert decision["provenance"]["gate_select_calls"] == "33"
    assert decision["provenance"]["selected_index"] == "0"
    assert decision["provenance"]["selected_score"] == "1.0"
    assert decision["provenance"]["selection_veto_count"] == "0"
    assert decision["provenance"]["selection_prediction_count"] == "32"
    assert decision["provenance"]["selection_rationale"]
    assert decision["timings"]["model_inference_s"] is None
    assert (
        "analytic surrogate performs no neural inference"
        in (decision["timings"]["missing_reasons"]["model_inference_s"])
    )
    assert pairing["executed_candidate_id"] == "candidate-00"
    assert pairing["prediction_candidate_id"] == "candidate-00"
    assert pairing["status"] == "truncated"
    assert pairing["truncated_trajectory"] is True
    assert pairing["prediction_horizon_steps"] == 50
    assert pairing["valid_paired_steps"] == 1
    assert len(pairing["missing_observations"]) == 49
    assert pairing["error_metrics_si"]

    rejection = json.loads(sidecars["risk_rejection"].read_text(encoding="utf-8"))
    assert rejection["gate_calls"]["total"] == 33
    assert rejection["gate_calls"]["expected_total"] == 33
    assert rejection["gate_calls"]["all_decisions_expected"] is True
    assert "33 gate calls per decision" in rejection["gate_calls"]["policy"]
    assert rejection["disposition_counts"]["selected"] == 1
    assert rejection["disposition_counts"]["evaluated"] == 32
    latency = json.loads(sidecars["phase_latency"].read_text(encoding="utf-8"))
    assert "33 gate calls" in latency["note"]
    provenance = json.loads(sidecars["provenance"].read_text(encoding="utf-8"))
    assert provenance["report_sha256"] == fixture.report_sha
    assert provenance["state_source"] == "oracle"
    assert provenance["horizon_steps"] == "50"
    assert provenance["prediction_pairing_scope"].find("allow_truncated=True") >= 0
    assert backend.events.count("apply") == 1
    assert backend.events.count("step") == 1
    assert fixture.shared.events[fixture.shared.events.index("backend.apply") - 1] == "validate"


def test_generation_is_reproducible_from_explicit_seed(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    _, _, first, _ = _run_one(fixture, tmp_path, name="seed-a")
    _, _, second, _ = _run_one(fixture, tmp_path, name="seed-b")
    expected_decision_seed = int(
        np.random.SeedSequence([fixture.config.pipeline.decision_seed_base, 0]).generate_state(1)[0]
    )
    assert first["decisions"][0]["draw_seed"] == expected_decision_seed
    assert second["decisions"][0]["draw_seed"] == expected_decision_seed
    assert first["decisions"][0]["draw_seeds"] == second["decisions"][0]["draw_seeds"]


def test_hard_veto_takes_precedence_and_prevents_apply(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch, hard_value=False)
    config = replace(fixture.config, bounds=replace(fixture.config.bounds, max_decisions=1))
    _, sidecars, records, backend = _run_one(fixture, tmp_path, config=config, name="hard-veto")
    decision = records["decisions"][0]
    assert decision["selected_candidate_id"] is None
    assert decision["no_safe_set"] is True
    assert all(c["disposition"] == "hard_vetoed" for c in decision["candidates"])
    assert all(c["selected_candidate_id"] is None for c in decision["candidates"])
    assert decision["provenance"]["selected_index"] == "none"
    assert decision["provenance"]["selected_score"] == "none"
    assert decision["provenance"]["selection_veto_count"] == "32"
    assert backend.events.count("apply") == 0
    assert records["pairings"] == []
    rejection = json.loads(sidecars["risk_rejection"].read_text(encoding="utf-8"))
    assert rejection["gate_calls"]["total"] == 33
    assert rejection["disposition_counts"]["hard_vetoed"] == 32
    assert rejection["pooled"]["rate"] == 1.0


def test_no_safe_set_uses_bounded_recovery_ladder(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch, hard_value=False, alternate=True)
    config = replace(
        fixture.config, bounds=replace(fixture.config.bounds, max_decisions=3, max_steps=3)
    )
    _, _, records, backend = _run_one(fixture, tmp_path, config=config, name="no-safe-recovery")
    transitions = [d["recovery_events"][-1] for d in records["decisions"]]
    assert transitions == ["reobserve", "alternate_candidate_family", "abort"]
    assert backend.events.count("apply") == 0
    assert all(d["selected_candidate_id"] is None for d in records["decisions"])


def test_live_safety_veto_precedes_backend_apply(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch, arm_veto=True)
    _, _, records, backend = _run_one(fixture, tmp_path, name="pre-apply-veto")
    decision = records["decisions"][0]
    assert decision["selected_candidate_id"] == "candidate-00"
    assert decision["execution"]["pre_apply_accepted"] is False
    assert "injected live safety veto" in str(decision["execution"]["pre_apply_reason"])
    assert backend.events.count("apply") == 0
    assert records["pairings"][0]["status"] == "abort_boundary"
    assert records["pairings"][0]["error_metrics_si"] == {}


def test_verification_failure_switches_to_alternate_candidate_family(tmp_path, monkeypatch):
    fixture = _make_fixture(
        tmp_path, monkeypatch, verifier_mode="fail_then_success", alternate=True
    )
    _, sidecars, records, _ = _run_one(fixture, tmp_path, name="verification-recovery")
    assert len(records["decisions"]) == 2
    assert records["decisions"][0]["verification"]["next_action"] == "retry_once"
    assert records["decisions"][1]["verification"]["next_action"] == "complete"
    assert len(fixture.solver.calls) > 49
    assert fixture.solver.calls[0] != fixture.solver.calls[49]
    recovery = json.loads(sidecars["verification_recovery"].read_text(encoding="utf-8"))
    assert recovery["verification_transitions"] == {"complete": 1, "retry_once": 1}


def test_selected_prediction_pairing_uses_the_correct_decision_trajectory(tmp_path, monkeypatch):
    fixture = _make_fixture(
        tmp_path, monkeypatch, verifier_mode="fail_then_success", alternate=True
    )
    _, _, records, _ = _run_one(fixture, tmp_path, name="trajectory-offset")
    assert len(records["pairings"]) == 2
    assert [pair["decision_id"] for pair in records["pairings"]] == [
        "validation-clear-i00-s0-d00",
        "validation-clear-i00-s0-d01",
    ]
    assert records["pairings"][0]["prediction_candidate_id"] == "candidate-00"
    assert records["pairings"][1]["prediction_candidate_id"] == "candidate-00"
    assert all(pair["error_metrics_si"] for pair in records["pairings"])


def test_sidecars_are_created_exclusively_and_never_overwritten(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    _, sidecars, _, _ = _run_one(fixture, tmp_path, name="exclusive")
    folder = sidecars["decisions"].parent
    writer = WorldMindEvidenceWriter(folder.parent)
    with pytest.raises(FileExistsError):
        writer.start({"fixture": "must not overwrite"})
    assert folder.joinpath("decisions.jsonl").stat().st_size > 0


def test_blocked_template_and_missing_reviewed_inputs_fail_closed(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    repo_template = Path(__file__).resolve().parents[1] / "configs" / "worldmind_production.yaml"
    with pytest.raises(ValueError, match="blocked"):
        load_worldmind_config(repo_template)

    bad = yaml.safe_load(fixture.config_path.read_text(encoding="utf-8"))
    bad["report"]["definitions"] = str(fixture.root / "missing-definitions.json")
    missing_definitions = fixture.root / "missing-definitions.yaml"
    missing_definitions.write_text(yaml.safe_dump(bad, sort_keys=False), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="definitions"):
        load_worldmind_config(missing_definitions)

    bad = yaml.safe_load(fixture.config_path.read_text(encoding="utf-8"))
    bad["calibration"]["evidence_path"] = str(fixture.root / "missing-calibration.json")
    missing_calibration = fixture.root / "missing-calibration.yaml"
    missing_calibration.write_text(yaml.safe_dump(bad, sort_keys=False), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="calibration"):
        load_worldmind_config(missing_calibration)


def test_unregistered_bindings_fail_closed(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    empty = replace(fixture.bindings, components={})
    with pytest.raises(ValueError, match="unregistered WorldMind bindings"):
        validate_worldmind_bindings(fixture.config, empty)


def test_plan_preflight_rejects_test_training_wrong_state_and_existing_output(
    tmp_path, monkeypatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    out = tmp_path / "preflight"
    with pytest.raises(ValueError, match="TEST and TRAIN are sealed"):
        plan_worldmind_validation(
            fixture.config_path,
            out,
            split="test",
            backend="fake",
            state_source="oracle",
            bindings=fixture.bindings,
        )
    with pytest.raises(ValueError, match="VALIDATION only"):
        plan_worldmind_validation(
            fixture.config_path,
            out,
            split="train",
            backend="fake",
            state_source="oracle",
            bindings=fixture.bindings,
        )
    with pytest.raises(ValueError, match="state source must be declared as oracle"):
        plan_worldmind_validation(
            fixture.config_path,
            out,
            split="validation",
            backend="fake",
            state_source="perception",
            bindings=fixture.bindings,
        )
    out.mkdir()
    with pytest.raises(FileExistsError, match="prior evidence"):
        plan_worldmind_validation(
            fixture.config_path,
            out,
            split="validation",
            backend="fake",
            state_source="oracle",
            bindings=fixture.bindings,
        )
    plan = build_episode_plan("validation", 0, n=0, allow_test=False)
    assert len(plan) == 18
    assert sorted({entry.seed for entry in plan}) == [0, 1, 2]


def test_cli_exposes_new_command_preserves_existing_commands_and_fails_closed(
    tmp_path, monkeypatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "eval-worldmind" in help_result.stdout
    assert "eval-vla" in help_result.stdout
    assert "eval " in help_result.stdout

    result = runner.invoke(
        app,
        [
            "eval-worldmind",
            "--config",
            str(fixture.config_path),
            "--out",
            str(tmp_path / "cli-fail"),
            "--backend",
            "fake",
        ],
    )
    assert result.exit_code == 1
    assert "preflight failed" in result.stderr
    assert "unregistered WorldMind bindings" in result.stderr
    assert not (tmp_path / "cli-fail").exists()
