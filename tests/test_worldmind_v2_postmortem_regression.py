"""Offline regressions for the Stage 1 V2 postmortem (non-scored diagnostic evidence).

The V2 campaign is read-only evidence: nothing here reads Webots, resumes a campaign, or writes
into either preserved run directory. The arm-joint case is explicitly reconstructed from the
preserved first observation plus the pinned corrected solver/limits; it is not presented as a
runtime-recorded target.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worldmind_production import SpyBackend  # noqa: PLC2701 - existing shared spy
from test_worldmind_stage1_runner import _one_episode_plan, _setup

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import MAX_JOINT_STEP_RAD
from vlarobot.backends.worldmind_bindings import corrected_scripted_waypoints
from vlarobot.core.types import (
    ApplyResult,
    GripperCmd,
    JointTarget,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    TaskSpec,
    Vec3,
    VerifyResult,
    WorldState,
)
from vlarobot.eval.runner import CampaignFailStop
from vlarobot.eval.worldmind_config import plan_worldmind_diagnostic
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import (
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.execution.worldmind_pipeline import _world_state
from vlarobot.execution.worldmind_stage1_pipeline import (
    Stage1WorldMindEpisodePolicy,
    run_stage1_evidence,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
JOINT = "arm_6_joint"
REVIEWED_LOWER = -1.39
REVIEWED_UPPER = 1.39
# Reconstructed from the preserved V2 episode-00 state and draw seed 8787345, not runtime-recorded.
RECONSTRUCTED_CURRENT = -1.2534098586703584
RECONSTRUCTED_RAW_ENDPOINT = -1.3899999999987871
RECONSTRUCTED_JITTERED_ENDPOINT = -1.3912135100874889
RECONSTRUCTED_DELTA = -0.13780365141713058
ORIGINAL_EXCEPTION = (
    "Safety validation rejected action: joint target for arm_6_joint violates position limits"
)
PRESERVED_AGGREGATES = {
    "v1": "08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712",
    "v2": "b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb",
    "v3": "d4204a6fae79c16988e68f0b7cb61b27c90ba614f55fbfa507d1f666cf3d4d29",
}
EXPECTED_PHASES = (
    ("INIT", 0, 1, GripperCmd),
    ("PREGRASP", 2, 11, JointTarget),
    ("DESCEND", 12, 17, JointTarget),
    ("CLOSE", 18, 20, GripperCmd),
    ("LIFT", 21, 28, JointTarget),
    ("TRANSIT", 29, 36, JointTarget),
    ("PLACE_DESCEND", 37, 42, JointTarget),
    ("RELEASE", 43, 44, GripperCmd),
    ("RETREAT", 45, 46, JointTarget),
    ("HOME", 47, 49, JointTarget),
)


def _limits() -> SafetyLimits:
    return SafetyLimits(
        joint_position={JOINT: (REVIEWED_LOWER, REVIEWED_UPPER)},
        joint_velocity={JOINT: 1.76},
        command_horizon=1.0,
    )


def _state(joint_value: float) -> WorldState:
    robot = RobotState(
        joint_pos={JOINT: joint_value},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
        gripper_opening=0.045,
        base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), IDENTITY),
    )
    return WorldState(SceneState([]), robot)


def _stub_solver(endpoint: dict[str, float], residual_m: float = 0.0) -> Any:
    return SimpleNamespace(
        solve=lambda _pose, _state, *, seed=0: SimpleNamespace(  # noqa: ARG005
            joints=dict(endpoint), position_error_m=residual_m, orientation_error_rad=0.0
        )
    )


class _SpyMonitor:
    def __init__(self, inner: SafetyMonitor) -> None:
        self.inner = inner
        self.limits = inner.limits
        self.calls: list[ApplyResult] = []

    def validate(self, action: Any, robot: Any) -> ApplyResult:
        verdict = self.inner.validate(action, robot)
        self.calls.append(verdict)
        return verdict


def _adapter(
    *,
    endpoint: dict[str, float],
    monitor: Any,
    hold_steps: int,
    residual_m: float = 0.0,
    jitter: JointJitter | None = None,
    position_tolerance_m: float = 0.005,
) -> ScriptedChunkAdapter:
    return ScriptedChunkAdapter(
        (ScriptedWaypoint("TRANSIT", hold_steps, pose=Pose3D(Vec3(0.0, 0.0, 0.8), IDENTITY)),),
        solver=_stub_solver(endpoint, residual_m),
        monitor=monitor,
        horizon_steps=hold_steps,
        control_dt_s=0.032,
        jitter=jitter or JointJitter(0.0, 0.0, 1),
        max_joint_step_rad=MAX_JOINT_STEP_RAD,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=0.01,
    )


def test_reconstructed_v2_rejection_is_produced_by_the_original_delta_only_rule() -> None:
    """Legacy formula, not endpoint IK, produced the rejected absolute target."""
    bounded_delta = max(
        -MAX_JOINT_STEP_RAD,
        min(MAX_JOINT_STEP_RAD, RECONSTRUCTED_JITTERED_ENDPOINT - RECONSTRUCTED_CURRENT),
    )
    legacy_target = RECONSTRUCTED_CURRENT + bounded_delta

    assert bounded_delta == pytest.approx(RECONSTRUCTED_DELTA, abs=1e-15)
    assert abs(bounded_delta) <= MAX_JOINT_STEP_RAD
    assert legacy_target == pytest.approx(RECONSTRUCTED_JITTERED_ENDPOINT, abs=1e-15)
    assert legacy_target < REVIEWED_LOWER

    verdict = SafetyMonitor(_limits()).validate(
        JointTarget({JOINT: legacy_target}), _state(RECONSTRUCTED_CURRENT).robot
    )
    assert not verdict.accepted
    assert verdict.reason == "joint target for arm_6_joint violates position limits"


def test_fixed_candidate_path_eliminates_the_reconstructed_v2_rejection() -> None:
    """The near-limit endpoint is redrawn by jitter; SafetyMonitor thresholds stay unchanged."""
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    chunk = _adapter(
        endpoint={JOINT: RECONSTRUCTED_RAW_ENDPOINT},
        monitor=monitor,
        hold_steps=1,
        jitter=JointJitter(0.001, 0.005, 10),
    ).draw()(_task(), _state(RECONSTRUCTED_CURRENT), None, 147)
    target = float(chunk.actions[0].q[JOINT])  # type: ignore[attr-defined]

    assert REVIEWED_LOWER <= target <= REVIEWED_UPPER
    assert abs(target - RECONSTRUCTED_CURRENT) <= MAX_JOINT_STEP_RAD
    assert target != pytest.approx(RECONSTRUCTED_JITTERED_ENDPOINT, abs=1e-12)
    assert len(monitor.calls) == len(chunk.actions)
    assert all(verdict.accepted for verdict in monitor.calls)


def test_generated_targets_respect_absolute_and_per_tick_limits_without_bypassing_safety() -> None:
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    chunk = _adapter(
        endpoint={JOINT: RECONSTRUCTED_RAW_ENDPOINT},
        monitor=monitor,
        hold_steps=16,
        jitter=JointJitter(0.001, 0.005, 10),
    ).draw()(_task(), _state(1.0), None, 147)

    current = 1.0
    for action in chunk.actions:
        target = float(action.q[JOINT])  # type: ignore[attr-defined]
        assert REVIEWED_LOWER <= target <= REVIEWED_UPPER
        assert abs(target - current) <= MAX_JOINT_STEP_RAD + 1e-12
        current = target
    assert len(monitor.calls) == len(chunk.actions)
    assert all(verdict.accepted for verdict in monitor.calls)


def test_endpoint_residual_outside_reviewed_tolerance_fails_closed() -> None:
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    with pytest.raises(ValueError, match="residual outside reviewed tolerance"):
        _adapter(
            endpoint={JOINT: 0.0},
            monitor=monitor,
            hold_steps=1,
            residual_m=0.01,
            position_tolerance_m=0.005,
        ).draw()(_task(), _state(0.0), None, 3)
    assert monitor.calls == []


def test_raw_endpoint_outside_reviewed_absolute_limits_fails_closed() -> None:
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    with pytest.raises(ValueError, match="violates reviewed absolute position limits"):
        _adapter(
            endpoint={JOINT: -1.5},
            monitor=monitor,
            hold_steps=1,
        ).draw()(_task(), _state(0.0), None, 3)
    assert monitor.calls == []


def test_stage1_partial_18_episode_fail_stop_finalizes_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch)
    _, plan = plan_worldmind_diagnostic(
        source,
        tmp_path / "frozen-plan",
        split="validation",
        backend="fake",
        state_source="oracle",
        bindings=fixture.bindings,
    )
    assert len(plan) == 18

    calls: list[int] = []

    def explode(_policy: Stage1WorldMindEpisodePolicy, _backend: Any, seed: int) -> Any:
        calls.append(seed)
        raise RuntimeError(ORIGINAL_EXCEPTION)

    monkeypatch.setattr(Stage1WorldMindEpisodePolicy, "execute", explode)
    out = tmp_path / "stage1-v2-reconstructed-partial"

    with pytest.raises(CampaignFailStop, match="arm_6_joint violates position limits"):
        run_stage1_evidence(
            plan,
            FakeBackend(),
            config,
            out,
            fixture.bindings,
            arm_mode="scripted_proposer",
            runtime_values=runtime,
        )

    assert calls == [0]
    assert not (out / "episode-01.meta.json").exists()
    for name in ("plan.json", "episodes.csv", "summary.json", "timings.json"):
        assert (out / name).is_file()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign_status"] == "incomplete_fail_stop"
    assert summary["planned_episodes"] == 18
    assert summary["n_episodes"] == 1
    assert summary["seeds"] == [0, 1, 2]
    assert len(summary["per_scene"]) == 6
    assert all(scene["seeds"] == [0, 1, 2] for scene in summary["per_scene"])
    assert ORIGINAL_EXCEPTION in summary["fail_stop"]["reason"]

    plan_payload = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert len(plan_payload["episodes"]) == 18
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["failure_state"] == "EXCEPTION"
    assert ORIGINAL_EXCEPTION in rows[0]["failure_reason"]
    timings = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    assert len(timings["episodes"]) == 1

    sidecars = out / "worldmind-sidecars"
    for name in (
        "provenance.json",
        "decisions.jsonl",
        "prediction_pairing.jsonl",
        "errors.jsonl",
        "risk_rejection.json",
        "phase_latency.json",
        "verification_recovery.json",
        "instrumentation.json",
    ):
        assert (sidecars / name).is_file()


def _task() -> Any:
    return SimpleNamespace(instruction="offline V2 postmortem", target="CAN", destination="target")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _task_spec(config: Any) -> TaskSpec:
    return TaskSpec(
        instruction=config.task.instruction,
        target=config.task.target,
        destination=config.task.destination,
    )


def _production_waypoints(fixture: Any, config: Any, plan: Any) -> None:
    probe = FakeBackend()
    observation = probe.reset(plan[0].scene, plan[0].seed)
    state = _world_state(observation, probe.get_ground_truth(), config, target=config.task.target)
    fixture.bindings.waypoint_sets["primary_waypoints"] = corrected_scripted_waypoints(
        config, state, _task_spec(config)
    )


class _VerifierSpy:
    def __init__(self, inner: Any, calls: list[dict[str, int]], events: list[str]) -> None:
        self.inner = inner
        self.calls = calls
        self.events = events

    def check(self, task: Any, predicted: Any, observed: Any) -> Any:
        self.calls.append(
            {"applies": self.events.count("apply"), "steps": self.events.count("step")}
        )
        return self.inner.check(task, predicted, observed)

    def next_action(self, result: Any) -> str:
        return self.inner.next_action(result)


class _QueueVerifier:
    def __init__(
        self,
        results: list[Any],
        transitions: list[str],
        calls: list[dict[str, int]],
        events: list[str],
    ) -> None:
        self.results = results
        self.transitions = transitions
        self.calls = calls
        self.events = events

    def check(self, task: Any, predicted: Any, observed: Any) -> Any:
        del task, predicted, observed
        self.calls.append(
            {"applies": self.events.count("apply"), "steps": self.events.count("step")}
        )
        if not self.results:
            raise AssertionError("verifier called more often than configured")
        return self.results.pop(0)

    def next_action(self, result: Any) -> str:
        del result
        if not self.transitions:
            raise AssertionError("verifier transition requested more often than configured")
        return self.transitions.pop(0)


class _NoVerification:
    def check(self, task: Any, predicted: Any, observed: Any) -> Any:
        del task, predicted, observed
        pytest.fail("verification must be skipped after an execution abort")

    def next_action(self, result: Any) -> str:
        del result
        pytest.fail("verification must be skipped after an execution abort")


class _ApplyRejectBackend:
    def __init__(self, inner: FakeBackend, shared: Any, reject_on: int) -> None:
        self.inner = inner
        self.shared = shared
        self.reject_on = reject_on
        self.applies = 0
        self.events: list[str] = []

    def _record(self, event: str) -> None:
        self.events.append(event)
        self.shared.events.append(f"backend.{event}")

    def reset(self, scenario: Any, seed: int) -> Any:
        self._record("reset")
        return self.inner.reset(scenario, seed)

    def apply(self, action: Any) -> ApplyResult:
        self._record("apply")
        self.applies += 1
        if self.applies == self.reject_on:
            return ApplyResult(False, "injected backend apply rejection")
        return self.inner.apply(action)

    def step(self, n_steps: int = 1) -> Any:
        self._record("step")
        return self.inner.step(n_steps)

    def get_ground_truth(self) -> Any:
        return self.inner.get_ground_truth()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _StepFailureBackend:
    def __init__(self, inner: FakeBackend, shared: Any, fail_on: int) -> None:
        self.inner = inner
        self.shared = shared
        self.fail_on = fail_on
        self.steps = 0
        self.events: list[str] = []

    def _record(self, event: str) -> None:
        self.events.append(event)
        self.shared.events.append(f"backend.{event}")

    def reset(self, scenario: Any, seed: int) -> Any:
        self._record("reset")
        return self.inner.reset(scenario, seed)

    def apply(self, action: Any) -> ApplyResult:
        self._record("apply")
        return self.inner.apply(action)

    def step(self, n_steps: int = 1) -> Any:
        self.steps += 1
        if self.steps == self.fail_on:
            self._record("step-error")
            raise RuntimeError("injected step failure")
        self._record("step")
        return self.inner.step(n_steps)

    def get_ground_truth(self) -> Any:
        return self.inner.get_ground_truth()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _IntermediateVetoMonitor:
    def __init__(self, inner: Any, backend: Any, veto_after: int) -> None:
        self.inner = inner
        self.backend = backend
        self.veto_after = veto_after

    @property
    def limits(self) -> Any:
        return self.inner.limits

    def validate(self, action: Any, robot: Any) -> ApplyResult:
        verdict = self.inner.validate(action, robot)
        if self.backend.events.count("apply") >= self.veto_after:
            return ApplyResult(False, "injected intermediate safety veto")
        return verdict


def _assert_safety_precedes_apply(shared_events: list[str], applies: int) -> None:
    for index, event in enumerate(shared_events):
        if event == "backend.apply":
            assert index > 0 and shared_events[index - 1] == "validate"
    assert shared_events.count("backend.apply") == applies


def _tree_hash(root: Path, repo_root: Path) -> str:
    lines: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(repo_root).as_posix()):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(repo_root).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def test_preserved_50_action_phase_cadence_matches_frozen_waypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, _ = _setup(tmp_path, monkeypatch)
    plan = _one_episode_plan()
    probe = FakeBackend()
    observation = probe.reset(plan[0].scene, plan[0].seed)
    state = _world_state(observation, probe.get_ground_truth(), config, target=config.task.target)
    waypoints = corrected_scripted_waypoints(config, state, _task_spec(config))
    offset = 0
    actual: list[tuple[str, int, int]] = []
    for waypoint in waypoints:
        actual.append((waypoint.label, offset, offset + waypoint.hold_steps - 1))
        offset += waypoint.hold_steps
    assert tuple(actual) == tuple((label, start, end) for label, start, end, _ in EXPECTED_PHASES)
    assert offset == 50


def test_preserved_campaign_aggregates_are_byte_identical() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for tag, expected in PRESERVED_AGGREGATES.items():
        root = repo_root / f"runs/worldmind/stage1-validation-{tag}-2026-11-20"
        assert _tree_hash(root, repo_root) == expected


def test_open_loop_chunk_executes_all_50_actions_with_terminal_only_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=50))
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = SpyBackend(FakeBackend(), fixture.shared)
    verify_calls: list[dict[str, int]] = []
    base_factory = fixture.bindings.verification["placement_verifier"]

    def verification_factory(loaded: Any, destination: Any, rest: Any, inputs: Any) -> Any:
        return _VerifierSpy(
            base_factory(loaded, destination, rest, inputs), verify_calls, backend.events
        )

    fixture.bindings.verification["placement_verifier"] = verification_factory
    out = tmp_path / "open-loop-success"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    assert backend.events == ["reset"] + ["apply", "step"] * 50
    _assert_safety_precedes_apply(fixture.shared.events, 50)
    assert verify_calls == [{"applies": 50, "steps": 50}]
    decision = _jsonl(sidecars["decisions"])[0]
    chunk = decision["execution"]["chunk"]
    records = chunk["actions"]
    assert [record["action_index"] for record in records] == list(range(50))
    assert chunk["chunk_action_count"] == 50
    assert chunk["attempted_action_count"] == 50
    assert chunk["applied_action_count"] == 50
    assert chunk["backend_step_count"] == 50
    assert chunk["observation_count"] == 51
    assert chunk["safety_validate_count"] == 50
    assert chunk["safety_accepted_count"] == 50
    assert chunk["termination_reason"] == "chunk_complete"
    assert chunk["missing_horizon_steps"] == []
    assert decision["verification"]["next_action"] == "complete"
    for _label, start, end, kind in EXPECTED_PHASES:
        assert all(record["action_type"] == kind.__name__ for record in records[start : end + 1])
    assert records[2]["action_type"] == "JointTarget"
    assert records[2]["action"]["q"]
    limits = fixture.bindings.safety["safety_factory"](config, {}).limits.joint_position
    previous: dict[str, float] = {}
    for record in records:
        if record["action_type"] != "JointTarget":
            continue
        for name, target in record["action"]["q"].items():
            lower, upper = limits[name]
            assert lower <= target <= upper
            if name in previous:
                assert abs(target - previous[name]) <= MAX_JOINT_STEP_RAD + 1e-9
            previous[name] = target
    pairing = _jsonl(sidecars["prediction_pairing"])[0]
    assert pairing["status"] == "paired"
    assert pairing["valid_paired_steps"] == 50
    assert pairing["prediction_horizon_steps"] == 50
    assert pairing["missing_observations"] == []
    assert pairing["truncated_trajectory"] is False
    provenance = json.loads(sidecars["provenance"].read_text(encoding="utf-8"))
    summary = json.loads(sidecars["instrumentation"].read_text(encoding="utf-8"))
    assert provenance["execution_mode"] == "diagnostic_open_loop_chunk"
    assert provenance["execution_mode_label"] == "diagnostic open-loop chunk execution"
    assert decision["provenance"]["execution_mode"] == "diagnostic_open_loop_chunk"
    assert summary["decisions_total"] == 1
    assert "open-loop" in provenance["prediction_pairing_scope"]
    assert sidecars["errors"].read_text(encoding="utf-8").strip() == ""


def test_open_loop_chunk_fails_stop_at_an_intermediate_safety_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=50))
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = SpyBackend(FakeBackend(), fixture.shared)
    base_safety = fixture.bindings.safety["safety_factory"]

    def safety_factory(loaded: Any, inputs: Any) -> Any:
        return _IntermediateVetoMonitor(base_safety(loaded, inputs), backend, 7)

    fixture.bindings.safety["safety_factory"] = safety_factory
    fixture.bindings.verification["placement_verifier"] = lambda *args: _NoVerification()
    out = tmp_path / "open-loop-safety-abort"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    assert backend.events == ["reset"] + ["apply", "step"] * 7
    _assert_safety_precedes_apply(fixture.shared.events, 7)
    decision = _jsonl(sidecars["decisions"])[0]
    chunk = decision["execution"]["chunk"]
    assert chunk["termination_reason"] == "safety_rejected"
    assert chunk["attempted_action_count"] == 8
    assert chunk["applied_action_count"] == 7
    assert chunk["backend_step_count"] == 7
    assert chunk["observation_count"] == 8
    assert chunk["safety_rejected_count"] == 1
    assert chunk["missing_horizon_steps"] == list(range(7, 50))
    assert chunk["actions"][7]["pre_apply_accepted"] is False
    assert "injected intermediate safety veto" in chunk["actions"][7]["pre_apply_reason"]
    assert decision["verification"] is None
    pairing = _jsonl(sidecars["prediction_pairing"])[0]
    assert pairing["status"] == "abort_boundary"
    assert pairing["valid_paired_steps"] == 7
    assert pairing["missing_observations"] == list(range(8, 51))
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["failure_state"] == "SAFETY_ABORT"
    assert sidecars["errors"].read_text(encoding="utf-8").strip() == ""


def test_open_loop_chunk_fails_stop_on_backend_apply_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=50))
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = _ApplyRejectBackend(FakeBackend(), fixture.shared, reject_on=8)
    fixture.bindings.verification["placement_verifier"] = lambda *args: _NoVerification()
    out = tmp_path / "open-loop-apply-reject"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    assert backend.events == ["reset"] + ["apply", "step"] * 7 + ["apply"]
    _assert_safety_precedes_apply(fixture.shared.events, 8)
    decision = _jsonl(sidecars["decisions"])[0]
    chunk = decision["execution"]["chunk"]
    assert chunk["termination_reason"] == "backend_apply_rejected"
    assert chunk["attempted_action_count"] == 8
    assert chunk["applied_action_count"] == 7
    assert chunk["apply_accepted_count"] == 7
    assert chunk["apply_rejected_count"] == 1
    assert chunk["actions"][7]["apply_accepted"] is False
    assert "injected backend apply rejection" in chunk["actions"][7]["apply_reason"]
    assert decision["verification"] is None
    pairing = _jsonl(sidecars["prediction_pairing"])[0]
    assert pairing["status"] == "abort_boundary"
    assert pairing["valid_paired_steps"] == 7
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["failure_state"] == "BACKEND_ABORT"


def test_open_loop_chunk_fails_stop_on_step_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=50))
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = _StepFailureBackend(FakeBackend(), fixture.shared, fail_on=8)
    fixture.bindings.verification["placement_verifier"] = lambda *args: _NoVerification()
    out = tmp_path / "open-loop-step-exception"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    assert backend.events == ["reset"] + ["apply", "step"] * 7 + ["apply", "step-error"]
    _assert_safety_precedes_apply(fixture.shared.events, 8)
    decision = _jsonl(sidecars["decisions"])[0]
    chunk = decision["execution"]["chunk"]
    assert chunk["termination_reason"] == "execution_exception"
    assert chunk["attempted_action_count"] == 8
    assert chunk["applied_action_count"] == 7
    assert chunk["backend_step_count"] == 7
    assert chunk["apply_accepted_count"] == 8
    assert chunk["apply_rejected_count"] == 0
    assert chunk["missing_horizon_steps"] == list(range(7, 50))
    assert "injected step failure" in chunk["error"]
    assert chunk["actions"][7]["apply_accepted"] is True
    assert chunk["actions"][7]["applied"] is False
    assert chunk["actions"][7]["backend_stepped"] is False
    assert decision["verification"] is None
    pairing = _jsonl(sidecars["prediction_pairing"])[0]
    assert pairing["status"] == "abort_boundary"
    assert pairing["valid_paired_steps"] == 7
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["failure_state"] == "EXECUTION_EXCEPTION"


def test_open_loop_chunk_preserves_retry_once_then_abort_and_verifier_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=2, max_steps=100))
    verification = config.verification
    assert (
        verification.distance_tolerance_m,
        verification.height_tolerance_m,
        verification.tilt_limit_rad,
        verification.home_tolerance_rad,
    ) == (0.01, 0.01, 0.1, 0.01)
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = SpyBackend(FakeBackend(), fixture.shared)
    verify_calls: list[dict[str, int]] = []
    results = [
        VerifyResult(False, "synthetic first miss"),
        VerifyResult(False, "synthetic second miss"),
    ]
    transitions = ["retry_once", "abort"]

    def verification_factory(loaded: Any, destination: Any, rest: Any, inputs: Any) -> Any:
        del loaded, destination, rest, inputs
        return _QueueVerifier(list(results), list(transitions), verify_calls, backend.events)

    fixture.bindings.verification["placement_verifier"] = verification_factory
    out = tmp_path / "open-loop-recovery"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    assert backend.events == ["reset"] + ["apply", "step"] * 100
    _assert_safety_precedes_apply(fixture.shared.events, 100)
    decisions = _jsonl(sidecars["decisions"])
    assert len(decisions) == 2
    assert all(
        row["execution"]["chunk"]["termination_reason"] == "chunk_complete" for row in decisions
    )
    assert [row["verification"]["next_action"] for row in decisions] == transitions
    assert verify_calls == [
        {"applies": 50, "steps": 50},
        {"applies": 100, "steps": 100},
    ]
    pairings = _jsonl(sidecars["prediction_pairing"])
    assert [row["status"] for row in pairings] == ["paired", "paired"]
    assert [row["valid_paired_steps"] for row in pairings] == [50, 50]
    recovery = json.loads(sidecars["verification_recovery"].read_text(encoding="utf-8"))
    assert recovery["verification_transitions"] == {"abort": 1, "retry_once": 1}
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["failure_state"] == "VERIFICATION_ABORT"
    assert (
        verification.distance_tolerance_m,
        verification.height_tolerance_m,
        verification.tilt_limit_rad,
        verification.home_tolerance_rad,
    ) == (0.01, 0.01, 0.1, 0.01)
