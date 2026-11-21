"""Offline FakeBackend regressions for diagnostic observed-state feedback execution.

Nothing here starts Webots, resumes a campaign, writes into a preserved V1-V4 run directory,
scores a pipeline, selects calibration parameters or reads TRAIN/TEST. The synthetic production
fixture supplies only deterministic bindings and config; feedback evidence is written to a fresh
temporary output directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worldmind_production import (  # noqa: PLC2701 - existing shared inert fixture helpers
    ARM_JOINTS,
    _diagnostic_config_path,
    _inert_runtime_probe,
    _make_fixture,
    _one_episode_plan,
)
from test_worldmind_v2_postmortem_regression import (  # noqa: PLC2701
    _production_waypoints,
    _tree_hash,
)
from test_worldmind_v4_postmortem_regression import (  # noqa: PLC2701
    AGGREGATES,
    ARM7,
    VETO_CASES,
    _state,
)
from test_worldmind_v4_postmortem_regression import (
    REPO_ROOT as V4_REPO_ROOT,
)

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import MAX_JOINT_STEP_RAD
from vlarobot.backends.worldmind_bindings import runtime_safety_limits
from vlarobot.core.types import ApplyResult, GripperCmd, JointTarget
from vlarobot.eval.decision_adapters import BackendApplyProbe, SafetyMonitorProbe
from vlarobot.eval.worldmind_config import (
    load_worldmind_diagnostic_config,
    validate_worldmind_diagnostic_bindings,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.execution.worldmind_pipeline import _world_state
from vlarobot.execution.worldmind_stage1_pipeline import (
    Stage1WorldMindEpisodePolicy,
    _execute_observed_state_feedback_chunk,
    _load_feedback_contract,
    _project_joint_target,
    run_stage1_evidence,
    validate_stage1_execution_mode,
)
from vlarobot.risk.comparison import Selection

FEEDBACK_MODE = "diagnostic_observed_state_feedback_chunk"
TARGET_RAD = 0.6
FROZEN_SOURCE_HASHES = {
    "src/vlarobot/execution/scripted_chunk_adapter.py": (
        "a3a3958bcb9448f6674fdc7ff65029eeabd6d536e0e3aeaf46f0c1c0a4e8bc92"
    ),
    "src/vlarobot/execution/safety_monitor.py": (
        "900c9218a9727a94cae0a1be9b071133c80925095cd7ca4b16241334adbc371a"
    ),
    "src/vlarobot/execution/worldmind_pipeline.py": (
        "27f3d45165d5e4e34fa4b670e41a709ff25206f2e5400fa441afe1dda887fdf8"
    ),
    "src/vlarobot/backends/worldmind_bindings.py": (
        "4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae"
    ),
    "src/vlarobot/planning/verification.py": (
        "e05330f3b6d36d711f367367dfed6a8780f8409ed933b3a657f98d772d48d747"
    ),
    "configs/worldmind_diagnostic.approved.yaml": (
        "9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03"
    ),
    "configs/report/definitions.approved.json": (
        "457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14"
    ),
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_steps: int = 1059,
    max_decisions: int = 1,
) -> tuple[Any, Any, dict[str, str]]:
    fixture = _make_fixture(tmp_path, monkeypatch)
    source = _diagnostic_config_path(fixture)
    config = load_worldmind_diagnostic_config(source)
    validate_worldmind_diagnostic_bindings(config, fixture.bindings)
    config = replace(
        config,
        bounds=replace(config.bounds, max_decisions=max_decisions, max_steps=max_steps),
    )
    _production_waypoints(fixture, config, _one_episode_plan())
    runtime = _inert_runtime_probe(config, fixture.bindings)
    assert config.pipeline.candidate_count == 32
    assert config.pipeline.horizon_steps == 50
    assert config.pipeline.control_dt_s == 0.032
    return fixture, config, runtime


class _IncrementalBackend:
    """FakeBackend that advances each joint by a fixed fraction of the commanded target."""

    def __init__(
        self, *, per_step_rad: float, initial_joint_pos: dict[str, float] | None = None
    ) -> None:
        self.inner = FakeBackend()
        self.per_step_rad = float(per_step_rad)
        self.initial_joint_pos = dict(initial_joint_pos or {})
        self.applied: list[JointTarget] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def reset(self, scenario: Any, seed: int) -> Any:
        observation = self.inner.reset(scenario, seed)
        if self.initial_joint_pos:
            self.inner._robot_state.joint_pos.update(self.initial_joint_pos)
            observation = self.inner._observation()
        return observation

    def get_ground_truth(self) -> Any:
        return self.inner.get_ground_truth()

    def apply(self, action: Any) -> ApplyResult:
        if not isinstance(action, JointTarget):
            return self.inner.apply(action)
        verdict = self.inner.safety_monitor.validate(action, self.inner._robot_state)
        if not verdict.accepted:
            return verdict
        self.applied.append(action)
        for name, target in action.q.items():
            current = float(self.inner._robot_state.joint_pos[name])
            delta = float(target) - current
            step = max(-self.per_step_rad, min(self.per_step_rad, delta))
            self.inner._robot_state.joint_pos[name] = current + step
        return ApplyResult(True)

    def step(self, n_steps: int = 1) -> Any:
        return self.inner.step(n_steps)


class _StalledBackend(_IncrementalBackend):
    """Backend that accepts every joint command but never changes the observed state."""

    def __init__(self) -> None:
        super().__init__(per_step_rad=0.0)


def _direct_feedback(
    *,
    fixture: Any,
    config: Any,
    backend: Any,
    actions: tuple[Any, ...],
    episode_steps_before: int = 0,
    waypoints: tuple[ScriptedWaypoint, ...] | None = None,
) -> tuple[Any, Any]:
    observation = backend.reset(_one_episode_plan()[0].scene, 0)
    ground_truth = backend.get_ground_truth()
    state = _world_state(observation, ground_truth, config, target=config.task.target)
    monitor = fixture.bindings.safety[config.safety.binding](config, {})
    selection = Selection(
        SimpleNamespace(actions=actions, source="scripted"),
        0,
        "offline feedback regression",
    )
    contract = _load_feedback_contract(config, fixture.solver)
    result = _execute_observed_state_feedback_chunk(
        decision_id="offline-d00",
        episode_id="offline-feedback",
        selection=selection,
        config=config,
        contract=contract,
        task=SimpleNamespace(
            instruction=config.task.instruction,
            target=config.task.target,
            destination=config.task.destination,
        ),
        observation=observation,
        state=state,
        ground_truth=ground_truth,
        backend_probe=BackendApplyProbe(backend),
        monitor_probe=SafetyMonitorProbe(monitor),
        waypoints=waypoints
        or (
            ScriptedWaypoint("PREGRASP", 1, joints={name: 0.0 for name in ARM_JOINTS}),
            ScriptedWaypoint("HOME", 1, joints={name: 0.0 for name in ARM_JOINTS}),
        ),
        episode_steps_before=episode_steps_before,
    )
    return result, backend


def test_frozen_sources_campaign_aggregates_and_budgets_are_unchanged() -> None:
    for relative, expected in FROZEN_SOURCE_HASHES.items():
        assert _sha256(V4_REPO_ROOT / relative) == expected
    for tag, expected in AGGREGATES.items():
        root = V4_REPO_ROOT / f"runs/worldmind/stage1-validation-{tag}-2026-11-20"
        assert _tree_hash(root, V4_REPO_ROOT) == expected
    config = load_worldmind_diagnostic_config(
        V4_REPO_ROOT / "configs/worldmind_diagnostic.approved.yaml"
    )
    assert config.bounds.max_steps == 1059
    assert config.pipeline.candidate_count == 32
    assert config.pipeline.horizon_steps == 50
    assert config.pipeline.control_dt_s == 0.032


def test_feedback_mode_is_explicit_and_full_system_is_rejected() -> None:
    assert validate_stage1_execution_mode(FEEDBACK_MODE) == FEEDBACK_MODE
    with pytest.raises(ValueError, match="unknown Stage 1 execution mode"):
        validate_stage1_execution_mode("")
    with pytest.raises(ValueError, match="only with scripted_proposer"):
        Stage1WorldMindEpisodePolicy(
            _one_episode_plan()[0].scene,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            "full_system_veto",
            FEEDBACK_MODE,
        )


def test_v4_arm7_veto_replay_accepts_only_observed_state_projection() -> None:
    limits = runtime_safety_limits()
    monitor = SafetyMonitor(limits)
    assert limits.joint_velocity["arm_7_joint"] == 1.76
    assert limits.command_horizon == 1.0
    for episode_id, case in VETO_CASES.items():
        observed = case["observed"]
        selected = case["rejected_target"]
        robot = _state({"arm_7_joint": observed})
        selected_action = JointTarget({"arm_7_joint": selected})
        assert not monitor.validate(selected_action, robot).accepted, episode_id
        executed, clamped = _project_joint_target(
            selected_action, robot.joint_pos, MAX_JOINT_STEP_RAD
        )
        assert clamped == ("arm_7_joint",)
        assert executed.q["arm_7_joint"] == pytest.approx(
            observed + (-MAX_JOINT_STEP_RAD if selected < observed else MAX_JOINT_STEP_RAD)
        )
        verdict = monitor.validate(executed, robot)
        assert verdict.accepted, (episode_id, verdict.reason)
        assert abs(executed.q["arm_7_joint"] - observed) <= MAX_JOINT_STEP_RAD + 1e-12


def test_v4_veto_replay_records_projected_executed_target_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    case = VETO_CASES["validation-clear-i03-s0"]
    observed = case["observed"]
    selected = case["rejected_target"]
    backend = _IncrementalBackend(
        per_step_rad=MAX_JOINT_STEP_RAD, initial_joint_pos={ARM7: observed}
    )
    result, _ = _direct_feedback(
        fixture=fixture,
        config=config,
        backend=backend,
        actions=(JointTarget({ARM7: selected}),),
        waypoints=(ScriptedWaypoint("PREGRASP", 1, joints={ARM7: observed}),),
    )

    assert result.diagnostic.termination_reason == "selected_chunk_converged"
    assert result.diagnostic.actions[0].terminal_reason == "target_converged"
    assert result.diagnostic.actions[0].convergence_tolerance_rad == 0.01
    assert result.diagnostic.actions[0].final_max_joint_error_rad <= 0.01
    assert result.diagnostic.budgets.max_joint_step_rad == MAX_JOINT_STEP_RAD
    assert len(result.diagnostic.attempts) > 1

    first = result.diagnostic.attempts[0]
    assert first.selected_action["q"][ARM7] == selected
    assert first.projection is not None
    assert first.projection.clamped_joints == (ARM7,)
    assert first.projection.selected_joint_target_rad[ARM7] == selected
    assert first.projection.executed_joint_target_rad[ARM7] == pytest.approx(
        observed - MAX_JOINT_STEP_RAD
    )
    assert first.executed_action is not None
    assert first.executed_action["q"][ARM7] == pytest.approx(observed - MAX_JOINT_STEP_RAD)
    assert first.executed_action["q"][ARM7] != selected
    assert first.safety_accepted is True
    assert first.apply_accepted is True
    assert first.backend_stepped is True
    assert result.diagnostic.prediction_provenance.comparability == "non_comparable"
    assert result.diagnostic.prediction_provenance.pairing_records_emitted == 0


def test_feedback_converges_with_progression_and_full_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch)
    fixture.solver.joints = {name: TARGET_RAD for name in ARM_JOINTS}
    backend = _IncrementalBackend(per_step_rad=0.05)
    out = tmp_path / "feedback-converged"
    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode=FEEDBACK_MODE,
        runtime_values=runtime,
    )

    assert sidecars["observed_state_feedback"].name == "observed-state-feedback.jsonl"
    provenance = json.loads((sidecars["provenance"]).read_text(encoding="utf-8"))
    assert provenance["execution_mode"] == FEEDBACK_MODE
    assert provenance["prediction_pairing_scope"].startswith("original H=50 candidate prediction")
    assert provenance["prediction_pairing_records"] == "not_emitted_non_comparable"
    assert provenance["feedback_evidence_schema_version"] == "3"
    assert provenance["feedback_step_bound_source"] == "pose_solver.execution_max_joint_step_rad"
    assert _jsonl(sidecars["prediction_pairing"]) == []

    rows = _jsonl(sidecars["observed_state_feedback"])
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == 3
    assert row["execution_mode"] == FEEDBACK_MODE
    assert row["selected_action_count"] == 50
    assert row["termination_reason"] == "selected_chunk_converged"
    assert row["episode_steps_before"] == 0
    assert row["episode_steps_after"] == row["backend_step_count"]
    assert row["episode_max_steps"] == 1059
    assert row["observation_count"] == row["backend_step_count"] + 1
    assert row["safety_validate_count"] == row["attempt_count"]
    assert row["safety_rejected_count"] == 0
    assert row["apply_rejected_count"] == 0
    assert row["apply_accepted_count"] == row["backend_step_count"]
    assert row["applied_action_count"] == row["backend_step_count"]
    assert row["backend_step_count"] == (
        sum(item["backend_step_count"] for item in row["actions"])
        + row["home_completion"]["backend_step_count"]
    )

    first = next(
        item
        for item in row["actions"]
        if item["selected_action_type"] == "JointTarget" and len(item["attempt_indices"]) > 1
    )
    attempts = [row["attempts"][index] for index in first["attempt_indices"]]
    assert len(attempts) > 1
    assert first["convergence_tolerance_rad"] == 0.01
    assert first["final_max_joint_error_rad"] <= 0.01
    for attempt in attempts:
        assert attempt["safety_accepted"] is True
        assert attempt["apply_accepted"] is True
        assert attempt["backend_stepped"] is True
        assert attempt["projection"] is not None
        assert attempt["projection"]["bound_rad"] == MAX_JOINT_STEP_RAD
        assert attempt["projection"]["clamped_joints"] == []
        assert attempt["projection"]["selected_joint_target_rad"] == attempt["selected_action"]["q"]
        assert attempt["projection"]["executed_joint_target_rad"] == attempt["executed_action"]["q"]
        assert attempt["max_joint_error_before_rad"] >= attempt["max_joint_error_after_rad"]
        assert attempt["progress_rad"] > 0
        assert attempt["observed_after"]["joint_pos"]
        assert attempt["episode_step_number"] <= 1059
    assert attempts[0]["max_joint_error_after_rad"] > attempts[-1]["max_joint_error_after_rad"]

    home_actions = [item for item in row["actions"] if item["home_action"]]
    assert home_actions
    assert all(item["convergence_tolerance_rad"] == 0.02 for item in home_actions)
    gripper_actions = [
        item for item in row["actions"] if item["selected_action_type"] == "GripperCmd"
    ]
    assert gripper_actions
    assert all(item["convergence_tolerance_rad"] is None for item in gripper_actions)

    budgets = row["budgets"]
    assert budgets["joint_stall_steps"] == 10
    assert budgets["joint_progress_epsilon_rad"] == 0.000001
    assert budgets["joint_move_timeout_steps"] == 800
    assert budgets["linear_move_timeout_steps"] == 600
    assert budgets["gripper_open_timeout_s"] == 4.0
    assert budgets["finger_speed_m_s"] == 0.03
    assert budgets["gripper_arrival_tolerance_m"] == pytest.approx(0.00096)
    assert budgets["gripper_timeout_steps"] == 125
    assert budgets["episode_max_steps"] == 1059
    assert budgets["applicability"]["linear_move_timeout"].startswith("not_applicable")
    assert budgets["applicability"]["gripper_open_timeout"].startswith("applied")
    assert budgets["applicability"]["episode_step_ceiling"].startswith("applied")
    provenance = row["prediction_provenance"]
    assert provenance["comparability"] == "non_comparable"
    assert provenance["pairing_records_emitted"] == 0
    assert provenance["original_prediction_horizon_steps"] == 50

    home = row["home_completion"]
    assert home["phase"] == "diagnostic_home_completion"
    assert home["source"] == ("configs/worldmind_diagnostic.approved.yaml:verification.home_joints")
    assert home["terminal_reason"] == "home_converged"
    assert home["final_max_joint_error_rad"] <= 0.02
    assert home["backend_step_count"] >= 0
    assert all(
        row["attempts"][index]["phase"] == "diagnostic_home_completion"
        for index in home["attempt_indices"]
    )
    summary = row["convergence_summary"]
    assert summary["converged_action_count"] == len(summary["converged_action_indices"])
    assert summary["non_converged_action_indices"] == []
    assert summary["non_converged_reasons"] == {}
    assert sidecars["diagnostic_events"].name == "diagnostic-events.jsonl"
    events = _jsonl(sidecars["diagnostic_events"])
    assert any(item["event_type"] == "diagnostic_home_completion" for item in events)
    assert sidecars["collision_proxy_attribution"].name == "collision-proxy-attribution.jsonl"
    proxies = _jsonl(sidecars["collision_proxy_attribution"])
    assert len(proxies) == 1
    assert proxies[0]["schema_version"] == 1
    assert proxies[0]["objects"]
    assert "proxy" in proxies[0]["note"].lower()

    decisions = _jsonl(sidecars["decisions"])
    assert len(decisions) == 1
    assert decisions[0]["execution"]["chunk"] is None
    assert decisions[0]["prediction_pairing"] is None
    selected_id = row["selected_candidate_id"]
    selected = next(
        item for item in decisions[0]["candidates"] if item["candidate_id"] == selected_id
    )
    assert selected["disposition"] == "selected"
    assert selected["selected_candidate_id"] == selected_id
    assert [item["selected_action_index"] for item in row["actions"]] == list(range(50))
    for action in row["actions"]:
        for index in action["attempt_indices"]:
            attempt = row["attempts"][index]
            assert attempt["selected_action_index"] == action["selected_action_index"]
            assert attempt["selected_action"] == action["selected_action"]


def test_feedback_stalls_and_times_out_with_reviewed_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    action = JointTarget({name: TARGET_RAD for name in ARM_JOINTS})
    actions = (action, action)

    stalled, stalled_backend = _direct_feedback(
        fixture=fixture, config=config, backend=_StalledBackend(), actions=actions
    )
    assert stalled.diagnostic.termination_reason == "joint_stall"
    assert stalled.backend_step_count == 10
    assert stalled.diagnostic.actions[0].terminal_reason == "joint_stall"
    assert stalled.diagnostic.attempts[-1].no_progress_steps == 10
    assert stalled.diagnostic.attempts[-1].progress_rad == 0.0
    assert all(item.safety_accepted for item in stalled.diagnostic.attempts)
    assert len(stalled_backend.applied) == 10

    timed_out, _ = _direct_feedback(
        fixture=fixture,
        config=config,
        backend=_IncrementalBackend(per_step_rad=0.0001),
        actions=actions,
    )
    assert timed_out.diagnostic.termination_reason == "joint_move_timeout"
    assert timed_out.backend_step_count == 800
    assert timed_out.diagnostic.actions[0].terminal_reason == "joint_move_timeout"
    assert all((item.no_progress_steps or 0) < 10 for item in timed_out.diagnostic.attempts)


def test_global_step_ceiling_is_not_reset_between_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    action = JointTarget({name: TARGET_RAD for name in ARM_JOINTS})
    result, backend = _direct_feedback(
        fixture=fixture,
        config=config,
        backend=_IncrementalBackend(per_step_rad=0.05),
        actions=(action, action),
        episode_steps_before=1058,
    )
    assert len(backend.applied) == 1
    assert result.backend_step_count == 1
    assert result.diagnostic.termination_reason == "global_step_ceiling"
    assert result.diagnostic.episode_steps_before == 1058
    assert result.diagnostic.episode_steps_after == 1059
    assert result.diagnostic.episode_max_steps == 1059
    assert result.diagnostic.actions[0].terminal_reason == "global_step_ceiling"


def test_feedback_abort_has_no_prediction_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch, max_steps=2)
    fixture.solver.joints = {name: TARGET_RAD for name in ARM_JOINTS}
    out = tmp_path / "feedback-abort"
    report, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        _IncrementalBackend(per_step_rad=0.05),
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode=FEEDBACK_MODE,
        runtime_values=runtime,
    )
    row = _jsonl(sidecars["observed_state_feedback"])[0]
    assert row["termination_reason"] == "global_step_ceiling"
    assert row["backend_step_count"] == 2
    assert row["episode_steps_after"] == 2
    assert row["episode_max_steps"] == 2
    assert _jsonl(sidecars["prediction_pairing"]) == []
    assert report.summary["stage1_diagnostic"]["execution_mode"] == FEEDBACK_MODE
    assert report.summary["stage1_diagnostic"]["scored_evaluation_allowed"] is False
    decisions = _jsonl(sidecars["decisions"])
    assert decisions and decisions[0]["prediction_pairing"] is None


def test_feedback_never_claims_gripper_settling_beyond_a_bounded_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    result, _ = _direct_feedback(
        fixture=fixture,
        config=config,
        backend=FakeBackend(),
        actions=(GripperCmd(opening=0.045),),
        waypoints=(ScriptedWaypoint("HOME", 1, joints={name: 0.0 for name in ARM_JOINTS}),),
    )
    assert result.diagnostic.termination_reason == "selected_chunk_converged"
    assert result.backend_step_count == 1
    record = result.diagnostic.actions[0]
    assert record.selected_action_type == "GripperCmd"
    assert record.convergence_tolerance_rad is None
    assert record.final_max_joint_error_rad is None
    assert record.executed_action_count == 1
    assert record.completion_unit == "m"
    assert record.completion_target == 0.045
    assert record.completion_error == 0.0
    budgets = result.diagnostic.budgets.applicability
    assert budgets["gripper_open_timeout"].startswith("applied")
    assert "no_linear_actions" in budgets["linear_move_timeout"]
