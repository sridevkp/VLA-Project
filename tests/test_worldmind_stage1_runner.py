"""Offline FakeBackend tests for the non-scored Stage 1 evidence runner.

No Webots process, real IK, neural inference, scoring, calibration selection, TEST/TRAIN
access or live evidence collection is used. The synthetic production fixture is reused only as
a deterministic binding/config source; Stage 1 remains an uncalibrated diagnostic here too.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worldmind_production import (  # noqa: PLC2701 - existing shared inert fixture
    SpyBackend,
    WorldMindFixture,
    _diagnostic_config_path,
    _inert_runtime_probe,
    _make_fixture,
    _one_episode_plan,
)
from typer.testing import CliRunner

from vlarobot import cli as cli_module
from vlarobot.backends.fake import FakeBackend
from vlarobot.cli import app
from vlarobot.eval.runner import EvalEpisode
from vlarobot.eval.worldmind_config import (
    load_worldmind_diagnostic_config,
    plan_worldmind_diagnostic,
    validate_worldmind_diagnostic_bindings,
)
from vlarobot.eval.worldmind_evidence import WorldMindEvidenceWriter
from vlarobot.execution.worldmind_pipeline import _world_state
from vlarobot.execution.worldmind_stage1_pipeline import (
    STAGE1_EVIDENCE_CLASS,
    STAGE1_GATE_CALL_POLICY,
    Stage1WorldMindEpisodePolicy,
    _hard_reason_codes,
    run_stage1_evidence,
    validate_stage1_arm_mode,
)
from vlarobot.world_model.rollout import Trajectory

FORBIDDEN_PROVENANCE_KEYS = frozenset(
    {
        "calibration_evidence_id",
        "calibration_id",
        "calibration_candidate_sha256",
        "calibration_evidence_sha256",
        "calibration_selected_candidate_id",
        "calibration_parameters",
        "calibration_provenance",
        "component_bindings",
        "component_units",
        "weights",
        "maximize",
        "selected_parameters",
        "approved",
        "approval_record",
        "candidate_path",
        "evaluator_binding",
    }
)


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **fixture_kwargs: Any
) -> tuple[WorldMindFixture, Path, Any, dict[str, str]]:
    fixture = _make_fixture(tmp_path, monkeypatch, **fixture_kwargs)
    source = _diagnostic_config_path(fixture)
    config = load_worldmind_diagnostic_config(source)
    validate_worldmind_diagnostic_bindings(config, fixture.bindings)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=1))
    runtime = _inert_runtime_probe(config, fixture.bindings)
    return fixture, source, config, runtime


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assert_nonscored(
    provenance: dict[str, Any],
    decisions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    assert FORBIDDEN_PROVENANCE_KEYS.isdisjoint(provenance)
    assert FORBIDDEN_PROVENANCE_KEYS.isdisjoint(summary["stage1_diagnostic"])
    assert provenance["status"] == "diagnostic_uncalibrated"
    assert provenance["scored_evaluation_allowed"] == "false"
    assert provenance["calibration_stage"] == "1"
    assert provenance["evidence_collection_split"] == "validation"
    assert provenance["independent_evaluation"] == "false"
    assert provenance["utility_ranking"] == "not_performed"
    assert summary["stage1_diagnostic"]["scored_evaluation_allowed"] is False
    assert summary["stage1_diagnostic"]["independent_evaluation"] is False
    for decision in decisions:
        assert decision["provenance"]["gate_call_policy"] == STAGE1_GATE_CALL_POLICY
        assert decision["provenance"]["gate_select_calls"] == "0"
        assert decision["provenance"]["utility_ranking"] == "not_performed"
        assert all(candidate["costs"] is None for candidate in decision["candidates"])
        assert decision["timings"]["riskgate_ranking_s"] is None
        assert (
            "does not run RiskGate" in decision["timings"]["missing_reasons"]["riskgate_ranking_s"]
        )


def _assert_recorder_layout(out: Path) -> None:
    for name in ("plan.json", "episodes.csv", "summary.json", "timings.json"):
        assert (out / name).is_file()
    for suffix in ("steps.npz", "actions.jsonl", "predicted.jsonl", "contacts.jsonl", "meta.json"):
        assert (out / f"episode-00.{suffix}").is_file()
    sidecars = out / "worldmind-sidecars"
    assert {path.name for path in sidecars.iterdir() if path.is_file()} == {
        "provenance.json",
        "decisions.jsonl",
        "prediction_pairing.jsonl",
        "errors.jsonl",
        "risk_rejection.json",
        "phase_latency.json",
        "verification_recovery.json",
        "instrumentation.json",
    }


def test_arm_mode_is_explicit_and_has_no_default() -> None:
    with pytest.raises(ValueError, match="unknown Stage 1 arm mode"):
        validate_stage1_arm_mode("")
    with pytest.raises(ValueError, match="unknown Stage 1 arm mode"):
        validate_stage1_arm_mode("full_system")
    assert validate_stage1_arm_mode("full_system_veto") == "full_system_veto"
    assert validate_stage1_arm_mode("scripted_proposer") == "scripted_proposer"


def test_diagnostic_plan_stays_frozen_validation_three_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, _, _ = _setup(tmp_path, monkeypatch)
    _, plan = plan_worldmind_diagnostic(
        source,
        tmp_path / "unused-plan-dir",
        split="validation",
        backend="fake",
        state_source="oracle",
        bindings=fixture.bindings,
    )
    assert len(plan) == 18
    assert sorted({entry.seed for entry in plan}) == [0, 1, 2]
    assert len({(entry.scene.name, entry.seed) for entry in plan}) == 18
    with pytest.raises(ValueError, match="TEST and TRAIN are sealed"):
        plan_worldmind_diagnostic(
            source,
            tmp_path / "unused-test-dir",
            split="test",
            backend="fake",
            state_source="oracle",
            bindings=fixture.bindings,
        )


def test_full_system_veto_records_hard_checks_and_never_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch)
    out = tmp_path / "full-system-veto"
    _, plan = plan_worldmind_diagnostic(
        source,
        out,
        split="validation",
        backend="fake",
        state_source="oracle",
        bindings=fixture.bindings,
    )
    backend = SpyBackend(FakeBackend(), fixture.shared)
    report, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="full_system_veto",
        runtime_values=runtime,
    )
    decisions = _jsonl(sidecars["decisions"])
    provenance = json.loads(sidecars["provenance"].read_text(encoding="utf-8"))
    summary = json.loads(sidecars["instrumentation"].read_text(encoding="utf-8"))
    assert len(decisions) == 18
    assert _assert_nonscored(provenance, decisions, report.summary) is None
    assert provenance["evidence_class"] == STAGE1_EVIDENCE_CLASS["full_system_veto"]
    assert provenance["configured_arm"] == "full_system"
    assert provenance["effective_arm"] == "full_system"
    assert provenance["full_system_checks_run"] == "true"
    assert provenance["diagnostic_actuation"] == "none"
    assert backend.events.count("apply") == 0
    assert backend.events.count("step") == 0
    assert backend.events.count("reset") == 18
    assert set(backend.events) == {"reset"}
    for decision in decisions:
        assert decision["arm"] == "full_system"
        assert decision["no_safe_set"] is True
        assert decision["selected_candidate_id"] is None
        assert decision["provenance"]["selection_policy"] == "no_selection_full_system_blocked"
        names = {name for name, _ in decision["candidates"][0]["hard_checks"]}
        assert {
            "prediction_complete",
            "prediction_constraints",
            "stage1_full_system_blocked",
        } <= names
        blocked = dict(decision["candidates"][0]["hard_checks"])["stage1_full_system_blocked"]
        assert blocked == "fail"
    assert summary["gate_select_calls"]["policy"] == STAGE1_GATE_CALL_POLICY
    assert summary["gate_select_calls"]["total"] == 0
    assert summary["gate_select_calls"]["expected_total"] == 0
    assert "RiskGate was not called" in summary["note"]
    _assert_recorder_layout(out)


def test_one_permitted_decision_advances_exactly_one_step_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    out = tmp_path / "lockstep"
    backend = SpyBackend(FakeBackend(), fixture.shared)
    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        runtime_values=runtime,
    )
    decision = _jsonl(sidecars["decisions"])[0]
    shared = fixture.shared.events
    assert backend.events == ["reset", "apply", "step"]
    assert shared[shared.index("backend.apply") - 1] == "validate"
    assert (
        shared.index("backend.reset") < shared.index("backend.apply") < shared.index("backend.step")
    )
    assert shared.count("validate") >= 1
    assert decision["execution"]["pre_apply_accepted"] is True
    assert decision["execution"]["apply_accepted"] is True
    assert decision["execution"]["backend_stepped"] is True


def test_scripted_proposer_is_labelled_and_preserves_live_safety_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch)
    del source
    out = tmp_path / "scripted-proposer"
    plan: tuple[EvalEpisode, ...] = _one_episode_plan()
    backend = SpyBackend(FakeBackend(), fixture.shared)
    report, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        runtime_values=runtime,
    )
    decision = _jsonl(sidecars["decisions"])[0]
    provenance = json.loads(sidecars["provenance"].read_text(encoding="utf-8"))
    _assert_nonscored(provenance, [decision], report.summary)
    assert provenance["evidence_class"] == "scripted diagnostic"
    assert provenance["configured_arm"] == "full_system"
    assert provenance["effective_arm"] == "scripted_chunk_adapter"
    assert provenance["full_system_checks_run"] == "false"
    assert report.summary["stage1_diagnostic"]["evidence_class"] == "scripted diagnostic"
    assert decision["arm"] == "scripted_chunk_adapter"
    assert decision["selected_candidate_id"] == "candidate-00"
    assert decision["provenance"]["evidence_class"] == "scripted diagnostic"
    assert decision["provenance"]["selection_policy"] == "first_safe_by_seed_order_no_utility"
    assert decision["execution"]["pre_apply_accepted"] is True
    assert decision["execution"]["apply_accepted"] is True
    shared = fixture.shared.events
    assert backend.events == ["reset", "apply", "step"]
    assert shared[shared.index("backend.apply") - 1] == "validate"
    assert shared.count("validate") >= 1
    assert _jsonl(sidecars["prediction_pairing"])[0]["status"] == "truncated"
    assert sidecars["errors"].read_text(encoding="utf-8").strip() == ""
    _assert_recorder_layout(out)


def test_hard_veto_prevents_scripted_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch, hard_value=False)
    del source
    out = tmp_path / "hard-veto"
    backend = SpyBackend(FakeBackend(), fixture.shared)
    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        runtime_values=runtime,
    )
    decision = _jsonl(sidecars["decisions"])[0]
    assert decision["selected_candidate_id"] is None
    assert decision["no_safe_set"] is True
    assert all(candidate["disposition"] == "hard_vetoed" for candidate in decision["candidates"])
    assert backend.events.count("apply") == 0
    assert "backend.apply" not in fixture.shared.events
    assert decision["recovery_events"][-1] == "reobserve"
    assert _jsonl(sidecars["prediction_pairing"]) == []


def test_live_safety_veto_precedes_any_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch, arm_veto=True)
    del source
    out = tmp_path / "live-veto"
    backend = SpyBackend(FakeBackend(), fixture.shared)
    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        runtime_values=runtime,
    )
    decision = _jsonl(sidecars["decisions"])[0]
    assert decision["execution"]["pre_apply_accepted"] is False
    assert "injected live safety veto" in decision["execution"]["pre_apply_reason"]
    assert backend.events == ["reset"]
    assert "backend.apply" not in fixture.shared.events
    assert "backend.step" not in fixture.shared.events
    assert "validate" in fixture.shared.events
    assert decision["execution"]["backend_stepped"] is False
    assert _jsonl(sidecars["prediction_pairing"])[0]["status"] == "abort_boundary"


def test_unverified_arm_sweep_stays_a_hard_full_system_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, _ = _setup(tmp_path, monkeypatch)
    scene = _one_episode_plan()[0].scene
    backend = FakeBackend()
    observation = backend.reset(scene, 0)
    state = _world_state(observation, backend.get_ground_truth(), config, target=config.task.target)
    policy = Stage1WorldMindEpisodePolicy(
        scene,
        config,
        fixture.bindings,
        WorldMindEvidenceWriter(tmp_path / "unused"),
        "full_system_veto",
    )
    chunk = SimpleNamespace(actions=(), source="scripted")
    trajectory = Trajectory(
        states=(state,),
        dt_s=config.pipeline.control_dt_s,
        hard_reasons=("unverified-arm-sweep",),
        assumptions=(),
        released_objects=(),
    )
    assessment = policy._assess(
        state, chunk, trajectory, hard_checks=list(config.assessment.hard_checks)
    )
    raw_results = dict(assessment.hard_checks)
    assert raw_results["unverified-arm-sweep"] is False
    assert raw_results["prediction_constraints"] is False
    assert raw_results["prediction_complete"] is True
    assert raw_results["stage1_full_system_blocked"] is False
    reasons, normalized = _hard_reason_codes(
        assessment.hard_checks, config.assessment.required_constraints
    )
    results = dict(normalized)
    assert results["unverified-arm-sweep"].value == "fail"
    assert results["prediction_constraints"].value == "fail"
    assert results["prediction_complete"].value == "pass"
    assert results["stage1_full_system_blocked"].value == "fail"
    assert "unverified-arm-sweep" in reasons
    assert (
        policy._select(
            [chunk], [SimpleNamespace(available=True, error_code=None)], [assessment]
        ).index
        is None
    )


def test_resume_reuses_sidecars_and_recorder_without_duplicating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch)
    del source
    out = tmp_path / "resume"
    plan = _one_episode_plan()
    kwargs = {
        "config": config,
        "out_dir": out,
        "bindings": fixture.bindings,
        "arm_mode": "scripted_proposer",
        "runtime_values": runtime,
    }
    first_backend = SpyBackend(FakeBackend(), fixture.shared)
    first_report, first_sidecars = run_stage1_evidence(plan, first_backend, **kwargs)
    first_decisions = _jsonl(first_sidecars["decisions"])
    first_sidecar_bytes = {name: path.read_bytes() for name, path in first_sidecars.items()}
    first_episode_files = sorted(path.name for path in out.glob("episode-*"))
    assert first_report.timings[0].resumed is False
    assert len(first_report.rows) == 1
    assert first_backend.events == ["reset", "apply", "step"]

    second_backend = SpyBackend(FakeBackend(), fixture.shared)
    second_report, second_sidecars = run_stage1_evidence(
        plan,
        second_backend,
        **{**kwargs, "resume": True},
    )
    assert second_backend.events == []
    assert _jsonl(second_sidecars["decisions"]) == first_decisions
    assert {
        name: path.read_bytes() for name, path in second_sidecars.items()
    } == first_sidecar_bytes
    assert sorted(path.name for path in out.glob("episode-*")) == first_episode_files
    assert len({row["episode_id"] for row in _jsonl(second_sidecars["decisions"])}) == 1
    assert len(second_report.rows) == 1
    assert second_report.timings[0].resumed is True
    assert second_report.summary["stage1_diagnostic"]["resumed"] is True
    instrumentation = json.loads(second_sidecars["instrumentation"].read_text(encoding="utf-8"))
    assert instrumentation["decisions_total"] == len(first_decisions)
    assert (out / "summary.json").read_text(encoding="utf-8").find('"resumed": true') >= 0


def test_resume_rejects_incompatible_mode_config_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    out = tmp_path / "resume-incompatible"
    plan = _one_episode_plan()
    common = {
        "config": config,
        "out_dir": out,
        "bindings": fixture.bindings,
        "arm_mode": "scripted_proposer",
        "runtime_values": runtime,
    }
    _, sidecars = run_stage1_evidence(plan, FakeBackend(), **common)
    original = {name: path.read_bytes() for name, path in sidecars.items()}
    altered_plan = (
        replace(plan[0], scene=replace(plan[0].scene, name="validation-cluttered-i01-s0")),
    )
    attempts = (
        {"arm_mode": "full_system_veto"},
        {"config": replace(config, bounds=replace(config.bounds, max_decisions=2))},
        {"plan": altered_plan},
    )
    for attempt in attempts:
        changed_plan = attempt.get("plan", plan)
        options = {**common, **{key: value for key, value in attempt.items() if key != "plan"}}
        with pytest.raises(ValueError, match="resume provenance differs"):
            run_stage1_evidence(
                changed_plan,
                FakeBackend(),
                **{**options, "resume": True},
            )
        assert {name: path.read_bytes() for name, path in sidecars.items()} == original


def test_cli_never_launches_webots_scores_or_calls_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, _, _ = _setup(tmp_path, monkeypatch)
    del source
    calls: dict[str, Any] = {}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("Stage 1 CLI called a forbidden scoring/backend path")

    def fake_preflight(*args: Any, **kwargs: Any) -> Any:
        del args
        calls["preflight"] = kwargs
        return SimpleNamespace(
            ready=True,
            blockers=(),
            config=SimpleNamespace(diagnostic=SimpleNamespace(scored_evaluation_allowed=False)),
            plan=_one_episode_plan(),
        )

    def fake_run(
        plan: Any,
        backend: Any,
        config: Any,
        out: Path,
        bindings: Any,
        *,
        arm_mode: str,
        runtime_values: Any,
        resume: bool,
    ) -> Any:
        calls["run"] = {
            "plan": plan,
            "backend": backend,
            "config": config,
            "out": out,
            "bindings": bindings,
            "arm_mode": arm_mode,
            "runtime_values": runtime_values,
            "resume": resume,
        }
        decisions = out / "worldmind-sidecars" / "decisions.jsonl"
        return (
            SimpleNamespace(summary={"n_episodes": 1}),
            {"decisions": decisions},
        )

    monkeypatch.setattr(cli_module, "launch_webots_session", forbidden)
    monkeypatch.setattr(
        "vlarobot.eval.worldmind_preflight.run_worldmind_diagnostic_preflight", fake_preflight
    )
    monkeypatch.setattr("vlarobot.eval.worldmind_preflight.run_worldmind_preflight", forbidden)
    monkeypatch.setattr("vlarobot.eval.worldmind_calibration.evaluate_calibration", forbidden)
    monkeypatch.setattr("vlarobot.eval.worldmind_calibration.build_calibration_evidence", forbidden)
    monkeypatch.setattr("vlarobot.eval.worldmind_calibration.load_calibration_evidence", forbidden)
    monkeypatch.setattr(
        "vlarobot.execution.worldmind_stage1_pipeline.run_stage1_evidence", fake_run
    )
    monkeypatch.setattr(
        "vlarobot.backends.worldmind_bindings.production_worldmind_bindings",
        lambda: fixture.bindings,
    )
    monkeypatch.setattr(
        "vlarobot.backends.worldmind_bindings.runtime_preflight_probe",
        lambda config, bindings: {"fixture": "probe"},
    )

    out = tmp_path / "cli-out"
    result = CliRunner().invoke(
        app,
        [
            "worldmind-stage1-evidence",
            "--config",
            str(tmp_path / "fixture" / "worldmind-diagnostic.yaml"),
            "--out",
            str(out),
            "--arm-mode",
            "full_system_veto",
            "--backend",
            "fake",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["preflight"]["allow_existing"] is False
    assert calls["run"]["arm_mode"] == "full_system_veto"
    assert calls["run"]["runtime_values"] == {"fixture": "probe"}
    assert "scored_evaluation_allowed=False" in result.output
    assert STAGE1_EVIDENCE_CLASS["full_system_veto"] in result.output


def test_no_riskgate_eq38_calibration_or_scored_metrics_are_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    out = tmp_path / "no-forbidden-paths"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("Stage 1 called a forbidden ranking/calibration path")

    for target in (
        "vlarobot.risk.gate.RiskGate.__init__",
        "vlarobot.risk.gate.RiskGate.select",
        "vlarobot.risk.comparison.ComparisonPlanner.__init__",
        "vlarobot.eval.report_contract.evaluate_components",
        "vlarobot.eval.worldmind_calibration.evaluate_calibration",
        "vlarobot.eval.worldmind_calibration.build_calibration_evidence",
        "vlarobot.eval.worldmind_calibration.load_calibration_candidate",
        "vlarobot.eval.worldmind_calibration.load_calibration_draft",
        "vlarobot.eval.worldmind_calibration.load_calibration_evidence",
        "vlarobot.eval.worldmind_calibration.rank_calibration_candidates",
    ):
        monkeypatch.setattr(target, forbidden)
    for name in list(fixture.bindings.components):
        monkeypatch.setitem(fixture.bindings.components, name, forbidden)
    for name in list(fixture.bindings.calibration_evaluators):
        monkeypatch.setitem(fixture.bindings.calibration_evaluators, name, forbidden)

    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        FakeBackend(),
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        runtime_values=runtime,
    )
    decision = _jsonl(sidecars["decisions"])[0]
    provenance = json.loads(sidecars["provenance"].read_text(encoding="utf-8"))
    instrumentation = json.loads(sidecars["instrumentation"].read_text(encoding="utf-8"))
    assert all(candidate["costs"] is None for candidate in decision["candidates"])
    assert provenance["utility_ranking"] == "not_performed"
    assert provenance["performance_claim"] == "not_permitted_bookkeeping_only"
    assert instrumentation["gate_select_calls"]["policy"] == STAGE1_GATE_CALL_POLICY
    assert instrumentation["gate_select_calls"]["total"] == 0
    assert instrumentation["gate_select_calls"]["expected_total"] == 0
    assert "RiskGate was not called" in instrumentation["note"]
