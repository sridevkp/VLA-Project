"""Offline ADR-0074 readiness regression: config split and CLI fail-stop exit.

No Webots process, campaign, scoring, calibration selection, full_system execution, TRAIN/TEST
access or Packet B/D execution is used. The CLI test drives the real Stage 1 finalizer with an
injected recorded EXCEPTION through FakeBackend.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from test_worldmind_stage1_runner import _one_episode_plan, _setup  # noqa: PLC2701
from typer.testing import CliRunner

from vlarobot import cli as cli_module
from vlarobot.backends.fake import FakeBackend
from vlarobot.cli import app
from vlarobot.eval.metrics import write_csv, write_json
from vlarobot.eval.runner import EvalEpisode, EvalReport
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "worldmind_diagnostic.yaml"
APPROVED = ROOT / "configs" / "worldmind_diagnostic.approved.yaml"
DEFINITIONS = ROOT / "configs" / "report" / "definitions.approved.json"
EXPECTED_SUBSTITUTIONS = frozenset(
    {
        "bounds.max_decisions",
        "bounds.max_steps",
        "report.definitions",
        "adapter.jitter.joint_std_rad",
        "adapter.jitter.joint_limit_rad",
        "adapter.jitter.max_attempts",
    }
)


def _leaf_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return {prefix or "root"}
        differences: set[str] = set()
        for key in left:
            child = f"{prefix}.{key}" if prefix else str(key)
            differences.update(_leaf_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix or "root"}
        differences = set()
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.update(_leaf_differences(left_item, right_item, f"{prefix}[{index}]"))
        return differences
    return set() if left == right else {prefix or "root"}


def test_approved_diagnostic_config_differs_only_in_six_owner_fields() -> None:
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    approved = yaml.safe_load(APPROVED.read_text(encoding="utf-8"))
    assert _leaf_differences(template, approved) == EXPECTED_SUBSTITUTIONS
    assert approved["bounds"]["max_decisions"] == 1060
    assert approved["bounds"]["max_steps"] == 1059
    assert approved["report"]["definitions"] == "report/definitions.approved.json"
    assert approved["adapter"]["jitter"] == {
        "joint_std_rad": 0.001,
        "joint_limit_rad": 0.005,
        "max_attempts": 10,
    }


def test_approved_diagnostic_config_loads_and_template_still_fails_closed() -> None:
    config = load_worldmind_diagnostic_config(APPROVED)
    assert config.status == "diagnostic_uncalibrated"
    assert config.diagnostic.scored_evaluation_allowed is False
    assert config.diagnostic.mode == "evidence_collection_only"
    assert config.diagnostic.calibration_stage == 1
    assert config.bounds.max_decisions == 1060
    assert config.bounds.max_steps == 1059
    assert config.report.definitions == DEFINITIONS.resolve()
    assert config.adapter.jitter.joint_std_rad == 0.001
    assert config.adapter.jitter.joint_limit_rad == 0.005
    assert config.adapter.jitter.max_attempts == 10
    assert config.pipeline.candidate_count == 32
    assert config.pipeline.horizon_steps == 50

    with pytest.raises(ValueError, match="bounds.max_decisions must be an integer >= 1"):
        load_worldmind_diagnostic_config(TEMPLATE)


def _two_episode_plan() -> tuple[EvalEpisode, ...]:
    first = _one_episode_plan()[0]
    second = EvalEpisode(
        index=1,
        scene=replace(first.scene, name="validation-clear-i00-s1"),
        seed=1,
    )
    return first, second


def test_cli_fail_stop_exits_one_preserves_partial_evidence_and_stops_after_first_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch)
    plan = _two_episode_plan()
    out = tmp_path / "fail-stop-cli"
    calls: list[dict[str, Any]] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("fail-stop CLI touched a forbidden backend path")

    def fake_preflight(*args: Any, **kwargs: Any) -> Any:
        del args
        assert kwargs["backend"] == "fake"
        assert kwargs["state_source"] == "oracle"
        assert kwargs["split"] == "validation"
        return SimpleNamespace(ready=True, blockers=(), config=config, plan=plan)

    def fake_run_evaluation(
        run_plan: Any,
        backend: Any,
        run_config: Any,
        *,
        policy_factory: Any,
        settings: Any,
        startup_s: float,
    ) -> EvalReport:
        del policy_factory, settings, startup_s
        calls.append(
            {
                "plan": tuple(run_plan),
                "backend": backend,
                "out_dir": run_config.out_dir,
                "stop_on_exception": run_config.stop_on_exception,
            }
        )
        assert len(run_plan) == 2
        assert isinstance(backend, FakeBackend)
        assert run_config.stop_on_exception is True

        run_dir = Path(run_config.out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "episodes": [
                    {"index": entry.index, "seed": entry.seed, "scene": entry.scene.name}
                    for entry in run_plan
                ]
            },
            run_dir / "plan.json",
        )
        row = {"episode_index": 0, "failure_state": "EXCEPTION", "failure_reason": "first boom"}
        write_csv([row], run_dir / "episodes.csv")
        (run_dir / "episode-00.meta.json").write_text(json.dumps(row), encoding="utf-8")

        return EvalReport(
            summary={
                "n_episodes": 1,
                "campaign_status": "incomplete_fail_stop",
                "fail_stop": {
                    "reason": "episode 0 recorded EXCEPTION: RuntimeError: first boom",
                    "episode_index": 0,
                    "episode_scene": "validation-clear-i00-s0",
                    "failure_state": "EXCEPTION",
                    "failure_reason": "first boom",
                    "completed_episodes": 1,
                    "planned_episodes": 2,
                },
            },
            rows=(row,),
            timings=(),
            plan_json=run_dir / "plan.json",
            episodes_csv=run_dir / "episodes.csv",
            summary_json=run_dir / "summary.json",
            timings_json=run_dir / "timings.json",
            total_wall_s=0.0,
        )

    monkeypatch.setattr(cli_module, "launch_webots_session", forbidden)
    monkeypatch.setattr(cli_module, "ZmqBackendClient", forbidden)
    monkeypatch.setattr(
        "vlarobot.eval.worldmind_preflight.run_worldmind_diagnostic_preflight", fake_preflight
    )
    monkeypatch.setattr(
        "vlarobot.backends.worldmind_bindings.production_worldmind_bindings",
        lambda: fixture.bindings,
    )
    monkeypatch.setattr(
        "vlarobot.backends.worldmind_bindings.runtime_preflight_probe",
        lambda _config, _bindings: runtime,
    )
    monkeypatch.setattr(
        "vlarobot.execution.worldmind_stage1_pipeline.run_evaluation", fake_run_evaluation
    )

    result = CliRunner().invoke(
        app,
        [
            "worldmind-stage1-evidence",
            "--config",
            str(source),
            "--out",
            str(out),
            "--arm-mode",
            "scripted_proposer",
            "--backend",
            "fake",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "CampaignFailStop" in result.output
    assert "first boom" in result.output
    assert len(calls) == 1
    assert calls[0]["plan"] == plan
    assert calls[0]["out_dir"] == out

    assert (out / "plan.json").is_file()
    assert (out / "episodes.csv").is_file()
    assert (out / "episode-00.meta.json").is_file()
    assert not (out / "episode-01.meta.json").exists()
    sidecars = out / "worldmind-sidecars"
    for name in ("provenance.json", "instrumentation.json", "errors.jsonl"):
        assert (sidecars / name).is_file()

    written = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert written["campaign_status"] == "incomplete_fail_stop"
    assert written["fail_stop"]["episode_index"] == 0
    assert written["fail_stop"]["planned_episodes"] == 2
    assert written["stage1_diagnostic"]["scored_evaluation_allowed"] is False
    plan_payload = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert len(plan_payload["episodes"]) == 2
