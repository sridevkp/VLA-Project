"""Eval runner + CLI: plan, recording, reproducibility, and fail-closed guards (ADR-0033)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vlarobot.backends.fake import FakeBackend
from vlarobot.cli import app
from vlarobot.eval.runner import (
    EvalRunConfig,
    build_episode_plan,
    canonical_pipeline,
    run_evaluation,
)
from vlarobot.eval.settings import load_eval_settings
from vlarobot.execution.scripted_baseline import scripted_policy_factory


def test_canonical_pipeline_and_fail_closed() -> None:
    assert canonical_pipeline("baseline") == "baseline"
    assert canonical_pipeline("scripted") == "baseline"
    assert canonical_pipeline("vla_only") == "vla_only"
    with pytest.raises(ValueError, match="not implemented"):
        canonical_pipeline("unimplemented-arm")


def test_build_episode_plan_seed_major_and_limits() -> None:
    settings = load_eval_settings()
    plan = build_episode_plan("train", 0, settings=settings)
    assert len(plan) == settings.sizes.train * settings.baseline_seeds_per_scene
    # Seed-major: all 8 scenes for seed 0, then all 8 for seed 1, etc.
    assert [entry.seed for entry in plan[: settings.sizes.train]] == [0] * settings.sizes.train
    assert [entry.seed for entry in plan[settings.sizes.train : 2 * settings.sizes.train]] == [
        1
    ] * settings.sizes.train
    # n=0 means all; n > available fails loud.
    assert len(build_episode_plan("train", 0, n=2, settings=settings)) == 2
    with pytest.raises(ValueError, match="exceeds"):
        build_episode_plan("train", 0, n=len(plan) + 1, settings=settings)
    with pytest.raises(ValueError, match="test split is frozen"):
        build_episode_plan("test", 0, settings=settings)


def test_eval_runner_is_reproducible_on_fake_backend(tmp_path: Path) -> None:
    settings = load_eval_settings()
    plan = build_episode_plan("train", 5, n=2, settings=settings)
    common = dict(policy_factory=scripted_policy_factory, settings=settings)

    first = run_evaluation(
        plan, FakeBackend(), EvalRunConfig(tmp_path / "a", base_seed=5, backend="fake"), **common
    )
    second = run_evaluation(
        plan, FakeBackend(), EvalRunConfig(tmp_path / "b", base_seed=5, backend="fake"), **common
    )

    assert first.rows == second.rows
    assert first.summary["success"] == second.summary["success"]
    assert first.summary["per_scene"] == second.summary["per_scene"]
    assert first.episodes_csv.read_text(encoding="utf-8") == second.episodes_csv.read_text(
        encoding="utf-8"
    )
    assert first.timings_json.is_file()

    # Plan, metadata, and evidence files are present and consistent.
    plan_payload = json.loads((tmp_path / "a" / "plan.json").read_text(encoding="utf-8"))
    assert [entry["index"] for entry in plan_payload["episodes"]] == [0, 1]
    meta = json.loads((tmp_path / "a" / "episode-00.meta.json").read_text(encoding="utf-8"))
    assert meta["pipeline"] == "baseline"
    assert meta["state_source"] == "oracle"
    assert meta["reset_seed"] == plan[0].seed
    assert (
        meta["outcome"]["success"] is False
    )  # FakeBackend has no physics; baseline cannot succeed
    assert (tmp_path / "a" / "episode-00.steps.npz").is_file()
    assert (tmp_path / "a" / "episode-00.actions.jsonl").is_file()

    header = first.episodes_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "scene" in header and "seed" in header and "success" in header and "proxy_hit" in header


def test_eval_cli_rejects_unknown_pipeline_and_test_split(tmp_path: Path) -> None:
    runner = CliRunner()
    unknown = runner.invoke(app, ["eval", "--pipeline", "vla_only", "--backend", "fake"])
    assert unknown.exit_code != 0
    assert "eval-vla" in unknown.output

    frozen_test = runner.invoke(
        app,
        ["eval", "--pipeline", "baseline", "--split", "test", "--backend", "fake"],
    )
    assert frozen_test.exit_code != 0  # Typer rejects the disallowed choice


def test_eval_cli_fake_smoke_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "eval-out"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--pipeline",
            "scripted",
            "--split",
            "train",
            "--n",
            "1",
            "--seed",
            "0",
            "--backend",
            "fake",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "success=" in result.output
    assert (out / "episodes.csv").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["pipeline"] == "baseline"  # scripted alias
    assert summary["measurement"] == "fake"
    assert summary["n_episodes"] == 1
    assert summary["pairing_unit"] == "(scene, seed)"
    assert summary["collision_proxy"]["label"].startswith("PROXY")
