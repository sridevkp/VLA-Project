"""Frozen-campaign guards (ADR-0029): sizes, arms, gate interpretation must not drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlarobot.eval.settings import load_eval_settings

REPO_EVAL_YAML = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"


def test_repo_eval_yaml_is_the_frozen_campaign() -> None:
    settings = load_eval_settings(REPO_EVAL_YAML)

    assert (settings.sizes.train, settings.sizes.validation, settings.sizes.test) == (8, 6, 10)
    assert settings.families == ("clear", "cluttered", "reach-limited")
    assert settings.arms == (
        "scripted_baseline",
        "vla_only",
        "vla_risk_veto",
        "vla_world_model",
        "full_system",
    )
    assert settings.vla_seeds_per_scene == 3
    assert settings.baseline_seeds_per_scene == 3
    assert settings.weeks == 4
    assert settings.deadline == "2026-12-12"
    assert settings.collision_proxy_max_displacement_m == pytest.approx(0.005)
    assert settings.collision_proxy_max_tilt_deg == pytest.approx(5.0)


def test_week2_gate_uses_literal_min_interpretation() -> None:
    gate = load_eval_settings(REPO_EVAL_YAML).gate

    # Literal `min` (owner answer, ADR-0029): the threshold is the WEAKER bound.
    assert gate.threshold(0.40) == pytest.approx(0.24)  # min(0.40, 0.60*0.40)
    assert gate.threshold(0.70) == pytest.approx(0.40)  # min(0.40, 0.42)
    assert gate.threshold(0.90) == pytest.approx(0.40)  # min(0.40, 0.54) — not 0.54

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gate.threshold(1.5)


def test_loader_rejects_drifted_sizes(tmp_path: Path) -> None:
    drifted = tmp_path / "eval.yaml"
    drifted.write_text(
        REPO_EVAL_YAML.read_text(encoding="utf-8").replace("train: 8", "train: 7"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen sizes"):
        load_eval_settings(drifted)


def test_loader_rejects_comparator_change(tmp_path: Path) -> None:
    drifted = tmp_path / "eval.yaml"
    drifted.write_text(
        REPO_EVAL_YAML.read_text(encoding="utf-8").replace("comparator: min", "comparator: max"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="literal 'min'"):
        load_eval_settings(drifted)
