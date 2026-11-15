"""One immutable development/validation pairing plan and metric implementation for all arms."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vlarobot.eval.metrics import paired_comparison
from vlarobot.eval.runner import EvalRunConfig, PolicyFactory, build_episode_plan, run_evaluation
from vlarobot.eval.settings import load_eval_settings
from vlarobot.risk.comparison import ARMS


def comparison_plan(split: str, seed: int) -> dict[str, Any]:
    """Never even generate TEST scene geometry during independent development."""
    if split not in ("train", "validation"):
        raise ValueError("independent development allows TRAIN/VALIDATION only; TEST is sealed")
    settings = load_eval_settings()
    if settings.arms != ARMS:
        raise ValueError("comparison arms differ from frozen contract")
    plan = build_episode_plan(split, seed)
    episodes = [{"index": e.index, "seed": e.seed, "scene": asdict(e.scene)} for e in plan]
    serialized = json.dumps(episodes, sort_keys=True).encode("utf-8")
    return {
        "arms": ARMS,
        "split": split,
        "base_seed": seed,
        "episodes": episodes,
        "pairing_sha256": hashlib.sha256(serialized).hexdigest(),
        "pairing_unit": "(scene, seed)",
        "scene_clustering_required": True,
        "test_accessed": False,
        "role_gate": "separate trained TRAIN VLA-only campaign",
    }


def run_development_comparison(
    split: str,
    seed: int,
    out: Path,
    *,
    factories: Mapping[str, PolicyFactory],
    backend_factory: Callable[[str], Any],
    measurement: str,
) -> dict[str, Any]:
    """Execute supplied genuine arm policies using the exact same recorder and metrics.

    Callers must label fake/synthetic experiments honestly. No implicit expert replacement,
    role decision or success result is manufactured if a trained policy is absent.
    """
    manifest = comparison_plan(split, seed)
    if set(factories) != set(ARMS):
        raise ValueError("all five explicit arm policies are required")
    if measurement not in ("measured", "simulated"):
        raise ValueError("explicit measured/simulated provenance required")
    if out.exists():
        raise FileExistsError("preserve existing comparison campaign")
    out.mkdir(parents=True)
    (out / "comparison-plan.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    plan = build_episode_plan(split, seed)
    reports = {}
    flags = {}
    for arm in ARMS:
        backend = backend_factory(arm)
        try:
            report = run_evaluation(
                plan,
                backend,
                EvalRunConfig(
                    out_dir=out / arm,
                    pipeline=arm,
                    split=split,
                    base_seed=seed,
                    backend="injected",
                    measurement=measurement,
                    state_source="explicit-policy",
                    record_frames=False,
                ),
                policy_factory=factories[arm],
            )
        finally:
            backend.stop()
        reports[arm] = report.summary
        flags[arm] = {
            (str(r["scene"]), int(str(r["seed"]))): bool(r["success"]) for r in report.rows
        }
        if len(flags[arm]) != len(plan):
            raise ValueError("duplicate scene/seed results")
    paired = {arm: asdict(paired_comparison(flags[ARMS[0]], flags[arm])) for arm in ARMS[1:]}
    result = {
        "measurement": measurement,
        "arms": reports,
        "paired_vs_baseline": paired,
        "pairing_sha256": manifest["pairing_sha256"],
        "role_gate": "UNMEASURED",
        "test_accessed": False,
    }
    (out / "comparison-summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
