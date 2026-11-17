"""Fail-closed, inert preflight for the production WorldMind composition.

The preflight checks configuration approval, binding registration, report/calibration provenance,
frozen VALIDATION provenance and a fresh output directory. It never launches Webots, solves IK,
runs inference, trains, actuates, scores a campaign or accesses TEST/TRAIN evidence. A runtime
geometry probe may be injected by the backend boundary; when absent, preflight is incomplete.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from vlarobot.eval.report_contract import load_definitions_candidate
from vlarobot.eval.runner import EvalEpisode, build_episode_plan
from vlarobot.eval.worldmind_calibration import CalibrationResult, evaluate_calibration
from vlarobot.eval.worldmind_config import (
    FROZEN_BASE_SEED,
    FROZEN_EPISODE_COUNT,
    FROZEN_SEEDS,
    FROZEN_SPLIT,
    FROZEN_STATE_SOURCE,
    WorldMindBindings,
    WorldMindConfig,
    load_worldmind_config,
    validate_worldmind_bindings,
)

RuntimeProbe = Callable[[WorldMindConfig, WorldMindBindings], Mapping[str, str]]


@dataclass(frozen=True)
class PreflightCheck:
    """One explicit gate with a pass/fail state and an evidence-backed detail."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class WorldMindPreflight:
    """Structured preflight result; ``ready`` is true only when every listed gate passes."""

    ready: bool
    checks: tuple[PreflightCheck, ...]
    config: WorldMindConfig | None
    plan: tuple[EvalEpisode, ...]
    calibration: CalibrationResult | None

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(f"{check.name}: {check.detail}" for check in self.checks if not check.passed)


def _check(name: str, action: Callable[[], str]) -> PreflightCheck:
    try:
        return PreflightCheck(name, True, action())
    except Exception as error:  # noqa: BLE001 - preflight reports every failure without escaping
        return PreflightCheck(name, False, f"{type(error).__name__}: {error}")


def run_worldmind_preflight(
    config_path: Path,
    out_dir: Path,
    *,
    split: str,
    backend: str,
    state_source: str,
    bindings: WorldMindBindings,
    runtime_probe: RuntimeProbe | None = None,
) -> WorldMindPreflight:
    """Collect all fail-closed gates; a missing or unverified gate blocks the pilot."""
    checks: list[PreflightCheck] = []
    checks.append(
        _check(
            "split",
            lambda: (
                "VALIDATION only"
                if split == FROZEN_SPLIT
                else (_ for _ in ()).throw(ValueError("TEST/TRAIN access is sealed"))
            ),
        )
    )
    checks.append(
        _check(
            "state_source",
            lambda: (
                "declared oracle"
                if state_source == FROZEN_STATE_SOURCE
                else (_ for _ in ()).throw(ValueError("only oracle state source is permitted"))
            ),
        )
    )
    checks.append(
        _check(
            "fresh_output",
            lambda: (
                "new output directory"
                if not out_dir.exists()
                else (_ for _ in ()).throw(FileExistsError(str(out_dir)))
            ),
        )
    )

    try:
        config = load_worldmind_config(config_path)
    except Exception as error:  # noqa: BLE001 - a blocked config must not escape preflight
        checks.append(PreflightCheck("config", False, f"{type(error).__name__}: {error}"))
        return WorldMindPreflight(False, tuple(checks), None, (), None)
    checks.append(PreflightCheck("config", True, f"status={config.status} split={config.split}"))

    checks.append(
        _check(
            "backend_declaration",
            lambda: (
                f"declared={config.backend.kind}"
                if config.backend.kind == backend
                else (_ for _ in ()).throw(ValueError("CLI backend differs from config"))
            ),
        )
    )
    checks.append(
        _check(
            "report_definitions",
            lambda: (
                "approved definitions and candidate reviewed"
                if load_definitions_candidate(
                    config.report.candidate_definitions, config.report.pdf
                )["review_status"]
                == "candidate"
                else (_ for _ in ()).throw(ValueError("candidate status changed"))
            ),
        )
    )

    def bindings_check() -> str:
        # validate_worldmind_bindings raises on any missing/invalid binding; a returned tuple
        # means every configured binding resolved to a registered object.
        validate_worldmind_bindings(config, bindings)
        return "all configured bindings registered"

    checks.append(_check("bindings", bindings_check))

    calibration_result: CalibrationResult | None = None

    def evaluate() -> str:
        nonlocal calibration_result
        evaluator = bindings.calibration_evaluators.get(config.calibration.evaluator_binding)
        if not callable(evaluator):
            raise ValueError(f"no registered evaluator: {config.calibration.evaluator_binding}")
        calibration_result = evaluate_calibration(config.calibration.candidate, evaluator)
        return (
            f"id={calibration_result.selected_candidate_id} "
            f"candidates={len(calibration_result.scores)}"
        )

    checks.append(_check("calibration", evaluate))

    def plan_check() -> str:
        plan = build_episode_plan(FROZEN_SPLIT, FROZEN_BASE_SEED, n=0, allow_test=False)
        if len(plan) != FROZEN_EPISODE_COUNT:
            raise ValueError("frozen VALIDATION episode count differs")
        if sorted({entry.seed for entry in plan}) != list(FROZEN_SEEDS):
            raise ValueError("frozen seed contract violated")
        pairs = [(entry.scene.name.rsplit("-s", 1)[0], entry.seed) for entry in plan]
        if len(set(pairs)) != FROZEN_EPISODE_COUNT:
            raise ValueError("frozen (scene, seed) pairs are not unique")
        return f"episodes={len(plan)} seeds={sorted({entry.seed for entry in plan})}"

    plan_check_result = _check("frozen_plan", plan_check)
    checks.append(plan_check_result)
    plan = (
        build_episode_plan(FROZEN_SPLIT, FROZEN_BASE_SEED, n=0, allow_test=False)
        if plan_check_result.passed
        else ()
    )

    def probe_check() -> str:
        if runtime_probe is None:
            raise ValueError("no backend runtime probe registered")
        values = runtime_probe(config, bindings)
        required = (
            "robot_config_sha256",
            "safety_config_sha256",
            "urdf_sha256",
            "corrected_registration_sha256",
            "pose_solver",
            "safety_limits",
            "waypoint_factory",
        )
        missing = [name for name in required if not values.get(name)]
        if missing:
            raise ValueError(f"incomplete runtime probe: {missing}")
        return ", ".join(f"{name}={values[name]}" for name in required)

    checks.append(_check("runtime_geometry", probe_check))

    def evidence_check() -> str:
        evidence = config.calibration.candidate.evidence
        return (
            f"evidence_id={evidence.evidence_id} episodes={len(evidence.episode_ids)} "
            f"sources={len(evidence.source_hashes)} sha256={evidence.sha256[:12]}"
        )

    checks.append(_check("calibration_evidence", evidence_check))
    ready = all(check.passed for check in checks)
    return WorldMindPreflight(ready, tuple(checks), config, plan, calibration_result)
