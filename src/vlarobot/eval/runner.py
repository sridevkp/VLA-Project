"""Baseline evaluation loop: plan -> recorded episodes -> metrics -> CSV/JSON report.

ADR-0029 week 1 (design recorded in ADR-0033). This module is deliberately backend- and
policy-agnostic: it imports only ``core`` and ``eval`` modules (AGENTS rule 1). The caller
supplies a connected backend and a ``policy_factory``; ``execution.scripted_baseline`` provides
the scripted expert, and the CLI owns the Webots process lifetime.

Episode plan (seed semantics resolved from the frozen config and committed generator):
``configs/eval.yaml`` freezes 3 baseline seeds per scene, and the generator draws scene
geometry from the seed itself. The plan is therefore **seed-major**: for each seed
``base_seed + k`` it generates that seed's full split, so a train baseline is
``[seed, seed+1, seed+2] x 8 scenes = 24`` episodes. The pairing key is
``(scene_id, seed)`` where ``scene_id`` is the generator name without its ``-s<seed>`` suffix
(``eval.metrics.scene_id``); the same seed also perturbs geometry, so a paired unit is a
scene slot/family within one repetition, not a fixed geometry. ``docs/STATISTICS.md`` records
the consequences for McNemar and scene-level clustering.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from vlarobot.core.types import ScenarioSpec
from vlarobot.eval.metrics import (
    EpisodeMetrics,
    EpisodeRecord,
    compute_episode_metrics,
    load_episode,
    mean_std,
    metrics_to_row,
    scene_id,
    summarize_success,
    write_csv,
    write_json,
)
from vlarobot.eval.recorder import EpisodeRecorder, RecordingBackend
from vlarobot.eval.scenarios import SPLITS, ScenarioLibrary, generate_scenes
from vlarobot.eval.settings import EvalSettings, load_eval_settings

PIPELINE_ALIASES: dict[str, str] = {
    "baseline": "baseline",
    "scripted": "baseline",
    "scripted_baseline": "baseline",
    "vla_only": "vla_only",
    "vla_risk_veto": "vla_risk_veto",
    "vla_world_model": "vla_world_model",
    "full_system": "full_system",
}
IMPLEMENTED_PIPELINES = tuple(sorted(set(PIPELINE_ALIASES.values())))
_EPISODE_STEM = "episode-{index:02d}"


class EpisodePolicy(Protocol):
    """One arm's episode runner: drive ``backend`` and return the outcome payload."""

    def execute(self, backend: Any, seed: int) -> Mapping[str, Any]:
        """Run one episode; the returned mapping must contain ``success``."""


PolicyFactory = Callable[[ScenarioSpec], EpisodePolicy]


def canonical_pipeline(name: str) -> str:
    """Canonical pipeline name (``scripted`` is an alias of ``baseline``, ADR-0029)."""
    try:
        return PIPELINE_ALIASES[str(name)]
    except KeyError as error:
        raise ValueError(
            f"pipeline {name!r} is not implemented in week 1; "
            f"implemented: {', '.join(IMPLEMENTED_PIPELINES)} (alias: scripted)."
        ) from error


@dataclass(frozen=True)
class EvalEpisode:
    """One planned episode: a generated scene plus the seed that drives reset and geometry."""

    index: int
    scene: ScenarioSpec
    seed: int


def build_episode_plan(
    split: str,
    base_seed: int,
    *,
    n: int = 0,
    allow_test: bool = False,
    settings: EvalSettings | None = None,
    library: ScenarioLibrary | None = None,
) -> tuple[EvalEpisode, ...]:
    """Seed-major episode plan for one split (``n=0`` means every planned episode).

    ``test`` stays fail-closed (ADR-0029: untouched until week 4) unless ``allow_test=True``.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; available: {', '.join(SPLITS)}")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError("base_seed must be an integer.")
    if n < 0:
        raise ValueError("n must be >= 0 (0 = every planned episode).")
    resolved = settings or load_eval_settings()

    plan: list[EvalEpisode] = []
    for offset in range(resolved.baseline_seeds_per_scene):
        seed = base_seed + offset
        scenes = generate_scenes(
            split,
            seed,
            allow_test=allow_test,
            library=library,
            sizes=resolved.sizes,
        )
        for scene in scenes:
            plan.append(EvalEpisode(index=len(plan), scene=scene, seed=seed))
    if n > 0:
        if n > len(plan):
            raise ValueError(
                f"n={n} exceeds the {len(plan)} planned episodes for split {split!r} "
                f"({resolved.baseline_seeds_per_scene} seeds x "
                f"{len(plan) // resolved.baseline_seeds_per_scene} scenes)."
            )
        plan = plan[:n]
    return tuple(plan)


@dataclass(frozen=True)
class EvalRunConfig:
    """Where and how one eval run records its episodes and report."""

    out_dir: Path
    pipeline: str = "baseline"
    split: str = "train"
    base_seed: int = 0
    backend: str = "webots"
    measurement: str = "measured"
    state_source: str = "oracle"
    record_frames: bool = False
    frame_hz: float = 1.0
    resume: bool = False


@dataclass(frozen=True)
class EpisodeTiming:
    """Measured wall time for one episode (kept out of the metric report)."""

    index: int
    scene: str
    seed: int
    wall_s: float
    resumed: bool


@dataclass(frozen=True)
class EvalReport:
    """One run's written report plus the in-memory rows."""

    summary: dict[str, Any]
    rows: tuple[dict[str, object], ...]
    timings: tuple[EpisodeTiming, ...]
    plan_json: Path
    episodes_csv: Path
    summary_json: Path
    timings_json: Path
    total_wall_s: float


def run_evaluation(
    plan: Sequence[EvalEpisode],
    backend: Any,
    config: EvalRunConfig,
    *,
    policy_factory: PolicyFactory,
    settings: EvalSettings | None = None,
    startup_s: float = 0.0,
) -> EvalReport:
    """Run every planned episode, record it, and write the per-episode + summary report."""
    resolved = settings or load_eval_settings()
    pipeline = canonical_pipeline(config.pipeline)
    if not plan:
        raise ValueError("plan must contain at least one episode.")
    indices = [entry.index for entry in plan]
    if len(set(indices)) != len(indices):
        raise ValueError("plan indices must be unique.")
    if config.split not in SPLITS:
        raise ValueError(f"unknown split {config.split!r}; available: {', '.join(SPLITS)}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = EpisodeRecorder(
        out_dir,
        frame_hz=config.frame_hz,
        record_frames=config.record_frames,
    )

    rows: list[dict[str, object]] = []
    timings: list[EpisodeTiming] = []
    run_started = perf_counter()
    for entry in plan:
        key = _plan_scene_id(entry)
        family = str(entry.scene.metadata.get("family", ""))
        completed = _completed_recording(out_dir, entry, pipeline) if config.resume else None
        if completed is not None:
            metrics = compute_episode_metrics(completed, resolved)
            wall_s = 0.0
            resumed = True
        else:
            metrics, wall_s = _run_recorded_episode(
                entry,
                backend,
                recorder,
                config,
                pipeline,
                resolved,
                policy_factory,
            )
            resumed = False
        row = metrics_to_row(metrics)
        row["family"] = family
        row["split"] = config.split
        rows.append(row)
        timings.append(
            EpisodeTiming(
                index=entry.index,
                scene=key,
                seed=entry.seed,
                wall_s=wall_s,
                resumed=resumed,
            )
        )
    total_wall_s = perf_counter() - run_started

    plan_json = out_dir / "plan.json"
    episodes_csv = out_dir / "episodes.csv"
    summary_json = out_dir / "summary.json"
    timings_json = out_dir / "timings.json"
    summary = _build_summary(rows, plan, config, resolved, pipeline)
    write_json(
        {
            "pipeline": pipeline,
            "split": config.split,
            "base_seed": config.base_seed,
            "episodes": [
                {"index": entry.index, "seed": entry.seed, "scene": asdict(entry.scene)}
                for entry in plan
            ],
        },
        plan_json,
    )
    write_csv(rows, episodes_csv)
    write_json(summary, summary_json)
    write_json(
        {
            "startup_s": startup_s,
            "total_wall_s": total_wall_s,
            "episodes": [asdict(timing) for timing in timings],
        },
        timings_json,
    )
    return EvalReport(
        summary=summary,
        rows=tuple(rows),
        timings=tuple(timings),
        plan_json=plan_json,
        episodes_csv=episodes_csv,
        summary_json=summary_json,
        timings_json=timings_json,
        total_wall_s=total_wall_s,
    )


def _run_recorded_episode(
    entry: EvalEpisode,
    backend: Any,
    recorder: EpisodeRecorder,
    config: EvalRunConfig,
    pipeline: str,
    settings: EvalSettings,
    policy_factory: PolicyFactory,
) -> tuple[EpisodeMetrics, float]:
    """Record one episode; an exception becomes a recorded failure row (run continues)."""
    started = perf_counter()
    paths = recorder.open_episode(entry.index, entry.seed, pipeline, config.state_source)
    try:
        policy = policy_factory(entry.scene)
        result = policy.execute(RecordingBackend(backend, recorder), entry.seed)
        if not isinstance(result, Mapping) or "success" not in result:
            raise ValueError("policy outcome must be a mapping with a 'success' key.")
        recorder.record_outcome(dict(result))
        paths = recorder.close_episode()
        metrics = compute_episode_metrics(load_episode(paths.meta), settings)
    except Exception as error:  # noqa: BLE001 - one bad episode must not lose the run
        if recorder.is_open:
            recorder.record_outcome(
                {
                    "success": False,
                    "failure_state": "EXCEPTION",
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
            recorder.close_episode()
        metrics = _exception_metrics(entry, pipeline, error)
    return metrics, perf_counter() - started


def _exception_metrics(entry: EvalEpisode, pipeline: str, error: Exception) -> EpisodeMetrics:
    """Metrics row for an episode that raised before producing a usable recording."""
    return EpisodeMetrics(
        scene=_plan_scene_id(entry),
        seed=entry.seed,
        arm=pipeline,
        success=False,
        failure_state="EXCEPTION",
        failure_reason=f"{type(error).__name__}: {error}",
        evaluation_failure_category="EXCEPTION",
        final_target_distance_m=None,
        can_height_error_m=None,
        can_tilt_deg=None,
        rejections=None,
        proxy_hit=False,
        proxy_max_displacement_m=None,
        proxy_max_tilt_deg=None,
        contact_steps=0,
        contact_nodes=(),
        fk_min_clearance_m=None,
    )


def _plan_scene_id(entry: EvalEpisode) -> str:
    """Pairing identity of a planned scene (generator name without its seed suffix)."""
    return scene_id({"scenario": {"name": entry.scene.name}, "reset_seed": entry.seed})


def _completed_recording(out_dir: Path, entry: EvalEpisode, pipeline: str) -> EpisodeRecord | None:
    """Load a finished recording for ``--resume`` after checking it belongs to this plan.

    A missing, unloadable/partial, or EXCEPTION/incomplete recording is treated as not
    completed so the episode is re-run (never double-counted). Only a loadable, plan-matching
    recording with a usable outcome is reused.
    """
    meta_path = out_dir / f"{_EPISODE_STEM.format(index=entry.index)}.meta.json"
    if not meta_path.is_file():
        return None
    try:
        record = load_episode(meta_path)
    except Exception:  # noqa: BLE001 - partial/corrupt previous attempt: rerun
        return None
    meta = record.meta
    outcome = meta.get("outcome")
    if not isinstance(outcome, Mapping):
        return None
    if meta.get("scenario") is None or outcome.get("failure_state") == "EXCEPTION":
        # The previous attempt raised before the episode was usable; run it again.
        return None
    if meta.get("pipeline") != pipeline:
        raise ValueError(
            f"{meta_path}: recorded pipeline {meta.get('pipeline')!r} != {pipeline!r}."
        )
    if meta.get("reset_seed") != entry.seed:
        raise ValueError(
            f"{meta_path}: recorded reset_seed {meta.get('reset_seed')!r} != {entry.seed}."
        )
    scenario = meta.get("scenario")
    name = scenario.get("name") if isinstance(scenario, Mapping) else None
    if name != entry.scene.name:
        raise ValueError(f"{meta_path}: recorded scene {name!r} != {entry.scene.name!r}.")
    return record


def _build_summary(
    rows: Sequence[Mapping[str, object]],
    plan: Sequence[EvalEpisode],
    config: EvalRunConfig,
    settings: EvalSettings,
    pipeline: str,
) -> dict[str, Any]:
    """Aggregate the episode rows (scene-level variation preserved, no single-run claims)."""
    flags = [bool(row["success"]) for row in rows]
    success = summarize_success(flags)

    rows_by_scene: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        rows_by_scene.setdefault(str(row["scene"]), []).append(row)
    plan_by_scene: dict[str, list[EvalEpisode]] = {}
    for entry in plan:
        plan_by_scene.setdefault(_plan_scene_id(entry), []).append(entry)

    per_scene: list[dict[str, Any]] = []
    for name, entries in plan_by_scene.items():
        scene_rows = rows_by_scene.get(name, [])
        scene_success = summarize_success([bool(row["success"]) for row in scene_rows])
        per_scene.append(
            {
                "scene": name,
                "family": str(entries[0].scene.metadata.get("family", "")),
                "n": scene_success.total,
                "seeds": sorted({entry.seed for entry in entries}),
                "successes": scene_success.successes,
                "rate": scene_success.rate,
                "ci_low": scene_success.ci_low,
                "ci_high": scene_success.ci_high,
            }
        )

    raw_failures = Counter(
        str(row.get("failure_state")) if row.get("failure_state") else "none"
        for row in rows
        if not row["success"]
    )
    failure_categories = Counter(
        str(row.get("evaluation_failure_category") or "task_unsuccessful")
        for row in rows
        if not row["success"]
    )
    proxy_hits = sum(1 for row in rows if row["proxy_hit"])
    contact_episodes = sum(1 for row in rows if int(cast(int, row["contact_steps"]) or 0) > 0)
    fk_measured = sum(1 for row in rows if row["fk_min_clearance_m"] is not None)
    return {
        "schema_version": 1,
        "pipeline": pipeline,
        "split": config.split,
        "backend": config.backend,
        "measurement": config.measurement,
        "state_source": config.state_source,
        "base_seed": config.base_seed,
        "seeds": sorted({entry.seed for entry in plan}),
        "n_episodes": len(rows),
        "pairing_unit": "(scene, seed)",
        "success": {
            "successes": success.successes,
            "total": success.total,
            "rate": success.rate,
            "ci_low": success.ci_low,
            "ci_high": success.ci_high,
        },
        "week2_gate": {
            "rule": "VLA-only success >= min(0.40, 0.60 * scripted baseline rate)",
            "baseline_rate": success.rate if pipeline == "baseline" else None,
            "threshold": settings.gate.threshold(success.rate) if pipeline == "baseline" else None,
        },
        "per_scene": per_scene,
        "failure_states": dict(sorted(failure_categories.items())),
        "raw_failure_states": dict(sorted(raw_failures.items())),
        "collision_proxy": {
            "label": "PROXY: object displacement >5 mm or tilt >5 deg (NOT collision detection)",
            "episodes_with_hit": proxy_hits,
            "episodes": len(rows),
            "episodes_with_actual_contact": contact_episodes,
            "contact_steps_total": sum(int(cast(int, row["contact_steps"]) or 0) for row in rows),
        },
        "continuous": {
            "final_target_distance_m": _continuous_stats(
                [row["final_target_distance_m"] for row in rows]
            ),
            "can_height_error_m": _continuous_stats([row["can_height_error_m"] for row in rows]),
            "can_tilt_deg": _continuous_stats([row["can_tilt_deg"] for row in rows]),
        },
        "rejections_total": sum(int(cast(int, row["rejections"]) or 0) for row in rows),
        "fk_clearance": {
            "measured_episodes": fk_measured,
            "note": (
                "standalone FK geometry measurement (ADR-0032); 0 measured means no FK-point "
                "source was wired for this run"
            ),
        },
    }


def _continuous_stats(values: Sequence[object]) -> dict[str, Any]:
    """Mean/sample-std over finite values; ``n=0`` instead of inventing a number."""
    floats = [float(cast(float, value)) for value in values if value is not None]
    finite = [value for value in floats if math.isfinite(value)]
    if not finite:
        return {"n": 0, "mean": None, "std": None}
    mean, std = mean_std(finite)
    return {"n": len(finite), "mean": mean, "std": None if not math.isfinite(std) else std}
