"""Resume an existing frozen baseline with disjoint fresh-session episode workers."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import SETTINGS_PATH
from vlarobot.eval.runner import (
    EvalEpisode,
    EvalRunConfig,
    _completed_recording,
    build_episode_plan,
    run_evaluation,
)
from vlarobot.eval.settings import load_eval_settings
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import DEFAULT_WORLD, launch_webots_session
from vlarobot.execution.scripted_baseline import scripted_policy_factory


def resume_parallel_baseline(out: Path, *, webots: str, workers: int) -> dict[str, Any]:
    """Keep completed outcomes; publish each newly recorded episode atomically (meta last)."""
    if workers not in (1, 2, 3):
        raise ValueError("worker count must be 1, 2 or 3")
    if list(out.glob("*.pid")):
        raise ValueError("finish the serial episode/session before parallel resumption")
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    project = SETTINGS_PATH.parent.parent
    for name, expected in provenance["files_sha256"].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen campaign source changed: {name}")
    settings = load_eval_settings()
    plan = build_episode_plan("train", 0, settings=settings)
    existing = {
        entry.index for entry in plan if _completed_recording(out, entry, "baseline") is not None
    }
    pending = [entry for entry in plan if entry.index not in existing]
    started = perf_counter()

    def worker(entry: EvalEpisode) -> dict[str, Any]:
        work = out / "parallel-episodes" / f"episode-{entry.index:02d}"
        if work.exists():
            raise FileExistsError(
                f"preserve prior worker artifact; select a new work version: {work}"
            )
        work.mkdir(parents=True)
        with launch_webots_session(
            webots=webots,
            world=DEFAULT_WORLD,
            trace_dir=work,
            trace_prefix=f"eval-baseline-train-s0-ep{entry.index:02d}",
            headless=False,
        ) as session:
            episode_started = perf_counter()
            config = EvalRunConfig(
                out_dir=work,
                pipeline="baseline",
                split="train",
                base_seed=0,
                backend="webots",
                measurement="measured",
                state_source="oracle",
                record_frames=True,
                frame_hz=1.0,
            )
            run_evaluation(
                [entry],
                session.client,
                config,
                policy_factory=scripted_policy_factory,
                settings=settings,
                startup_s=session.startup_s,
            )
            duration = perf_counter() - episode_started
            startup = session.startup_s
        prefix = f"episode-{entry.index:02d}"
        if _completed_recording(work, entry, "baseline") is None:
            raise ValueError(f"episode {entry.index} has no complete physical recording")
        for suffix in ["steps.npz", "actions.jsonl", "predicted.jsonl", "contacts.jsonl"]:
            target = out / f"{prefix}.{suffix}"
            if target.exists():
                raise FileExistsError(
                    f"partial root artifact must be preserved before resume: {target}"
                )
            shutil.copyfile(work / target.name, target)
        if (work / "frames" / prefix).exists():
            shutil.copytree(work / "frames" / prefix, out / "frames" / prefix)
        meta = out / f"{prefix}.meta.json"
        shutil.copyfile(work / meta.name, meta.with_suffix(".tmp"))
        meta.with_suffix(".tmp").replace(meta)
        return {
            "index": entry.index,
            "seed": entry.seed,
            "scene": entry.scene.name,
            "startup_s": startup,
            "policy_wall_s": duration,
            "source_dir": str(work),
        }

    records = []
    write_json(
        out / "campaign-status.json",
        {
            "status": "running",
            "phase": "parallel-resume",
            "completed_before_resume": len(existing),
            "workers": workers,
        },
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, entry) for entry in pending]
            for future in futures:
                records.append(future.result())
                write_json(
                    out / "parallel-runtime.json",
                    {"episodes": records, "resume_wall_s": perf_counter() - started},
                )
        if any(_completed_recording(out, e, "baseline") is None for e in plan):
            raise ValueError("refusing aggregate without every frozen episode physically recorded")
        report = run_evaluation(
            plan,
            FakeBackend(),
            EvalRunConfig(
                out_dir=out,
                pipeline="baseline",
                split="train",
                base_seed=0,
                backend="webots",
                measurement="measured",
                state_source="oracle",
                record_frames=True,
                frame_hz=1.0,
                resume=True,
            ),
            policy_factory=scripted_policy_factory,
            settings=settings,
        )
        rate = report.summary["success"]["rate"]
        result = {
            "status": "completed",
            "phase": "parallel-resume",
            "episodes": 24,
            "success": report.summary["success"],
            "parallel_resume_wall_s": perf_counter() - started,
            "literal_vla_gate_threshold": min(0.4, 0.6 * rate),
            "vla_gate_result": "UNMEASURED: no VLA-only evaluation",
            "non_discriminating_gate_owner_decision_required": rate == 0,
        }
        write_json(out / "campaign-status.json", result)
        return result
    except Exception as error:
        write_json(
            out / "campaign-status.json",
            {
                "status": "failed",
                "phase": "parallel-resume",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
