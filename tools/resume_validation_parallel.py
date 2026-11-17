"""Preserve serial work and finish fixed VALIDATION pairs in disjoint fresh-session workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--handover", action="store_true")
    parser.add_argument("--webots", required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    if Path(sys.prefix).resolve() != (project / ".venv").resolve():
        raise RuntimeError("physical reference evaluation must use the frozen project .venv")
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    out = base / "campaign"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = "calibration/urdf-base-train-v1.yaml"
    sys.path.insert(0, str(project / "src"))
    from vlarobot.backends.fake import FakeBackend
    from vlarobot.eval.runner import (
        EvalRunConfig,
        _completed_recording,
        build_episode_plan,
        run_evaluation,
    )
    from vlarobot.execution.demo_collection import write_json
    from vlarobot.execution.episode_runner import launch_webots_session
    from vlarobot.execution.scripted_baseline import ScriptedBaselinePolicy

    plan = build_episode_plan("validation", 0)
    from importlib.metadata import version

    runtime = {
        "python": sys.version,
        "prefix": sys.prefix,
        "numpy": version("numpy"),
        "ikpy": version("ikpy"),
        "scipy": version("scipy"),
    }
    if (runtime["numpy"], runtime["ikpy"], runtime["scipy"]) != ("2.4.6", "4.4.4", "1.17.1"):
        raise RuntimeError("frozen reference numerical versions differ")
    write_json(base / "runtime-provenance.json", runtime)
    frozen = json.loads(
        (project / "runs/eval/baseline-train-s0-fresh-v2-registered/provenance.json").read_text(
            encoding="utf-8"
        )
    )
    for name, digest in frozen["files_sha256"].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen source differs: " + name)
    if args.handover:
        import psutil

        watch_pid = int((base / "watch.pid").read_text(encoding="utf-8"))
        if psutil.pid_exists(watch_pid):
            process = psutil.Process(watch_pid)
            if str(base / "watch.py") not in process.cmdline():
                raise RuntimeError("watch ownership mismatch")
            process.terminate()
            process.wait(timeout=15)
        parent = psutil.Process(int((base / "supervisor.pid").read_text(encoding="utf-8")))
        owned = parent.children(recursive=True)
        cli = [p for p in owned if "vlarobot.cli" in p.cmdline() and "eval" in p.cmdline()]
        if len(cli) != 1:
            raise RuntimeError("ambiguous serial campaign ownership")
        cli[0].send_signal(signal.SIGINT)
        try:
            cli[0].wait(timeout=45)
        except psutil.TimeoutExpired:
            cli[0].terminate()
            cli[0].wait(timeout=15)
        for p in reversed(owned):
            try:
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    p.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.wait(timeout=15)
    for e in plan:
        if _completed_recording(out, e, "baseline") is not None:
            continue
        saved = base / "interruptions" / "serial-handover" / f"episode-{e.index:02d}"
        paths = list(out.glob(f"episode-{e.index:02d}.*")) + list(
            out.glob(f"eval-baseline-validation-s0-ep{e.index:02d}.*")
        )
        frames = out / "frames" / f"episode-{e.index:02d}"
        if frames.exists():
            paths.append(frames)
        for path in paths:
            saved.mkdir(parents=True, exist_ok=True)
            if (saved / path.name).exists():
                raise FileExistsError("preserve earlier interrupted artifacts")
            path.rename(saved / path.name)
    pending = [e for e in plan if _completed_recording(out, e, "baseline") is None]
    started = perf_counter()

    def config(folder):
        return EvalRunConfig(
            out_dir=folder,
            pipeline="baseline",
            split="validation",
            base_seed=0,
            backend="webots",
            measurement="measured",
            state_source="oracle",
            record_frames=True,
            frame_hz=1.0,
        )

    def progress(status):
        rows = []
        for e in plan:
            if _completed_recording(out, e, "baseline") is None:
                continue
            m = json.loads((out / f"episode-{e.index:02d}.meta.json").read_text(encoding="utf-8"))
            rows.append({"index": e.index, "scene": e.scene.name, "seed": e.seed, **m["outcome"]})
        write_json(
            base / "parallel-progress.json",
            {
                "status": status,
                "split": "validation",
                "completed": len(rows),
                "planned": 18,
                "successes": sum(r["success"] for r in rows),
                "episodes": rows,
                "workers": args.workers,
                "test_accessed": False,
                "wall_s_parallel": perf_counter() - started,
            },
        )

    def worker(entry):
        folder = base / "parallel-episodes" / f"episode-{entry.index:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        with launch_webots_session(
            webots=args.webots,
            world=project / "worlds/factory.wbt",
            trace_dir=folder,
            trace_prefix=f"val-ep{entry.index:02d}",
            headless=False,
        ) as session:
            run_evaluation(
                [entry],
                session.client,
                config(folder),
                policy_factory=lambda scene: ScriptedBaselinePolicy(
                    scene, trace_path=folder / "policy.jsonl"
                ),
            )
        if _completed_recording(folder, entry, "baseline") is None:
            raise ValueError("incomplete episode preserved: " + str(folder))
        prefix = f"episode-{entry.index:02d}"
        for path in folder.glob(prefix + ".*"):
            if path.name.endswith(".meta.json"):
                continue
            if (out / path.name).exists():
                raise FileExistsError("duplicate episode publication")
            shutil.copyfile(path, out / path.name)
        if (folder / "frames" / prefix).exists():
            shutil.copytree(folder / "frames" / prefix, out / "frames" / prefix)
        meta = out / (prefix + ".meta.json")
        shutil.copyfile(folder / meta.name, meta.with_suffix(".tmp"))
        meta.with_suffix(".tmp").replace(meta)
        return entry.index

    progress("running")
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, e): e for e in pending}
        for future in as_completed(futures):
            try:
                print("completed", future.result(), flush=True)
            except Exception as error:
                errors.append({"index": futures[future].index, "error": repr(error)})
            progress("running")
    if errors:
        write_json(base / "parallel-errors.json", errors)
        progress("failed-preserving-evidence")
        raise RuntimeError("incomplete validation recordings retained")

    def no_replacement(scene):
        raise RuntimeError("missing real recording; no synthetic replacement")

    from dataclasses import replace

    report = run_evaluation(
        plan, FakeBackend(), replace(config(out), resume=True), policy_factory=no_replacement
    )
    write_json(base / "completed-validation.json", report.summary)
    progress("completed")


if __name__ == "__main__":
    main()
