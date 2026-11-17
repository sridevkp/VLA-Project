"""Owned-process collection recovery with immutable attempt indices and unchanged admission."""

from __future__ import annotations

import json
import subprocess
from importlib import import_module
from pathlib import Path
from time import sleep, time
from typing import Any

from vlarobot.execution.collection_health import inspect_collection
from vlarobot.execution.demo_collection import write_json


def matching_collectors(psutil: Any, collection: Path) -> list[Any]:
    """Only the exact collector module/output identity authorizes process recovery."""
    matches = []
    for process in psutil.process_iter(["cmdline"]):
        try:
            args = process.info["cmdline"] or []
            if (
                "vlarobot.cli" in args
                and "collect-demos" in args
                and "--out" in args
                and Path(args[args.index("--out") + 1]).resolve() == collection.resolve()
                and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                matches.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return matches


def watch_collection(
    collection: Path,
    state_dir: Path,
    command_file: Path,
    *,
    interval_s: float = 60,
    stale_s: float = 600,
) -> None:
    """Recover an exited/stalled owned collector; preserve interrupted raw attempts on resume."""
    if interval_s <= 0 or stale_s <= interval_s:
        raise ValueError("invalid watchdog cadence")
    psutil = import_module("psutil")
    state_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads(command_file.read_text(encoding="utf-8"))
    if (
        "collect-demos" not in original
        or Path(original[original.index("--out") + 1]).resolve() != collection.resolve()
    ):
        raise ValueError("watchdog command must target this exact collection")
    command = original if "--resume" in original else [*original, "--resume"]
    recoveries = max(
        (int(p.stem.split("-")[-1]) for p in state_dir.glob("recovery-*.log")), default=0
    )
    while True:
        try:
            health = inspect_collection(collection, state_dir / "integrity.json")
            state = json.loads((collection / "collection-status.json").read_text(encoding="utf-8"))
            if state["status"] == "attempt_limit":
                write_json(
                    state_dir / "status.json",
                    {
                        "status": "attempt-limit",
                        "health": health,
                        "admission_requirements_unchanged": True,
                    },
                )
                return
            if health["admitted"] >= state["target"]:
                write_json(
                    state_dir / "status.json",
                    {
                        "status": "target-complete",
                        "admitted": health["admitted"],
                        "recoveries": recoveries,
                    },
                )
                return
            owners = matching_collectors(psutil, collection)
            if len(owners) > 1:
                raise RuntimeError(
                    "multiple collectors for same output; refusing ambiguous recovery"
                )
            completed_attempts = {
                json.loads(line)["attempt"]
                for line in (collection / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            }
            active_logs = [
                p / "episode.log"
                for p in (collection / "attempts").glob("attempt-*")
                if not (p / "result.json").exists()
                and (p / "episode.log").exists()
                and int(p.name.split("-")[-1]) not in completed_attempts
            ]
            stalled = bool(
                owners
                and active_logs
                and all(time() - p.stat().st_mtime > stale_s for p in active_logs)
            )
            if stalled:
                owner = owners[0]
                descendants = owner.children(recursive=True)
                # All killed processes are verified descendants of this exact task-owned CLI.
                for process in [owner, *descendants]:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs([owner, *descendants], timeout=10)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=10)
                owners = []
            if not owners:
                recoveries += 1
                log_path = state_dir / f"recovery-{recoveries:03d}.log"
                if log_path.exists():
                    raise FileExistsError("preserve prior watchdog recovery history")
                with log_path.open("x", encoding="utf-8") as log:
                    subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                sleep(1)
            write_json(
                state_dir / "status.json",
                {
                    "status": "monitoring",
                    "admitted": health["admitted"],
                    "attempted": health["attempted"],
                    "admission_rate": health["admission_rate"],
                    "verified_frames": health["verified_frames"],
                    "disk_free_bytes": health["disk_free_bytes"],
                    "recoveries": recoveries,
                    "admission_requirements_unchanged": True,
                },
            )
        except Exception as error:
            write_json(
                state_dir / "status.json",
                {
                    "status": "integrity-or-recovery-failure",
                    "error": f"{type(error).__name__}: {error}",
                    "training_approved": False,
                },
            )
            raise
        sleep(interval_s)
