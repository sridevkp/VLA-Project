"""Short isolated real SmolVLA training/resume diagnostic; approved payload stays immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]], saves: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate unchanged campaign cost from completed updates, excluding warmup per process."""
    if not rows or any(not r["actual_optimizer_update"] for r in rows):
        raise ValueError("throughput requires completed optimizer updates")
    steady = [r for r in rows if r["update_in_process"] > 2]
    if not steady or not saves:
        raise ValueError("need post-warmup updates and checkpoint timing")
    seconds = statistics.mean(r["update_s"] + r["dataload_s"] for r in steady)
    checkpoint_s = max(r["write_and_integrity_s"] for r in saves)
    return {
        "observed_optimizer_updates": len(rows),
        "steady_steps_per_s": 1 / seconds,
        "estimated_20000_steps_hours": (20000 * seconds + 20 * checkpoint_s) / 3600,
        "estimated_1000_step_interval_minutes": (1000 * seconds + checkpoint_s) / 60,
        "max_checkpoint_write_and_integrity_s": checkpoint_s,
        "estimated_20_checkpoint_bytes": 20 * max(r.get("checkpoint_bytes", 0) for r in saves),
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
        "estimate_excludes_download_install_and_session_restart": True,
        "short_sample_not_session_guarantee": True,
        "diagnostic_checkpoint_cadence_only": True,
    }


def worker(dataset: Path, out: Path, updates: int, phase: int) -> None:
    """Use installed LeRobot main/update/save/load, observing actual optimizer hooks."""
    import torch
    import yaml
    from lerobot.scripts import lerobot_train as installed

    import vlarobot.vla.training as training
    import vlarobot.vla.training_evidence as evidence

    if not torch.cuda.is_available():
        raise RuntimeError("T4 preflight requires authorized CUDA; CPU fallback is forbidden")
    config = yaml.safe_load(Path("configs/training.yaml").read_text(encoding="utf-8"))
    if config != {"steps": 20000, "batch_size": 8, "seed": 0, "save_freq": 1000}:
        raise ValueError("preflight requires the accepted frozen training configuration")
    target = updates * phase
    trackers: list[Any] = []
    rows: list[dict[str, Any]] = []
    saves: list[dict[str, Any]] = []
    original_evidence = evidence.OptimizerStepEvidence
    original_update = installed.update_policy
    original_save = installed.save_checkpoint
    original_load = installed.load_training_state
    original_mark = training.mark_checkpoint_complete

    class DiagnosticEvidence(original_evidence):
        def __init__(self, optimizer: Any, path: Path, **kwargs: Any) -> None:
            kwargs["identity"] = {**kwargs["identity"], "run_kind": "smolvla-t4-preflight"}
            kwargs["record_every"] = 1
            super().__init__(optimizer, path, **kwargs)
            trackers.append(self)

    def observe_update(*args: Any, **kwargs: Any) -> Any:
        tracker = trackers[-1]
        before = tracker.calls
        torch.cuda.synchronize()
        if before == 0:
            if phase == 2:
                from safetensors import safe_open

                checkpoint = training.latest_complete_checkpoint(out)
                accelerator = kwargs["accelerator"]
                policy = accelerator.unwrap_model(args[1], keep_fp32_wrapper=True)
                state = policy.state_dict()
                with safe_open(
                    checkpoint / "pretrained_model/model.safetensors", framework="pt"
                ) as f:
                    for name in f.keys():
                        if not torch.equal(state[name].detach().cpu(), f.get_tensor(name)):
                            raise ValueError("restored model tensor differs: " + name)
                write_json(out / "t4-model-restored.json", {"all_saved_model_tensors_equal": True})
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        result = original_update(*args, **kwargs)
        torch.cuda.synchronize()
        if tracker.calls != before + 1:
            raise RuntimeError("installed update did not complete exactly one optimizer update")
        metrics = result[0]
        loss = float(metrics.loss.val)
        if not math.isfinite(loss):
            raise ValueError("nonfinite training loss")
        row = {
            "phase": phase,
            "global_step": tracker.start_step + tracker.calls,
            "update_in_process": tracker.calls,
            "actual_optimizer_update": True,
            "loss": loss,
            "update_s": perf_counter() - started,
            "dataload_s": float(metrics.dataloading_s.val),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        rows.append(row)
        with (out / "t4-updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return result

    def observe_save(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        result = original_save(*args, **kwargs)
        saves.append({"step": kwargs["step"], "started": started})
        return result

    def observe_load(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        result = original_load(*args, **kwargs)
        if result[0] != updates:
            raise ValueError("restored global step differs from first diagnostic checkpoint")
        from lerobot.datasets.utils import flatten_dict
        from lerobot.utils.random_utils import serialize_rng_state
        from safetensors.torch import load_file

        checkpoint = Path(args[0])
        saved_rng = load_file(checkpoint / "training_state/rng_state.safetensors")
        actual_rng = flatten_dict(serialize_rng_state())
        if set(saved_rng) != set(actual_rng) or any(
            not torch.equal(saved_rng[k], actual_rng[k].cpu()) for k in saved_rng
        ):
            raise ValueError("restored Python/NumPy/Torch/CUDA RNG differs from checkpoint")
        optimizer, scheduler = result[1:]
        if not optimizer.state or scheduler is None:
            raise ValueError("optimizer or scheduler restoration missing")
        saved_optimizer = load_file(checkpoint / "training_state/optimizer_state.safetensors")
        actual_optimizer = flatten_dict({"state": optimizer.state_dict()["state"]})
        if set(saved_optimizer) != set(actual_optimizer) or any(
            not torch.equal(saved_optimizer[k], actual_optimizer[k].cpu()) for k in saved_optimizer
        ):
            raise ValueError("restored optimizer tensors differ from checkpoint")
        saved_scheduler = json.loads(
            (checkpoint / "training_state/scheduler_state.json").read_text(encoding="utf-8")
        )
        if json.loads(json.dumps(scheduler.state_dict())) != saved_scheduler:
            raise ValueError("restored scheduler differs from checkpoint")
        write_json(
            out / "t4-restore.json",
            {
                "restored_global_step": result[0],
                "installed_restore_completed": True,
                "rng_tensors_equal": True,
                "optimizer_tensors_equal": True,
                "scheduler_state_equal": True,
                "optimizer_state_entries": len(optimizer.state),
                "scheduler_state": scheduler.state_dict(),
                "restore_wall_s": perf_counter() - started,
                "checkpoint_sha_verified_before_launch": True,
            },
        )
        return result

    class DiagnosticBoundary(Exception):
        pass

    def finish_checkpoint(path: Path) -> Path:
        result = original_mark(path)
        elapsed = perf_counter() - saves[-1].pop("started")
        saves[-1]["write_and_integrity_s"] = elapsed
        saves[-1]["checkpoint_bytes"] = sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file()
        )
        write_json(out / f"t4-phase-{phase}-saves.json", saves)
        if int(path.name) >= target:
            raise DiagnosticBoundary
        return result

    evidence.OptimizerStepEvidence = DiagnosticEvidence
    installed.update_policy = observe_update
    installed.save_checkpoint = observe_save
    installed.load_training_state = observe_load
    training.mark_checkpoint_complete = finish_checkpoint
    # Diagnostic-only save cadence; model, batch, seed, loss, scheduler and 20k horizon unchanged.
    # Production configs and output are never edited or resumed from this diagnostic directory.
    try:
        training.launch_approved_training(dataset, out, **{**config, "save_freq": updates})
    except DiagnosticBoundary:
        pass
    checkpoint = training.latest_complete_checkpoint(out)
    if checkpoint is None or int(checkpoint.name) != target:
        raise RuntimeError("diagnostic boundary has no complete hash-verified checkpoint")
    write_json(
        out / f"t4-phase-{phase}.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "phase": phase,
            "final_global_step": target,
            "updates_in_process": len(rows),
            "complete_checkpoint": str(checkpoint),
            "test_split_used": False,
            "diagnostic_only_not_production_training": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--worker-phase", type=int, choices=(1, 2))
    parser.add_argument("--available-session-hours", type=float)
    args = parser.parse_args()
    if args.updates < 4 or args.updates >= 1000:
        raise ValueError("diagnostic length must be between 4 and 999 updates per process")
    if args.worker_phase:
        worker(args.dataset.resolve(), args.out.resolve(), args.updates, args.worker_phase)
        return
    if args.out.exists():
        raise FileExistsError("preserve prior preflight; select a new versioned output")
    # LeRobot owns creation of the training output, so supervisor evidence lives beside it.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frozen = Path("configs/training.yaml")
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    started = perf_counter()
    for phase in (1, 2):
        with args.out.with_name(args.out.name + f".phase-{phase}.log").open(
            "x", encoding="utf-8"
        ) as log:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--dataset",
                        str(args.dataset.resolve()),
                        "--out",
                        str(args.out.resolve()),
                        "--updates",
                        str(args.updates),
                        "--worker-phase",
                        str(phase),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError:
                log.flush()
                log_path = Path(log.name)
                print(
                    "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:]), flush=True
                )
                raise
    rows = [json.loads(line) for line in (args.out / "t4-updates.jsonl").read_text().splitlines()]
    saves = [
        r
        for phase in (1, 2)
        for r in json.loads((args.out / f"t4-phase-{phase}-saves.json").read_text())
    ]
    if [r["global_step"] for r in rows] != list(range(1, args.updates * 2 + 1)):
        raise ValueError("diagnostic optimizer updates are not complete and contiguous")
    if not (args.out / "t4-model-restored.json").is_file():
        raise ValueError("restored model has not been compared with saved checkpoint")
    report = summarize(rows, saves)
    report["restore_verified"] = json.loads((args.out / "t4-restore.json").read_text())
    report["preflight_complete"] = True
    report.update(
        preflight_wall_s=perf_counter() - started,
        production_training_started=False,
        diagnostic_only=True,
        production_config_sha256=before,
    )
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != before:
        raise ValueError("frozen training configuration changed during preflight")
    if args.available_session_hours is not None:
        report["available_session_hours"] = args.available_session_hours
        report["estimated_full_run_fits_session"] = (
            report["estimated_20000_steps_hours"] < args.available_session_hours
        )
        report["estimated_checkpoint_interval_fits_session"] = (
            report["estimated_1000_step_interval_minutes"] / 60 < args.available_session_hours
        )
    write_json(args.out / "t4-preflight-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
