"""Select complete resumed runs or preserve unrecoverable partial runs under a new version."""

from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path

from vlarobot.vla.training import latest_complete_checkpoint


def _active_lock(output: Path) -> Path:
    return output.parent / (output.name + ".active.json")


def training_is_active(output: Path) -> bool:
    lock = _active_lock(output)
    if not lock.exists():
        return False
    value = json.loads(lock.read_text(encoding="utf-8"))
    psutil = import_module("psutil")
    try:
        process = psutil.Process(value["pid"])
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and process.create_time() == value["process_create_time"]
        )
    except psutil.NoSuchProcess:
        return False


def select_training_output(base: Path) -> Path:
    """Never overwrite partial output; restore highest complete checkpoint across owned restarts."""
    candidates = [
        base,
        *sorted(p for p in base.parent.glob(base.name + ".restart-*") if p.is_dir()),
    ]
    complete = []
    for path in candidates:
        if training_is_active(path):
            raise RuntimeError(f"training is already active: {path}")
        checkpoint = latest_complete_checkpoint(path)
        if checkpoint is not None:
            complete.append((int(checkpoint.name), path))
    if complete:
        return max(complete, key=lambda item: (item[0], item[1].name))[1]
    if not base.exists():
        return base
    index = 1
    while True:
        path = base.with_name(base.name + f".restart-{index:04d}")
        if not path.exists():
            return path
        index += 1


def claim_training_run(output: Path) -> Path:
    """Host-process identity protects against duplicate launches after notebook reconnection."""
    psutil = import_module("psutil")
    lock = _active_lock(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        if training_is_active(output):
            raise RuntimeError(f"training is already active: {output}")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "process_create_time": psutil.Process().create_time(),
                    "output": str(output),
                }
            )
        )
    return lock
