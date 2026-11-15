"""Strict read-back guards for recovery; no training, deletion or writes on import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def require_isolated(original: Path, recovery: Path) -> None:
    original, recovery = original.resolve(), recovery.resolve()
    if original == recovery or original in recovery.parents or recovery in original.parents:
        raise ValueError("recovery must not overlap the original production directory")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe manifest path: {name}")
    path = root.joinpath(*relative.parts)
    if root.is_symlink() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink in manifest path: {name}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty file: {path}")
    return path


def verify_checkpoint(checkpoint: Path, required: tuple[str, ...]) -> dict[str, Any]:
    marker = regular(checkpoint, "COMPLETE.json")
    marker_hash = sha256(marker)
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("schema_version") != 2 or record.get("step") != int(checkpoint.name):
        raise ValueError("checkpoint schema/step mismatch")
    sizes, hashes = record.get("files_and_sizes", {}), record.get("files_sha256", {})
    if not isinstance(sizes, dict) or not isinstance(hashes, dict):
        raise ValueError("invalid checkpoint inventory")
    if set(sizes) != set(hashes) or not set(required).issubset(hashes):
        raise ValueError("incomplete checkpoint manifest inventory")
    for name, expected in hashes.items():
        path = regular(checkpoint, name)
        if path.stat().st_size != sizes[name] or sha256(path) != expected:
            raise ValueError(f"checkpoint size/hash mismatch: {name}")
        if path.stat().st_size != sizes[name]:
            raise ValueError(f"checkpoint changed during validation: {name}")
    step = json.loads(regular(checkpoint, "training_state/training_step.json").read_text())
    if step.get("step") != record["step"] or sha256(marker) != marker_hash:
        raise ValueError("checkpoint step/manifest changed during validation")
    return record


def latest(output: Path, required: tuple[str, ...]) -> Path | None:
    candidates = sorted(
        (p for p in (output / "checkpoints").glob("*") if p.name.isdigit()),
        key=lambda p: int(p.name),
        reverse=True,
    )
    for checkpoint in candidates:
        try:
            verify_checkpoint(checkpoint, required)
        except (OSError, ValueError, TypeError, KeyError):
            continue
        return checkpoint
    return None


def verify_export(export: Path, required: tuple[str, ...], expected_step: int) -> dict[str, Any]:
    manifest = json.loads(regular(export, "export-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_step") != expected_step:
        raise ValueError("export is not the required final step")
    hashes = manifest.get("files_sha256", {})
    model_required = {
        n.removeprefix("pretrained_model/") for n in required if n.startswith("pretrained_model/")
    }
    if not model_required.issubset(hashes):
        raise ValueError("export manifest omits required model assets")
    for name, expected in hashes.items():
        if sha256(regular(export, name)) != expected:
            raise ValueError(f"export fingerprint mismatch: {name}")
    return manifest


def install_guards() -> None:
    """Patch only this recovery process; approved source archives remain unchanged."""
    from vlarobot.vla import checkpoint_export, training

    required = training.CHECKPOINT_FILES
    training.latest_complete_checkpoint = lambda output: latest(output, required)
    checkpoint_export.latest_complete_checkpoint = training.latest_complete_checkpoint
    original_mark = training.mark_checkpoint_complete

    def checked_mark(checkpoint: Path) -> Path:
        result = original_mark(checkpoint)
        verify_checkpoint(result, required)
        return result

    training.mark_checkpoint_complete = checked_mark
    original_export = checkpoint_export.export_complete_checkpoint

    def checked_export(output: Path, out: Path) -> Path:
        checkpoint = latest(output, required)
        if checkpoint is None or int(checkpoint.name) != 20000:
            raise ValueError("recovery export requires a physically verified step 20000")
        source = verify_checkpoint(checkpoint, required)
        result = original_export(output, out)
        manifest = verify_export(result, required, 20000)
        for name, digest in source["files_sha256"].items():
            if name.startswith("pretrained_model/"):
                if manifest["files_sha256"].get(name.removeprefix("pretrained_model/")) != digest:
                    raise ValueError(f"export differs from verified checkpoint: {name}")
        verify_checkpoint(checkpoint, required)
        checkpoint_export.require_trained_export(result)
        return result

    checkpoint_export.export_complete_checkpoint = checked_export
