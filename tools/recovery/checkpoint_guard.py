"""Strict read-back guards for recovery; no training, deletion or writes on import."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


def resolve_project_root(root: Path, expected: Path) -> Path:
    """Resolve only the explicitly trusted project entrypoint, not its descendants."""
    expected = Path(os.path.abspath(expected))
    canonical = root.resolve(strict=True)
    if expected.resolve(strict=True) != expected or canonical != expected or not canonical.is_dir():
        raise ValueError("project shortcut target differs from the confirmed canonical root")
    return canonical


def path_audit(root: Path, name: str = "COMPLETE.json") -> dict[str, Any]:
    """Report leaf and ancestor lstat semantics; never implicitly trust a link."""
    root = Path(os.path.abspath(root))
    leaf = root / name
    entries = []
    for path in (leaf, root, *root.parents):
        try:
            info = path.lstat()
            entries.append(
                {
                    "path": str(path),
                    "is_symlink": path.is_symlink(),
                    "link_target": os.readlink(path) if path.is_symlink() else None,
                    "resolved": str(path.resolve(strict=True)),
                    "lstat_mode": oct(info.st_mode),
                    "is_mount": path.is_mount(),
                }
            )
        except OSError as error:
            entries.append({"path": str(path), "error": str(error)})
    return {"root": str(root), "entries_leaf_first": entries}


def require_isolated(original: Path, recovery: Path) -> None:
    original, recovery = original.resolve(), recovery.resolve()
    if original == recovery or original in recovery.parents or recovery in original.parents:
        raise ValueError("recovery must not overlap the original production directory")


def require_canonical_path(path: Path) -> None:
    """Reject descendant aliases before writing even to a not-yet-existing output."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("run/output path must be canonical and absolute, without traversal")
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"symlink in canonical run/output path: {component}")
    if absolute.resolve() != absolute:
        raise ValueError(f"run/output path differs from canonical spelling: {absolute}")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"unsafe manifest path: {name}")
    root = Path(os.path.abspath(root))
    # Callers construct paths from the once-resolved project root. No symlink
    # exceptions apply to checkpoint/export roots or their internal components.
    if root.is_symlink():
        raise ValueError(f"symlink at approved checkpoint/export root: {root}")
    for ancestor in root.parents:
        if ancestor.is_symlink():
            raise ValueError(f"symlink in checkpoint/export ancestor: {ancestor}")
    canonical_root = root.resolve(strict=True)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in manifest descendant: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(canonical_root):
        raise ValueError(f"manifest path escapes approved root: {name}")
    return resolved


def verify_checkpoint(
    checkpoint: Path, required: tuple[str, ...], *, expected_step: int | None = None
) -> dict[str, Any]:
    marker = regular(checkpoint, "COMPLETE.json")
    for entry in checkpoint.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"symlink in checkpoint descendant: {entry}")
    marker_hash = sha256(marker)
    record = json.loads(marker.read_text(encoding="utf-8"))
    step_number = int(checkpoint.name) if expected_step is None else expected_step
    if record.get("schema_version") != 2 or record.get("step") != step_number:
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
