"""Versioned source and approved native-data handoff for authenticated Colab/Drive training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from vlarobot.execution.demo_collection import write_json
from vlarobot.vla.training import validate_training_approval


def file_sha256(path: Path) -> str:
    """Hash large Drive payloads without loading an entire archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_source_bundle(project: Path, out: Path) -> dict[str, Any]:
    """Package a reviewable exact source snapshot without credentials, weights or trajectories."""
    out.mkdir(parents=True, exist_ok=False)
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.split("\0")
    prefixes = ("src/", "configs/", "docs/", "tests/", "notebooks/", "model/compat/")
    roots = {"pyproject.toml", "uv.lock", ".python-version", "README.md", "AGENTS.md"}
    hashes = {}
    with zipfile.ZipFile(out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(files)):
            if name not in roots and not name.startswith(prefixes):
                continue
            path = project / name
            if path.is_symlink():
                raise ValueError("source bundle must not contain symlinks")
            if not path.is_file():
                continue
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr("snapshot-manifest.json", json.dumps(hashes, indent=2) + "\n")
    report = {
        "status": "source-ready-awaiting-approved-80-success-dataset",
        "source_files_sha256": hashes,
        "source_zip_sha256": file_sha256(out / "source.zip"),
        "training_started": False,
        "gpu_route": "Colab T4 / Google Drive; authentication required",
    }
    write_json(out / "bundle-status.json", report)
    return report


def package_approved_dataset(dataset: Path, out: Path) -> dict[str, Any]:
    """Add only verified native payload; authentication does not bypass training approval."""
    approval = validate_training_approval(dataset)
    status = json.loads((out / "bundle-status.json").read_text(encoding="utf-8"))
    if file_sha256(out / "source.zip") != status["source_zip_sha256"]:
        raise ValueError("prepared source snapshot changed")
    archive_path = out / "dataset-train-v1.zip"
    if archive_path.exists():
        raise FileExistsError("preserve existing dataset handoff archive")
    integrity = json.loads((dataset / "dataset-integrity.json").read_text(encoding="utf-8"))
    files = list(integrity["files_sha256"]) + [
        "dataset-integrity.json",
        "installed-forward.json",
        "training_approval.json",
    ]
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name in files:
            path = dataset / name
            if path.is_symlink():
                raise ValueError("dataset payload must not contain symlinks")
            archive.write(path, arcname=name)
    status.update(
        status="approved-payload-ready-awaiting-Colab-authentication",
        dataset_successes=len(approval["episodes"]),
        dataset_zip_sha256=file_sha256(archive_path),
        dataset_zip_bytes=archive_path.stat().st_size,
        training_approval_sha256=file_sha256(dataset / "training_approval.json"),
    )
    write_json(out / "bundle-status.json", status)
    return status
