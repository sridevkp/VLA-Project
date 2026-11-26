"""Tests for gitignored V2 evidence backup packaging; no model or simulator dependency."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/recovery/backup_v2_evidence.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=True,
    )


def test_backup_build_copy_and_verify(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    nested = source_root / "episodes"
    nested.mkdir(parents=True)
    (nested / "episode.json").write_text('{"ok": true}\n', encoding="utf-8")
    (source_root / "empty.log").write_bytes(b"")

    manifest_path = tmp_path / "manifest.json"
    _run(
        "build",
        "--output",
        str(manifest_path),
        "--source",
        str(source_root),
        "runs/v2-fixture",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["file_count"] == 2
    assert [item["relative_path"] for item in manifest["files"]] == [
        "runs/v2-fixture/empty.log",
        "runs/v2-fixture/episodes/episode.json",
    ]

    destination = tmp_path / "external-backup"
    copied = _run(
        "copy",
        "--manifest",
        str(manifest_path),
        "--destination",
        str(destination),
    )
    receipt = json.loads(copied.stdout)
    assert receipt["verified_files"] == 2
    assert (destination / "V2_BACKUP_RECEIPT.json").is_file()
    assert (destination / "V2_EVIDENCE_MANIFEST.json").is_file()

    verified = _run(
        "verify",
        "--manifest",
        str(manifest_path),
        "--destination",
        str(destination),
    )
    assert json.loads(verified.stdout)["verified_files"] == 2


def test_backup_verify_fails_after_tamper(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    (source_root / "diagnostic.json").write_text('{"success": false}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _run(
        "build",
        "--output",
        str(manifest_path),
        "--source",
        str(source_root),
        "runs/v2-fixture",
    )
    destination = tmp_path / "external-backup"
    _run("copy", "--manifest", str(manifest_path), "--destination", str(destination))

    target = destination / "runs/v2-fixture/diagnostic.json"
    target.write_text('{"success": true}\n', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--manifest",
            str(manifest_path),
            "--destination",
            str(destination),
        ],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "hash or size mismatch" in result.stderr
