"""Read-only hash preflight for frozen Week 3 evidence; never writes or runs a campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPORT_SHA256 = "23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b"
BASELINE_PROVENANCE_SHA256 = "690f1597be07251200d74afe66ccec24fa06567ce29ba20f7c80175841d3aca9"
VLA_PROVENANCE_SHA256 = "2375d2ab188406f1bcd600eb7cede307af226222a248236a79fdad27545f8b02"
VLA_SUMMARY_SHA256 = "9695b1da66a2ec38dceba4b012e049cdd1c8412cc3c8aca7f0d4b076e7a3a416"
PLANNING_EVIDENCE_SHA256 = "67ef6dc80835d37367fd5fddcb6ddaac1df736136282c0ecfb58007705d2e679"
BASELINE_LOCKED_FILE_COUNT = 12
VLA_MANIFEST_COUNT = 225
PLANNING_SOURCE_COUNT = 12
PLANNING_FROZEN_COUNT = 12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_recorded_path(
    raw: str, *, repo_root: Path, locked_root: Path, analytic_root: Path
) -> Path:
    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((repo_root / path, locked_root / path))
    parts = path.parts
    for marker, root in (
        ("VLA-Project-analytic", analytic_root),
        ("VLA-Project-week3", repo_root),
        ("VLA-Project", locked_root),
    ):
        if marker in parts:
            suffix = Path(*parts[parts.index(marker) + 1 :])
            candidates.append(root / suffix)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"recorded source not found: {raw}")


def verify_baseline(provenance_path: Path, locked_root: Path) -> dict[str, Any]:
    anchor = sha256(provenance_path)
    payload = load_json(provenance_path)
    files = payload.get("files_sha256")
    if not isinstance(files, dict) or len(files) != BASELINE_LOCKED_FILE_COUNT:
        raise ValueError("baseline provenance does not contain exactly 12 locked files")
    matched = 0
    for relative, expected in files.items():
        path = locked_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"baseline locked file mismatch: {relative}")
        matched += 1
    if anchor != BASELINE_PROVENANCE_SHA256:
        raise ValueError("baseline provenance SHA mismatch")
    return {
        "provenance_sha256_ok": True,
        "locked_source_files_verified": matched,
        "locked_source_root": str(locked_root),
    }


def verify_vla_campaign(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if len(manifest) != VLA_MANIFEST_COUNT:
        raise ValueError(f"VLA manifest has {len(manifest)} entries, expected {VLA_MANIFEST_COUNT}")
    for relative, record in manifest.items():
        path = root / relative
        if not isinstance(record, dict) or not path.is_file():
            raise ValueError(f"VLA campaign file missing or malformed: {relative}")
        if sha256(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"VLA campaign file mismatch: {relative}")
    provenance_ok = sha256(root / "provenance.json") == VLA_PROVENANCE_SHA256
    summary_ok = sha256(root / "summary.json") == VLA_SUMMARY_SHA256
    if not provenance_ok or not summary_ok:
        raise ValueError("VLA campaign provenance or summary SHA mismatch")
    return {
        "manifest_entries_verified": len(manifest),
        "provenance_sha256_ok": provenance_ok,
        "summary_sha256_ok": summary_ok,
    }


def verify_planning_evidence(
    path: Path, *, repo_root: Path, locked_root: Path, analytic_root: Path
) -> dict[str, Any]:
    if sha256(path) != PLANNING_EVIDENCE_SHA256:
        raise ValueError("planning evidence JSON SHA mismatch")
    payload = load_json(path)
    sources = payload.get("source_sha256")
    frozen = payload.get("frozen_source_verified_after_run")
    if not isinstance(sources, dict) or len(sources) != PLANNING_SOURCE_COUNT:
        raise ValueError("planning evidence pinned-source count mismatch")
    if not isinstance(frozen, dict) or len(frozen) != PLANNING_FROZEN_COUNT:
        raise ValueError("planning evidence frozen-source count mismatch")
    for raw, expected in sources.items():
        resolved = resolve_recorded_path(
            raw, repo_root=repo_root, locked_root=locked_root, analytic_root=analytic_root
        )
        if sha256(resolved) != expected:
            raise ValueError(f"planning pinned source mismatch: {raw}")
    for relative, expected in frozen.items():
        resolved = repo_root / relative
        if not resolved.is_file() or sha256(resolved) != expected:
            raise ValueError(f"planning frozen source mismatch: {relative}")
    return {
        "evidence_sha256_ok": True,
        "pinned_sources_verified": len(sources),
        "frozen_sources_verified": len(frozen),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--locked-source-root", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, default=Path.home() / "Downloads" / "Project_Report.pdf"
    )
    parser.add_argument(
        "--analytic-root", type=Path, default=default_repo.parent / "VLA-Project-analytic"
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    locked_root = args.locked_source_root.resolve()
    report = args.report.resolve()
    analytic_root = args.analytic_root.resolve()
    baseline_provenance = locked_root / (
        "runs/eval/baseline-train-s0-fresh-v2-registered/provenance.json"
    )
    vla_root = locked_root / "runs/eval/vla-train-s0-step20000-recovered-v1"
    vla_manifest = repo_root / "docs/evidence/vla-train-step20000-files.json"
    planning_evidence = repo_root / "docs/evidence/planning-optimization-final.json"

    report_ok = sha256(report) == REPORT_SHA256
    if not report_ok:
        raise ValueError("authoritative report SHA mismatch")
    result = {
        "report_sha256_ok": report_ok,
        "baseline": verify_baseline(baseline_provenance, locked_root),
        "vla_campaign": verify_vla_campaign(vla_root, vla_manifest),
        "planning_evidence": verify_planning_evidence(
            planning_evidence,
            repo_root=repo_root,
            locked_root=locked_root,
            analytic_root=analytic_root,
        ),
        "test_split_accessed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FROZEN EVIDENCE PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
