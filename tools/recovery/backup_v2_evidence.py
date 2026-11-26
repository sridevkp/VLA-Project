"""Package and verify the gitignored V2 raw evidence on an external target.

This tool is part of evidence preservation, not model or controller behaviour.
It uses only the Python standard library and preserves paths, sizes and SHA-256
hashes. It never follows symlinks and never invokes training or simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "worldmind-v2-evidence-backup-v1"
MANIFEST_NAME = "V2_EVIDENCE_MANIFEST.json"
RECEIPT_NAME = "V2_BACKUP_RECEIPT.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a regular file without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iter_regular_files(root: Path) -> Iterable[Path]:
    """Yield sorted regular files under a root, rejecting all symlinks."""
    if root.is_symlink():
        raise ValueError(f"symlink source root is not allowed: {root}")
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise ValueError(f"source root does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink source is not allowed: {path}")
        if path.is_file():
            yield path


def default_sources() -> list[tuple[Path, str]]:
    """Return the accepted V2 raw-evidence roots used in this worktree."""
    repo_root = Path(__file__).resolve().parents[2]
    return [
        (
            repo_root / "runs/worldmind/v2-train-recovered-diag-1",
            "runs/worldmind/v2-train-recovered-diag-1",
        ),
        (
            repo_root / "runs/worldmind/v2-train-endpoint-1",
            "runs/worldmind/v2-train-endpoint-1",
        ),
        (
            Path.home() / "Downloads/v2-transition-warmstart-01/step-021000",
            "external/v2-transition-warmstart-01/step-021000",
        ),
    ]


def build_manifest(sources: list[tuple[Path, str]]) -> dict[str, Any]:
    """Hash every source file into a deterministic, path-preserving manifest."""
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root, prefix in sources:
        prefix_path = PurePosixPath(prefix)
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            raise ValueError(f"unsafe manifest prefix: {prefix}")
        if root.is_symlink():
            raise ValueError(f"symlink source root is not allowed: {root}")
        resolved_root = root.resolve()
        for path in iter_regular_files(resolved_root):
            relative = PurePosixPath(prefix) / PurePosixPath(
                path.relative_to(resolved_root).as_posix()
            )
            relative_text = relative.as_posix()
            if relative_text in seen:
                raise ValueError(f"duplicate manifest path: {relative_text}")
            seen.add(relative_text)
            files.append(
                {
                    "source": str(path),
                    "relative_path": relative_text,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    files.sort(key=lambda item: item["relative_path"])
    return {
        "schema": SCHEMA,
        "purpose": "Preserve gitignored V2 raw diagnostics and frozen checkpoint evidence",
        "source_roots": [str(root) for root, _ in sources],
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest after checking its schema and path safety."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
        raise ValueError(f"not a {SCHEMA} manifest: {path}")
    for item in manifest["files"]:
        relative = PurePosixPath(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {item['relative_path']}")
    return manifest


def verify_destination(manifest: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Verify copied files against the manifest; fail closed on any mismatch."""
    if destination.is_symlink():
        raise ValueError(f"symlink backup destination is not allowed: {destination}")
    verified = 0
    total_bytes = 0
    for item in manifest["files"]:
        target = destination.joinpath(*PurePosixPath(item["relative_path"]).parts)
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"missing or non-regular backup file: {target}")
        size = target.stat().st_size
        digest = sha256_file(target)
        if size != item["size_bytes"] or digest != item["sha256"]:
            raise ValueError(f"backup hash or size mismatch: {target}")
        verified += 1
        total_bytes += size
    return {"verified_files": verified, "verified_bytes": total_bytes}


def copy_manifest(
    manifest: dict[str, Any], destination: Path, manifest_path: Path
) -> dict[str, Any]:
    """Copy all manifest files to an external destination and verify them."""
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"backup destination is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        source = Path(item["source"])
        target = destination.joinpath(*PurePosixPath(item["relative_path"]).parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"missing or non-regular source file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError(f"symlink backup target is not allowed: {target}")
        shutil.copy2(source, target)
    result = verify_destination(manifest, destination)
    receipt = {
        "schema": SCHEMA,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "destination": str(destination),
        **result,
    }
    receipt_path = destination / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="hash the accepted source roots into a manifest")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("PATH", "PREFIX"),
        help="repeatable source root and destination prefix; defaults to accepted V2 paths",
    )

    copy = subparsers.add_parser("copy", help="copy and verify a manifest on an external target")
    copy.add_argument("--manifest", type=Path, required=True)
    copy.add_argument("--destination", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify an existing external backup")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        sources = (
            [(Path(path), prefix) for path, prefix in args.source]
            if args.source
            else default_sources()
        )
        manifest = build_manifest(sources)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = {
            "manifest": str(args.output),
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        }
    elif args.command == "copy":
        destination = args.destination.resolve()
        manifest = load_manifest(args.manifest)
        result = copy_manifest(manifest, destination, args.manifest.resolve())
        (destination / MANIFEST_NAME).write_text(
            args.manifest.read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        manifest = load_manifest(args.manifest)
        result = verify_destination(manifest, args.destination.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
