"""SSH artifact handoff with receiver-side fsync, read-back and explicit receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

CHUNK = 1024 * 1024
MAX_BYTES = 20 * 1024**3


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", value) or ".." in value:
        raise ValueError("unsafe artifact identifier")
    return value


def sync_directory(path: Path) -> bool:
    if os.name == "nt":
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def receive(stream: BinaryIO, root: Path) -> dict[str, Any]:
    """Publish into a new exclusive directory; retain partial evidence on failure."""
    raw = stream.readline(8193)
    if len(raw) > 8192 or not raw.endswith(b"\n"):
        raise ValueError("invalid transfer header")
    header = json.loads(raw)
    artifact_id = identifier(header["artifact_id"])
    name = identifier(header["filename"])
    size = header["bytes"]
    expected = header["sha256"]
    if (
        type(size) is not int
        or not 0 < size <= MAX_BYTES
        or not re.fullmatch(r"[a-f0-9]{64}", expected)
    ):
        raise ValueError("invalid transfer size/hash")
    root = root.absolute()
    if root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("receiver root must have canonical spelling and no symlinks")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < size + CHUNK:
        raise ValueError("insufficient independent receiver storage")
    directory = root / artifact_id
    directory.mkdir(exist_ok=False)
    sync_directory(root)
    partial = directory / (name + ".partial")
    remaining = size
    hasher = hashlib.sha256()
    with partial.open("xb") as output:
        while remaining:
            data = stream.read(min(CHUNK, remaining))
            if not data:
                raise ValueError("truncated transfer; partial preserved")
            output.write(data)
            hasher.update(data)
            remaining -= len(data)
        output.flush()
        os.fsync(output.fileno())
    if stream.read(1):
        raise ValueError("unexpected trailing transfer bytes")
    if hasher.hexdigest() != expected or digest(partial) != expected:
        raise ValueError("receiver read-back SHA-256 mismatch; partial preserved")
    final = directory / name
    partial.rename(final)
    directory_synced = sync_directory(directory)
    report = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "filename": name,
        "bytes": size,
        "sha256": expected,
        "receiver_path": str(final),
        "receiver_readback_verified": True,
        "file_fsync": True,
        "directory_fsync": directory_synced,
        "receiver_free_bytes_after": shutil.disk_usage(directory).free,
    }
    with (directory / "receipt.json").open("x", encoding="utf-8") as receipt:
        json.dump(report, receipt, indent=2)
        receipt.write("\n")
        receipt.flush()
        os.fsync(receipt.fileno())
    sync_directory(directory)
    return report


def ssh_command(config: dict[str, Any]) -> list[str]:
    destination = config["destination"]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", destination):
        raise ValueError("invalid SSH destination; configure a hostname or SSH alias")
    port = int(config.get("port", 22))
    if not 1 <= port <= 65535:
        raise ValueError("invalid SSH port")
    remote = [
        config["remote_python"],
        config["remote_helper"],
        "receive",
        "--root",
        config["remote_root"],
    ]
    # Remote endpoint is explicitly Linux; quote its command for the SSH login shell.
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(port),
        "-i",
        config["identity_file"],
        "-o",
        f"UserKnownHostsFile={config['known_hosts']}",
        destination,
        shlex.join(remote),
    ]


def send_command(
    artifact: Path,
    command: list[str],
    artifact_id: str,
    receipt_path: Path,
    timeout_s: int,
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Bounded-memory sender; success requires a matching independent receipt."""
    identifier(artifact_id)
    identifier(artifact.name)
    if artifact.is_symlink() or not artifact.is_file() or receipt_path.exists():
        raise ValueError("invalid source or receipt already exists")
    expected = digest(artifact)
    size = artifact.stat().st_size
    header = {
        "artifact_id": artifact_id,
        "filename": artifact.name,
        "bytes": size,
        "sha256": expected,
    }
    started = time.perf_counter()
    # A local temporary wire file avoids buffering multi-GB payloads in RAM and
    # allows subprocess timeout to cover the entire network transfer.
    with tempfile.TemporaryFile() as wire:
        wire.write(json.dumps(header).encode("utf-8") + b"\n")
        with artifact.open("rb") as source:
            shutil.copyfileobj(source, wire, CHUNK)
        wire.seek(0)
        result = subprocess.run(
            command, stdin=wire, capture_output=True, timeout=timeout_s, check=True
        )
    report = json.loads(result.stdout.decode("utf-8"))
    if (
        report.get("artifact_id") != artifact_id
        or report.get("sha256") != expected
        or report.get("bytes") != size
        or report.get("receiver_readback_verified") is not True
        or report.get("file_fsync") is not True
        or report.get("directory_fsync") is not True
    ):
        raise ValueError("independent durable receipt missing or mismatched")
    if digest(artifact) != expected:
        raise ValueError("source changed during handoff")
    report["sender_elapsed_s"] = time.perf_counter() - started
    if config_sha256 is not None:
        report["transfer_config_sha256"] = config_sha256
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def send(artifact: Path, config_file: Path, artifact_id: str, receipt_path: Path) -> dict[str, Any]:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    return send_command(
        artifact,
        ssh_command(config),
        artifact_id,
        receipt_path,
        int(config.get("timeout_s", 1800)),
        config_sha256=digest(config_file),
    )


def pull(remote_file: str, output: Path, expected: str, size: int, config_file: Path) -> None:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    command = ssh_command(config)
    command[-1] = shlex.join(
        [
            config["remote_python"],
            config["remote_helper"],
            "emit",
            "--file",
            remote_file,
        ]
    )
    if output.exists() or not re.fullmatch(r"[a-f0-9]{64}", expected) or not 0 < size <= MAX_BYTES:
        raise ValueError("invalid/existing download or expected metadata")
    partial = output.with_name(output.name + ".partial")
    with partial.open("xb") as stream:
        subprocess.run(
            command, stdout=stream, check=True, timeout=int(config.get("timeout_s", 1800))
        )
        stream.flush()
        os.fsync(stream.fileno())
    if partial.stat().st_size != size or digest(partial) != expected:
        raise ValueError("download size/SHA-256 mismatch; partial retained")
    partial.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("receive", "send", "emit", "pull"))
    parser.add_argument("--root", type=Path)
    # Remote Linux source paths must remain strings even on a Windows sender.
    parser.add_argument("--file")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--bytes", type=int)
    args = parser.parse_args()
    if args.mode == "emit":
        local_file = Path(args.file) if args.file is not None else None
        if local_file is None or local_file.is_symlink() or not local_file.is_file():
            parser.error("regular --file required")
        with local_file.open("rb") as stream:
            shutil.copyfileobj(stream, sys.stdout.buffer, CHUNK)
        return
    if args.mode == "pull":
        if None in (args.file, args.out, args.expected_sha256, args.bytes, args.config):
            parser.error("--file, --out, --expected-sha256, --bytes, --config required")
        pull(args.file, args.out, args.expected_sha256, args.bytes, args.config)
        print("INDEPENDENT_INPUT_DOWNLOAD_VERIFIED")
        return
    if args.mode == "receive":
        if args.root is None:
            parser.error("--root required")
        result = receive(sys.stdin.buffer, args.root)
    else:
        if None in (args.file, args.config, args.artifact_id, args.receipt):
            parser.error("--file, --config, --artifact-id, --receipt required")
        result = send(Path(args.file), args.config, args.artifact_id, args.receipt)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
