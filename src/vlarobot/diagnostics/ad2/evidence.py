"""Hash-pinned, immutable loading of saved targets; no solver/model imports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class EvidenceError(ValueError):
    """The sealed diagnostic identity cannot be established."""


def digest(path: Path) -> str:
    """Hash a regular file without following a final symlink."""
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise EvidenceError(f"Not a regular evidence file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def confined(root: Path, relative: str) -> Path:
    """Reject traversal, symlink components and platform-dependent path spelling."""
    parts = PurePosixPath(relative)
    if (
        not relative
        or parts.is_absolute()
        or ".." in parts.parts
        or "\\" in relative
        or ":" in relative
        or str(parts) != relative
    ):
        raise EvidenceError("Invalid evidence path")
    current = root.absolute()
    for parent in (current, *current.parents):
        if parent.is_symlink():
            raise EvidenceError("Symlinked evidence root")
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceError("Symlinked evidence component")
    return current


def read_json(path: Path, expected_sha256: str | None = None) -> Any:
    """Reject duplicate JSON keys and non-standard nonfinite literals."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvidenceError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise EvidenceError(f"Nonfinite JSON: {value}")

    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise EvidenceError("JSON changed after verification")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class Targets:
    """Only immutable, verified scalar tuples are exposed to fixtures."""

    identity: str
    manifest_sha256: str
    joints: tuple[str, ...]
    a: tuple[float, ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    scene: str
    seed: int

    def case(self, name: str) -> tuple[float, ...]:
        """Return a sealed case, without target reconstruction."""
        if name not in ("A", "B", "C"):
            raise EvidenceError("Unknown case")
        return {"A": self.a, "B": self.b, "C": self.c}[name]


def load_targets(manifest: Path, expected_sha256: str, roots: dict[str, Path]) -> Targets:
    """Verify every sealed source before parsing any target or snapshot."""
    if digest(manifest) != expected_sha256:
        raise EvidenceError("Untrusted manifest")
    spec = read_json(manifest, expected_sha256)
    if spec["schema_version"] != 1 or spec["scene"] != "train-clear-i00-s100":
        raise EvidenceError("Wrong evidence schema or TRAIN scene")
    if spec["seed"] != 100 or spec["frame"] != [0, 0]:
        raise EvidenceError("Wrong seed/snapshot")
    paths: dict[str, Path] = {}
    for key, record in spec["files"].items():
        if record["root"] not in roots:
            raise EvidenceError("Missing explicit evidence root")
        path = confined(roots[record["root"]], record["path"])
        if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise EvidenceError(f"Source mismatch: {key}")
        paths[key] = path

    def verified_json(key: str) -> Any:
        return read_json(paths[key], spec["files"][key]["sha256"])

    result = verified_json("result")
    provenance = verified_json("provenance")
    historical = spec.get("provenance_files", {})
    original = provenance.get("inputs_sha256", {})
    if set(historical) != set(original):
        raise EvidenceError("Incomplete original provenance coverage")
    for old_path, key in historical.items():
        if spec["files"][key]["sha256"] != original[old_path]:
            raise EvidenceError("Original provenance hash mismatch")
    native = verified_json("native")
    meta = verified_json("meta")
    if result["identity"] != spec["identity"] or provenance["identity"] != spec["identity"]:
        raise EvidenceError("A-D1 identity mismatch")
    if (native["row"]["episode_index"], native["row"]["frame_index"]) != (0, 0):
        raise EvidenceError("Native snapshot mismatch")
    if meta["scenario"]["name"] != spec["scene"] or meta["reset_seed"] != spec["seed"]:
        raise EvidenceError("Recording identity mismatch")
    if result["actual_numerical_ik_calls"] != 2 or result["original_gate"] != "FAILED_0_of_24":
        raise EvidenceError("Unexpected original diagnostic/gate")
    names = tuple(f"arm_{i}_joint" for i in range(1, 8))
    values = []
    for case in ("A", "B", "C"):
        q = result["cases"][case]["joints"]
        if set(q) != set(names):
            raise EvidenceError("Wrong joint set")
        if case != "A" and verified_json(case)["joints"] != q:
            raise EvidenceError("Saved case disagreement")
        target = tuple(q[name] for name in names)
        if any(type(x) is not float or not math.isfinite(x) for x in target):
            raise EvidenceError("Invalid scalar target")
        if list(target) != spec["targets"][case]:
            raise EvidenceError("Saved target differs from independent seal")
        values.append(target)
    if values[0] != values[1]:
        raise EvidenceError("A and B must be identical")
    return Targets(spec["identity"], expected_sha256, names, *values, spec["scene"], spec["seed"])
