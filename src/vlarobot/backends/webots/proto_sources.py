"""Strict local resolution of the pinned Webots R2025a PROTO sources.

This module reads copied, byte-pinned PROTO files and their provenance manifest. It never
downloads, launches Webots, imports the Webots controller, applies an action, steps physics or
makes a safety decision. Unsupported geometry is reported explicitly, never substituted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

PROTO_SOURCE_SCHEMA_VERSION = 1
_RELEVANCE = frozenset(
    {"unrelated", "relevant_collision", "relevant_obstacle", "relevant_grasp", "relevant_nested"}
)
_PROTO_RE = re.compile(r"\bPROTO\s+([A-Za-z_][A-Za-z0-9_.-]*)\s*\[")
_EXTERNPROTO_RE = re.compile(r'EXTERNPROTO\s+"([^"]+)"')
_SOLID_RE = re.compile(r"\bSolid\s*\{")
_BOUNDING_RE = re.compile(r"\bboundingObject\s+([A-Za-z_][A-Za-z0-9_]*)")
_SHAPE_RE = re.compile(
    r"\b(Box|Cylinder|Capsule|Sphere|Plane|Group|Pose|Transform|Solid|"
    r"Mesh|IndexedFaceSet|Cone|ElevationGrid)\s*\{"
)
_CONTAINER_TYPES = frozenset({"Group", "Pose", "Transform", "Solid"})
_UNSUPPORTED_LEAF_TYPES = frozenset({"Mesh", "IndexedFaceSet", "Cone", "ElevationGrid"})
_TYPE_TOKEN_RE = re.compile(
    r"\b(Box|Cylinder|Capsule|Sphere|Plane|Group|Pose|Transform|Solid|"
    r"Mesh|IndexedFaceSet|Cone|ElevationGrid)\b"
)
_SELF_COLLISION_RE = re.compile(r"\bselfCollision\s+(TRUE|FALSE)")


@dataclass(frozen=True)
class ProtoSource:
    """One pinned PROTO source file and its reviewed classification."""

    name: str
    url: str
    path: Path
    sha256: str
    relevance: str
    reason: str

    @property
    def relevant(self) -> bool:
        return self.relevance != "unrelated"


@dataclass(frozen=True)
class ProtoInspection:
    """Mechanical facts extracted from one PROTO source; not a collision certificate."""

    name: str
    symbol: str | None
    solid_count: int
    bounding_object_count: int
    bounding_object_types: tuple[str, ...]
    shape_types: tuple[str, ...]
    self_collision: bool | None
    unsupported_bounding_types: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.symbol is not None and not self.unsupported_bounding_types


@dataclass(frozen=True)
class ProtoSourceManifest:
    """Verified local PROTO inventory with extraction results."""

    source_path: Path
    webots_version: str
    external: tuple[ProtoSource, ...]
    nested: tuple[ProtoSource, ...]
    inspections: Mapping[str, ProtoInspection]
    world_sha256: str
    world_urls: frozenset[str]
    forwarded_unresolved: tuple[str, ...]

    @property
    def relevant(self) -> tuple[ProtoSource, ...]:
        return tuple(item for item in (*self.external, *self.nested) if item.relevant)

    @property
    def unrelated(self) -> tuple[ProtoSource, ...]:
        return tuple(item for item in self.external if not item.relevant)

    @property
    def unresolved(self) -> tuple[str, ...]:
        """Explicitly reviewed forwarding gaps; never inferred or silently resolved."""
        return tuple(sorted(self.forwarded_unresolved))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _sha_text(value: object, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def _resolve(root: Path, value: object, name: str) -> Path:
    path = Path(_text(value, name)).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _rows(value: object, name: str, root: Path) -> tuple[ProtoSource, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    result: list[ProtoSource] = []
    for index, item in enumerate(value):
        row = _mapping(item, f"{name}[{index}]")
        if set(row) != {"name", "url", "asset", "sha256", "relevance", "reason"}:
            raise ValueError(f"{name}[{index}] has unexpected keys")
        relevance = _text(row["relevance"], f"{name}[{index}].relevance")
        if relevance not in _RELEVANCE:
            raise ValueError(f"{name}[{index}] has unsupported relevance={relevance!r}")
        path = _resolve(root, row["asset"], f"{name}[{index}].asset")
        digest = _sha_text(row["sha256"], f"{name}[{index}].sha256")
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"{name}[{index}] source hash mismatch or missing file: {path}")
        result.append(
            ProtoSource(
                _text(row["name"], f"{name}[{index}].name"),
                _text(row["url"], f"{name}[{index}].url"),
                path,
                digest,
                relevance,
                _text(row["reason"], f"{name}[{index}].reason"),
            )
        )
    if len({item.name for item in result}) != len(result):
        raise ValueError(f"{name} contains duplicate PROTO names")
    if len({item.url for item in result}) != len(result):
        raise ValueError(f"{name} contains duplicate PROTO URLs")
    return tuple(result)


def _brace_block(text: str, token_start: int) -> str:
    """Return the balanced brace block belonging to a VRML field token."""
    opening = text.find("{", token_start)
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    return ""


def _inspect(source: ProtoSource) -> ProtoInspection:
    text = source.path.read_text(encoding="utf-8")
    symbols = _PROTO_RE.findall(text)
    self_collision_values = _SELF_COLLISION_RE.findall(text)
    shape_types: set[str] = set()
    unsupported: set[str] = set()
    for match in _BOUNDING_RE.finditer(text):
        opening = text.find("{", match.start())
        head = text[match.start() : opening] if opening >= 0 else text[match.start() :]
        found = set(_TYPE_TOKEN_RE.findall(head))
        block = _brace_block(text, match.start())
        if "IS" in head.split() or "USE" in head.split():
            # Forwarded boundingObject fields are not resolved by textual inspection. The
            # manifest names the ones reviewed as unresolved; others carry local DEF/IS
            # substitutions that are checked by the scene catalog and live geometry probe.
            unsupported.add(match.group(1))
            continue
        found.update(_SHAPE_RE.findall(block))
        shape_types.update(found)
        unsupported.update(found & _UNSUPPORTED_LEAF_TYPES)
    bounding = tuple(sorted(shape_types - _CONTAINER_TYPES))
    return ProtoInspection(
        name=source.name,
        symbol=symbols[0] if symbols else None,
        solid_count=len(_SOLID_RE.findall(text)),
        bounding_object_count=len(_BOUNDING_RE.findall(text)),
        bounding_object_types=bounding,
        shape_types=tuple(sorted(shape_types)),
        self_collision=(self_collision_values[0] == "TRUE") if self_collision_values else None,
        unsupported_bounding_types=tuple(sorted(unsupported)),
    )


def load_proto_source_manifest(
    path: Path,
    *,
    world_path: Path,
    expected_world_sha256: str | None = None,
) -> ProtoSourceManifest:
    """Load and verify the pinned source inventory against the pinned factory world.

    The world ``EXTERNPROTO`` URL set must match the 27 manifest rows exactly. Any missing,
    duplicated, altered or unsubstituted source raises; no geometry is inferred from a guess.
    """
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "PROTO source manifest")
    if set(root) != {
        "schema_version",
        "status",
        "webots_version",
        "provenance",
        "scene_catalog",
        "protos",
        "nested_sources",
        "unresolved_forwarded_geometry",
    }:
        raise ValueError("PROTO source manifest has unexpected top-level keys")
    if root["schema_version"] != PROTO_SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported PROTO source manifest schema_version")
    if _text(root["status"], "status") != "diagnostic_not_certified":
        raise ValueError("PROTO source manifest status must be diagnostic_not_certified")
    provenance = _mapping(root["provenance"], "provenance")
    if set(provenance) != {
        "cache_log",
        "cache_log_sha256",
        "webots_version_file",
        "webots_version_sha256",
        "method",
    }:
        raise ValueError("PROTO provenance has unexpected keys")
    provenance_pairs = (
        ("cache_log", "cache_log_sha256"),
        ("webots_version_file", "webots_version_sha256"),
    )
    for path_key, hash_key in provenance_pairs:
        recorded = Path(_text(provenance[path_key], f"provenance.{path_key}"))
        recorded_hash = _sha_text(provenance[hash_key], f"provenance.{hash_key}")
        if not recorded.is_file() or _sha256(recorded) != recorded_hash:
            raise ValueError(f"PROTO provenance file hash mismatch: {recorded}")
    if "byte-for-byte copy" not in _text(provenance["method"], "provenance.method"):
        raise ValueError("PROTO provenance must declare byte-for-byte copies")

    external = _rows(root["protos"], "protos", source.parent)
    nested = _rows(root["nested_sources"], "nested_sources", source.parent)
    if len(external) != 27:
        raise ValueError(f"expected 27 external PROTO references, got {len(external)}")
    world = Path(world_path).expanduser().resolve()
    world_sha = _sha256(world)
    if expected_world_sha256 is not None and world_sha != _sha_text(
        expected_world_sha256, "expected_world_sha256"
    ):
        raise ValueError("pinned factory world hash mismatch")
    world_urls = frozenset(_EXTERNPROTO_RE.findall(world.read_text(encoding="utf-8")))
    if world_urls != frozenset(item.url for item in external):
        raise ValueError("world EXTERNPROTO set does not match the resolved PROTO manifest")

    unresolved = tuple(
        _text(value, "unresolved_forwarded_geometry")
        for value in cast(Sequence[object], root["unresolved_forwarded_geometry"])
    )
    external_names = {item.name for item in external}
    if (
        not unresolved
        or len(set(unresolved)) != len(unresolved)
        or not set(unresolved).issubset(external_names)
    ):
        raise ValueError("unresolved_forwarded_geometry must name external PROTOs uniquely")
    for name in unresolved:
        inspection = _inspect(next(item for item in external if item.name == name))
        if not inspection.unsupported_bounding_types:
            raise ValueError(f"{name} does not contain a forwarded boundingObject")

    inspections = {item.name: _inspect(item) for item in (*external, *nested)}
    return ProtoSourceManifest(
        source_path=source,
        webots_version=_text(root["webots_version"], "webots_version"),
        external=external,
        nested=nested,
        inspections=inspections,
        world_sha256=world_sha,
        world_urls=world_urls,
        forwarded_unresolved=unresolved,
    )


def manifest_report(manifest: ProtoSourceManifest) -> dict[str, object]:
    """Return a JSON-serializable, non-scored inventory report."""
    return {
        "schema_version": PROTO_SOURCE_SCHEMA_VERSION,
        "status": "resolved_local_sources_not_certified",
        "webots_version": manifest.webots_version,
        "world_sha256": manifest.world_sha256,
        "external_count": len(manifest.external),
        "nested_count": len(manifest.nested),
        "relevant_count": len(manifest.relevant),
        "unrelated_count": len(manifest.unrelated),
        "unresolved": list(manifest.unresolved),
        "protos": [
            {
                "name": item.name,
                "url": item.url,
                "path": str(item.path),
                "sha256": item.sha256,
                "relevance": item.relevance,
                "reason": item.reason,
                "inspection": {
                    "symbol": manifest.inspections[item.name].symbol,
                    "solid_count": manifest.inspections[item.name].solid_count,
                    "bounding_object_count": manifest.inspections[item.name].bounding_object_count,
                    "bounding_object_types": list(
                        manifest.inspections[item.name].bounding_object_types
                    ),
                    "shape_types": list(manifest.inspections[item.name].shape_types),
                    "self_collision": manifest.inspections[item.name].self_collision,
                    "unsupported_bounding_types": list(
                        manifest.inspections[item.name].unsupported_bounding_types
                    ),
                },
            }
            for item in manifest.external
        ],
        "nested_sources": [
            {
                "name": item.name,
                "url": item.url,
                "path": str(item.path),
                "sha256": item.sha256,
                "relevance": item.relevance,
            }
            for item in manifest.nested
        ],
    }


__all__ = [
    "PROTO_SOURCE_SCHEMA_VERSION",
    "ProtoInspection",
    "ProtoSource",
    "ProtoSourceManifest",
    "load_proto_source_manifest",
    "manifest_report",
]
