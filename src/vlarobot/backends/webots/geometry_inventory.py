"""Read-only inventory of actual cached Webots boundingObjects and URDF collision nodes.

This is coverage evidence, not a collision checker or a certificate of frame alignment.
Unresolved USE/IS/template geometry is retained explicitly rather than declared covered.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NODE = re.compile(r"(?:(?:DEF)\s+(\w+)\s+)?(\w+)\s*\{")


def proto_inventory(path: Path) -> dict[str, Any]:
    """Inventory source boundingObject blocks with explicit owner/joint and unresolved fields."""
    source = path.read_text(encoding="utf-8")
    clean = re.sub(
        r'"(?:\\.|[^"\\])*"|#[^\n]*',
        lambda m: m[0] if m[0].startswith('"') else " " * len(m[0]),
        source,
    )
    pairs = {}
    stack = []
    for token in re.finditer(r'"(?:\\.|[^"\\])*"|[{}]', clean):
        if token[0] == "{":
            stack.append(token.start())
        elif token[0] == "}":
            if not stack:
                raise ValueError("unbalanced PROTO source")
            pairs[stack.pop()] = token.start()
    if stack:
        raise ValueError("unbalanced PROTO source")
    nodes: list[tuple[int, int, int, str, str | None]] = []
    for m in _NODE.finditer(clean):
        opening = m.end() - 1
        if opening in pairs:
            nodes.append((m.start(), opening, pairs[opening], m[2], m[1]))
    proto = re.search(r"\bPROTO\s+(\w+)", clean)
    default = re.search(r'field\s+SFString(?:\{[^}]*\})?\s+name\s+"([^"]+)"', clean)
    blocks: list[dict[str, Any]] = []
    for m in re.finditer(r"\bboundingObject\s+(?:DEF\s+\w+\s+)?(\w+)\s*\{", clean):
        opening = m.end() - 1
        end = pairs[opening]
        owners = [n for n in nodes if n[3] in ("Solid", "Robot") and n[1] < opening < n[2]]
        owner = min(owners, key=lambda n: n[2] - n[1]) if owners else None
        name = None
        joint = None
        if owner:
            chars = list(clean[owner[1] + 1 : owner[2]])
            for n in nodes:
                if owner[1] < n[0] and n[2] < owner[2]:
                    lo, hi = n[0] - owner[1] - 1, n[2] - owner[1]
                    chars[lo:hi] = " " * (hi - lo)
            local = "".join(chars)
            label = re.search(r'\bname\s+"([^"]+)"', local)
            name = (
                label[1]
                if label
                else (default[1] if default and re.search(r"\bname\s+IS\s+name\b", local) else None)
            )
            parents = [
                n for n in nodes if n[3] in ("HingeJoint", "SliderJoint") and n[1] < owner[0] < n[2]
            ]
            if parents:
                joint = min(parents, key=lambda n: n[2] - n[1])[4]
        text = clean[opening : end + 1]
        shapes = []
        for n in nodes:
            if opening <= n[1] < end and n[3] in ("Box", "Cylinder", "Capsule", "Sphere"):
                body = clean[n[1] + 1 : n[2]]
                dimensions = {}
                for key, width in [("size", 3), ("radius", 1), ("height", 1)]:
                    value = re.search(
                        r"\b"
                        + key
                        + r"\s+("
                        + _NUMBER
                        + r"(?:\s+"
                        + _NUMBER
                        + r"){"
                        + str(width - 1)
                        + "})",
                        body,
                    )
                    if value:
                        dimensions[key] = [float(v) for v in value[1].split()]
                shapes.append({"type": n[3], "dimensions": dimensions})
        unresolved = bool(re.search(r"\b(?:USE|IS|Mesh|IndexedFaceSet)\b|%[<{]", text))
        blocks.append(
            {
                "owner": name,
                "parent_joint_def": joint,
                "line": source.count("\n", 0, m.start()) + 1,
                "primitives": shapes,
                "unresolved_geometry": unresolved,
                "source_block_sha256": hashlib.sha256(
                    source[m.start() : end + 1].encode("utf-8")
                ).hexdigest(),
            }
        )
    statements = list(re.finditer(r"\bboundingObject\b", clean))
    unparsed = len(statements) - len(blocks)
    return {
        "unparsed_bounding_object_statements": unparsed,
        "source_geometry_fully_resolved": unparsed == 0
        and all(not b["unresolved_geometry"] for b in blocks),
        "proto": proto[1] if proto else None,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bounding_objects": blocks,
        "primitive_counts": dict(Counter(p["type"] for b in blocks for p in b["primitives"])),
    }


def urdf_inventory(path: Path) -> dict[str, Any]:
    """Preserve every collision origin/primitive; fixed-joint neighbors are not auto-excluded."""
    root = ET.parse(path).getroot()
    links = []
    for link in root.findall("link"):
        geometries = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            origin = collision.find("origin")
            geometries.append(
                {
                    "origin": origin.attrib if origin is not None else {},
                    "primitives": [{"type": p.tag, "attributes": p.attrib} for p in geometry]
                    if geometry is not None
                    else [],
                    "unresolved_geometry": geometry is None
                    or any(p.tag not in ("box", "cylinder", "sphere") for p in geometry),
                }
            )
        links.append({"name": link.attrib["name"], "collisions": geometries})
    joints = []
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError("URDF joint has incomplete link coverage")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib["type"],
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
            }
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "links": links,
        "joints": joints,
    }
