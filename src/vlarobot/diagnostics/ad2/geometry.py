"""Conservative static *coverage inventory*, not a collision or swept-volume solver."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .evidence import digest


def _source(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def geometry_inventory(urdf: Path, world: Path, assets: dict[str, Path]) -> dict:
    """Enumerate existing sources; unresolved expansion always remains partial/unsupported."""
    robot = ET.parse(urdf).getroot()
    links = []
    for link in robot.findall("link"):
        collisions = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            shapes = (
                []
                if geometry is None
                else [{"kind": child.tag, "parameters": child.attrib} for child in geometry]
            )
            origin = collision.find("origin")
            collisions.append(
                {"origin": None if origin is None else origin.attrib, "shapes": shapes}
            )
        links.append(
            {
                "name": link.attrib["name"],
                "collisions": collisions,
                "coverage": "partial" if collisions else "unsupported",
                "reason": "URDF source only; live transforms/containment unverified",
            }
        )
    joints = []
    for joint in robot.findall("joint"):
        parent, child, limit = joint.find("parent"), joint.find("child"), joint.find("limit")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type"),
                "parent": None if parent is None else parent.attrib["link"],
                "child": None if child is None else child.attrib["link"],
                "limits": None if limit is None else limit.attrib,
                "exclusion_approved": False,
            }
        )
    sources = {}
    for name, path in sorted(assets.items()):
        text = path.read_text(encoding="utf-8")
        sources[name] = {
            **_source(path),
            "coverage": "partial",
            "bounding_object_lines": [
                i for i, line in enumerate(text.splitlines(), 1) if "boundingObject" in line
            ],
            "nested_solidbox_lines": [
                text[: m.start()].count("\n") + 1 for m in re.finditer(r"\bSolidBox\s*\{", text)
            ],
            "motor_limit_source_lines": [
                {"line": i, "text": line.strip()}
                for i, line in enumerate(text.splitlines(), 1)
                if re.search(
                    r"\b(maxVelocity|maxTorque|maxForce|controlPID|acceleration|minPosition|maxPosition)\b",
                    line,
                )
            ],
            "controller_literal_candidates": re.findall(r'\bcontroller\s+"([^"]*)"', text),
            "self_collision_source_lines": [
                {"line": i, "text": line.strip()}
                for i, line in enumerate(text.splitlines(), 1)
                if "selfCollision" in line
            ],
            "template_expansion_required": "%<" in text,
            "reason": "Source enumeration does not resolve composed geometry",
        }
    text = world.read_text(encoding="utf-8")
    required = {
        "Tiago",
        "TiagoBase",
        "TiagoBody",
        "TiagoFrontArm",
        "TiagoGripper",
        "Table",
        "SolidBox",
        "Can",
        "Apple",
        "Hammer",
        "Astra",
        "HokuyoUrg04lxug01",
    }
    return {
        "kind": "AD2-static-coverage-v1",
        "collision_safety_certified": False,
        "live_execution_authorized": False,
        "overall_coverage": "partial",
        "urdf_static_enumeration": "complete",
        "sources": {"urdf": _source(urdf), "world": _source(world), "assets": sources},
        "robot_links": links,
        "adjacency_not_approved_exclusions": joints,
        "missing_assets": sorted(required - assets.keys()),
        "world_controller_literals": re.findall(r'\bcontroller\s+"([^"]*)"', text),
        "world_definitions": re.findall(r"\bDEF\s+(\w+)\s+(\w+)\s*\{", text),
        "required_scene_geometry": {
            "arm_wrist_gripper_body": "partial: source primitives; all-link frames unresolved",
            "table_top_and_four_nested_feet": "partial: five SolidBox instances; no expanded proof",
            "CAN_APP_HAM": "partial: source dimensions; snapshot transforms required",
            "floor_support": "unsupported: Plane and allowed-support contact semantics unproven",
            "self_collision": "unsupported: disabled by default; adjacency is not an exclusion",
            "moving_objects_controllers": "partial: world/PROTO controllers require motion bounds",
            "swept_volumes": "unsupported: no certified conservative motion/geometry envelope",
        },
        "stop_reasons": [
            "No swept-volume proof",
            "No all-link frame/containment proof",
            "No approved margin or self/support contact exclusions",
            "Nested world geometry and other controller motion require closure",
        ],
    }
