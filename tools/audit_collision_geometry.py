"""Audit pinned actual Webots asset coverage; never certify missing or unmapped geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve audit version")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vlarobot.backends.webots.geometry_inventory import proto_inventory, urdf_inventory

    base = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/"
    assets = {
        name: base + suffix
        for name, suffix in {
            "Tiago": "robots/pal_robotics/tiago/protos/Tiago.proto",
            "TiagoBase": "robots/pal_robotics/tiago_base/protos/TiagoBase.proto",
            "TiagoBody": "robots/pal_robotics/tiago_extensions/protos/TiagoBody.proto",
            "TiagoFrontArm": "robots/pal_robotics/tiago_extensions/protos/TiagoFrontArm.proto",
            "TiagoGripper": "robots/pal_robotics/tiago_extensions/protos/TiagoGripper.proto",
            "Astra": "devices/orbbec/protos/Astra.proto",
            "HokuyoUrg04lxug01": "devices/hokuyo/protos/HokuyoUrg04lxug01.proto",
        }.items()
    }
    protos = {}
    missing = []
    for name, url in assets.items():
        p = args.cache / hashlib.sha1(url.encode("utf-8")).hexdigest()
        if not p.is_file():
            missing.append(name)
            continue
        report = proto_inventory(p)
        report["url"] = url
        if report["proto"] != name:
            raise ValueError("cached URL/proto identity mismatch")
        protos[name] = report
    urdf = urdf_inventory(args.project / "src/vlarobot/backends/webots/tiago.urdf")
    counts = {
        link["name"]: sum(len(g["primitives"]) for g in link["collisions"])
        for link in urdf["links"]
    }
    joint_children = {j["name"]: j["child"] for j in urdf["joints"]}
    aliases = {
        ("TiagoFrontArm", "Left shoulder"): "TIAGo front arm",
        ("TiagoBase", "TIAGo"): "base_link",
    }
    gaps = []
    mapping = []
    unmapped = []
    for name, report in protos.items():
        for block in report["bounding_objects"]:
            owner = block["owner"]
            if name == "TiagoFrontArm" and block["parent_joint_def"] == "ARM_1_JOINT":
                owner = joint_children["arm_1_joint"]
            owner = aliases.get((name, owner), owner)
            if owner not in counts:
                unmapped.append(
                    {"proto": name, "owner": block["owner"], "primitives": len(block["primitives"])}
                )
            else:
                mapping.append(
                    {
                        "proto": name,
                        "owner": block["owner"],
                        "urdf_link": owner,
                        "webots_count": len(block["primitives"]),
                        "urdf_count": counts[owner],
                    }
                )
            if owner in counts and len(block["primitives"]) != counts[owner]:
                gaps.append(
                    {
                        "proto": name,
                        "link": owner,
                        "webots_primitives": len(block["primitives"]),
                        "urdf_primitives": counts[owner],
                        "status": "not equivalent; shape containment/frame proof required",
                    }
                )
    world = args.project / "worlds/factory.wbt"
    source = world.read_text(encoding="utf-8")
    tiago = (args.cache / hashlib.sha1(assets["Tiago"].encode()).hexdigest()).read_text(
        encoding="utf-8"
    )
    report = {
        "status": "NOT CERTIFIED",
        "scope": "read-only actual pinned asset coverage audit, not physical collision acceptance",
        "urdf": urdf,
        "webots": protos,
        "missing_assets": missing,
        "coverage_differences": gaps,
        "link_mapping": mapping,
        "unmapped_bounding_objects": unmapped,
        "frozen_world_sha256": hashlib.sha256(world.read_bytes()).hexdigest(),
        "self_collision_enabled_in_frozen_world": "selfCollision TRUE" in source,
        "proto_self_collision_default_false": bool(re.search(r"selfCollision\s+FALSE", tiago)),
        "unverified": [
            "per-link URDF-to-Webots registration",
            "primitive containment for differences",
            "fixed-joint versus physically adjacent exclusions",
            "carried-object enclosure and swept bounds",
            "all nearby world obstacle bounding geometry",
            "physical swept/self-collision validation",
        ],
        "test_accessed": False,
        "training_or_world_modified": False,
    }
    args.out.mkdir(parents=True)
    (args.out / "coverage-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage_differences": gaps,
                "link_mapping": mapping,
                "unmapped_bounding_objects": unmapped,
                "missing_assets": missing,
                "self_collision_enabled_in_frozen_world": report[
                    "self_collision_enabled_in_frozen_world"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
