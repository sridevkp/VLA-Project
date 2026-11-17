"""Compare actual live expanded-link poses with registered URDF frames; no movement or fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


def rpy(angles: list[float]) -> np.ndarray:
    x, y, z = angles
    cx, sx, cy, sy, cz, sz = (
        math.cos(x),
        math.sin(x),
        math.cos(y),
        math.sin(y),
        math.cos(z),
        math.sin(z),
    )
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )


def axis_rotation(axis: list[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-geometry", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve frame comparison")
    urdf = args.project / "src/vlarobot/backends/webots/tiago.urdf"
    registration = args.project / "configs/calibration/urdf-base-train-v1.yaml"
    live = json.loads(args.live.read_text(encoding="utf-8"))
    q = {
        n.replace("_sensor_finger_", "_finger_").removesuffix("_sensor"): v
        for n, v in live["joint_positions"].items()
    }
    root = ET.parse(urdf).getroot()
    joints = root.findall("joint")
    child_names = {j.find("child").attrib["link"] for j in joints}
    roots = {link.attrib["name"] for link in root.findall("link")} - child_names
    if len(roots) != 1:
        raise ValueError("URDF must be a single rooted tree")
    registration_matrix = np.asarray(
        yaml.safe_load(registration.read_text(encoding="utf-8"))["T_robot_base_urdf_root"]
    )
    transforms = {
        next(iter(roots)): np.asarray(
            yaml.safe_load(registration.read_text(encoding="utf-8"))["T_robot_base_urdf_root"]
        )
    }
    pending = joints.copy()
    while pending:
        progress = False
        for joint in list(pending):
            parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, 3] = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                t[:3, :3] = rpy([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()])
            motion = np.eye(4)
            kind = joint.attrib["type"]
            axis = joint.find("axis")
            a = (
                [float(v) for v in axis.attrib.get("xyz", "1 0 0").split()]
                if axis is not None
                else [1.0, 0.0, 0.0]
            )
            value = q.get(joint.attrib["name"], 0.0)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = axis_rotation(a, value)
            elif kind == "prismatic":
                motion[:3, 3] = np.asarray(a) * value
            transforms[child] = transforms[parent] @ t @ motion
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("disconnected/cyclic URDF")
    corrections = None
    if args.source_geometry:
        corrections = yaml.safe_load(args.source_geometry.read_text(encoding="utf-8"))
        arm_child = next(
            j.find("child").attrib["link"] for j in joints if j.attrib["name"] == "arm_1_joint"
        )
        arm_links = {arm_child}
        for _ in range(len(joints)):
            arm_links.update(
                j.find("child").attrib["link"]
                for j in joints
                if j.find("parent").attrib["link"] in arm_links
            )
        for name in list(transforms):
            if name not in arm_links:
                transforms[name] = np.linalg.inv(registration_matrix) @ transforms[name]
    rows = []
    mapped = set()
    for obj in live["objects"]:
        name = obj["name"]
        if obj["def"] == "ARM_1":
            name = "TIAGo front arm_3"
        if obj["id"] == live["robot_id"]:
            name = "base_link"
        if name not in transforms:
            continue
        observed = np.asarray(obj["T_base_owner"]).reshape(4, 4)
        predicted = transforms[name]
        rows.append(
            {
                "urdf_link": name,
                "live_name": obj["name"],
                "position_error_m": float(np.linalg.norm(observed[:3, 3] - predicted[:3, 3])),
                "orientation_error_rad": math.acos(
                    float(
                        np.clip((np.trace(observed[:3, :3] @ predicted[:3, :3].T) - 1) / 2, -1, 1)
                    )
                ),
                "live_primitive_count": len(obj["shapes"]),
            }
        )
        mapped.add(name)
    required = {
        link.attrib["name"] for link in root.findall("link") if link.find("collision") is not None
    }
    primitive_errors = []
    links = {link.attrib["name"]: link for link in root.findall("link")}
    live_by_name = {obj["name"]: obj for obj in live["objects"]}
    for row in rows:
        name = row["urdf_link"]
        obj = live_by_name[row["live_name"]]
        candidates = []
        for collision in links[name].findall("collision"):
            origin = collision.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                transform[:3, :3] = rpy(
                    [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
                )
            if corrections and name in corrections["collision_origins"]:
                override = corrections["collision_origins"][name]
                if collision is links[name].findall("collision")[override["index"]]:
                    transform[:3, 3] = override["xyz_m"]
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            for primitive in geometry:
                candidates.append((primitive.tag, primitive.attrib, transforms[name] @ transform))
        unmatched = list(range(len(candidates)))
        observed_transform = np.asarray(obj["T_base_owner"]).reshape(4, 4)
        for shape in obj["shapes"]:
            if not shape["supported"]:
                continue
            actual = observed_transform @ np.asarray(shape["T_owner_shape"])
            options = [i for i in unmatched if candidates[i][0] == shape["type"].lower()]
            if not options:
                continue
            chosen = min(
                options, key=lambda i: np.linalg.norm(candidates[i][2][:3, 3] - actual[:3, 3])
            )
            unmatched.remove(chosen)
            kind, attributes, predicted = candidates[chosen]
            parameters = shape["parameters"]
            if kind == "box":
                dims = np.asarray([float(v) for v in attributes["size"].split()])
                geometry_difference = float(np.max(np.abs(dims - parameters["size"])))
                p_half = np.abs(predicted[:3, :3]) @ (dims / 2)
                a_half = np.abs(actual[:3, :3]) @ (np.asarray(parameters["size"]) / 2)
            elif kind in ("cylinder", "capsule"):
                radius = float(attributes["radius"])
                height = float(attributes["length"])
                geometry_difference = max(
                    abs(radius - parameters["radius"]), abs(height - parameters["height"])
                )
                p_half = radius * np.linalg.norm(predicted[:3, :2], axis=1) + height / 2 * np.abs(
                    predicted[:3, 2]
                )
                a_half = parameters["radius"] * np.linalg.norm(actual[:3, :2], axis=1) + parameters[
                    "height"
                ] / 2 * np.abs(actual[:3, 2])
            elif kind == "sphere":
                radius = float(attributes["radius"])
                geometry_difference = abs(radius - parameters["radius"])
                p_half = np.full(3, radius)
                a_half = np.full(3, parameters["radius"])
            else:
                continue
            overhang = float(
                np.max(
                    np.maximum(
                        np.maximum(
                            predicted[:3, 3] - p_half - (actual[:3, 3] - a_half),
                            actual[:3, 3] + a_half - (predicted[:3, 3] + p_half),
                        ),
                        0,
                    )
                )
            )
            primitive_errors.append(
                {
                    "link": name,
                    "type": kind,
                    "dimension_difference_m": geometry_difference,
                    "center_error_m": float(np.linalg.norm(predicted[:3, 3] - actual[:3, 3])),
                    "aabb_undercoverage_m": overhang,
                }
            )
    result = {
        "status": "MEASURED one-pose frame comparison; collision safety NOT CERTIFIED",
        "frames": rows,
        "matched_primitive_errors": primitive_errors,
        "frame_origin_differences_are_not_collision_surface_errors": True,
        "unobserved_collision_links": sorted(required - mapped),
        "self_collision_enabled": live["self_collision_enabled"],
        "scope": "passive factory initial pose, actual sensors; no fitted correction",
        "source_geometry_diagnostic": str(args.source_geometry) if args.source_geometry else None,
        "not_deployed_as_safety_certificate": True,
        "test_accessed": False,
        "training_or_world_modified": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (urdf, registration, args.live)
        },
    }
    args.out.mkdir(parents=True)
    (args.out / "link-frame-errors.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
