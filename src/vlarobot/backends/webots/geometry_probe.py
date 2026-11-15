"""Read-only live boundingObject export in a separate development world; no motor commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def snapshot(supervisor: Any) -> dict[str, Any]:
    """Export actual expanded PROTO geometry and transforms, preserving unsupported nodes."""
    robot = supervisor.getSelf()

    def field(node, name):
        return node.getBaseNodeField(name) if node.isProto() else node.getField(name)

    def values(node, name, method, default):
        f = field(node, name)
        return getattr(f, method)() if f else default

    def transform(node):
        t = np.asarray(values(node, "translation", "getSFVec3f", [0.0, 0.0, 0.0]))
        aa = np.asarray(values(node, "rotation", "getSFRotation", [0.0, 0.0, 1.0, 0.0]))
        scale = np.asarray(values(node, "scale", "getSFVec3f", [1.0, 1.0, 1.0]))
        length = np.linalg.norm(aa[:3])
        rotation = np.eye(3)
        if aa[3] != 0 and length > 0:
            axis = aa[:3] / length
            cross = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
            )
            rotation = np.eye(3) + np.sin(aa[3]) * cross + (1 - np.cos(aa[3])) * (cross @ cross)
        result = np.eye(4)
        result[:3, :3] = rotation @ np.diag(scale)
        result[:3, 3] = t
        return result

    def child_nodes(node):
        children = field(node, "children")
        if children:
            for i in range(children.getCount()):
                child = children.getMFNode(i)
                if child:
                    yield child
        endpoint = field(node, "endPoint")
        if endpoint:
            child = endpoint.getSFNode()
            if child:
                yield child

    def shapes(node, matrix):
        kind = node.getBaseTypeName()
        matrix = matrix @ transform(node)
        if kind in ("Box", "Cylinder", "Capsule", "Sphere", "Plane"):
            dimensions = {}
            for key, method in [
                ("size", "getSFVec3f"),
                ("radius", "getSFFloat"),
                ("height", "getSFFloat"),
            ]:
                f = field(node, key)
                if f:
                    if kind == "Plane" and key == "size":
                        method = "getSFVec2f"
                    dimensions[key] = getattr(f, method)()
            return [
                {
                    "type": kind,
                    "parameters": dimensions,
                    "T_owner_shape": matrix.tolist(),
                    "supported": kind != "Plane",
                }
            ]
        result = []
        if kind == "Shape":
            f = field(node, "geometry")
            child = f.getSFNode() if f else None
            if child:
                return shapes(child, matrix)
        elif kind in ("Pose", "Transform", "Group"):
            for child in child_nodes(node):
                result.extend(shapes(child, matrix))
            return result
        return [{"type": kind, "supported": False, "T_owner_shape": matrix.tolist()}]

    seen = set()
    objects = []

    def visit(node):
        if node.getId() in seen:
            return
        seen.add(node.getId())
        f = field(node, "boundingObject")
        if f and f.getSFNode():
            name = values(node, "name", "getSFString", node.getTypeName())
            objects.append(
                {
                    "id": node.getId(),
                    "name": name,
                    "def": node.getDef(),
                    "type": node.getTypeName(),
                    "T_base_owner": node.getPose(robot),
                    "T_world_owner": node.getPose(),
                    "shapes": shapes(f.getSFNode(), np.eye(4)),
                }
            )
        for child in child_nodes(node):
            visit(child)

    visit(supervisor.getRoot())
    # PROTO internals may not be exposed through children. Device ancestors are actual
    # expanded nodes, also used by the project's existing kinematics telemetry.
    device_names = []
    joint_positions = {}
    motors = []
    for index in range(supervisor.getNumberOfDevices()):
        device = supervisor.getDeviceByIndex(index)
        device_names.append(device.getName())
        if type(device).__name__ == "PositionSensor":
            joint_positions[device.getName()] = device.getValue()
        if hasattr(device, "getMaxVelocity"):
            motors.append(
                {
                    "name": device.getName(),
                    "max_velocity": device.getMaxVelocity(),
                    "commanded_velocity": device.getVelocity(),
                }
            )
        node = supervisor.getFromDevice(device._tag)
        ancestry = set()
        while node is not None and node.getId() not in ancestry:
            ancestry.add(node.getId())
            visit(node)
            node = node.getParentNode()
    if not objects:
        raise RuntimeError("No bounding objects traversed; geometry export is incomplete")
    return {
        "status": "MEASURED live expanded boundingObject export; collision safety NOT CERTIFIED",
        "robot_id": robot.getId(),
        "self_collision_enabled": values(robot, "selfCollision", "getSFBool", None),
        "sim_time_s": supervisor.getTime(),
        "objects": objects,
        "motor_commands_sent": 0,
        "devices": device_names,
        "joint_positions": joint_positions,
        "motor_velocity_settings": motors,
        "scene": "development copy of factory initial pose; no TRAIN/VALIDATION/TEST task rollout",
        "unsupported_shapes": [
            {"owner": o["name"], "type": s["type"]}
            for o in objects
            for s in o["shapes"]
            if not s["supported"]
        ],
    }


def main() -> None:
    from controller import Supervisor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    for index in range(supervisor.getNumberOfDevices()):
        device = supervisor.getDeviceByIndex(index)
        if type(device).__name__ == "PositionSensor":
            device.enable(timestep)
    if supervisor.step(timestep) == -1:
        raise RuntimeError("probe world stopped")
    report = snapshot(supervisor)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
