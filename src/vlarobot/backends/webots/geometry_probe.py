"""Read-only live boundingObject export in a separate development world; no motor commands."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

TABLE_TARGET_OBSERVATION_SCHEMA_VERSION = 1
_TABLE_DESCRIPTOR_KEYS = frozenset(
    {
        "schema_version",
        "parent_type",
        "nested_type",
        "bounding_shape_type",
        "source_names",
        "source_protos",
    }
)


def _target_vector(value: object, method: str, size: int) -> list[float] | None:
    if value is None or not hasattr(value, method):
        return None
    raw = getattr(value, method)()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != size:
        return None
    numbers = [float(item) for item in raw]
    return numbers if all(math.isfinite(number) for number in numbers) else None


class _VisitTracker:
    """Deduplicate by runtime ID when valid, by wrapper identity when ID is unavailable.

    Webots internal PROTO nodes may report ``getId() == -1``. That sentinel is not a unique
    runtime ID: distinct ``SolidBox`` feet can all report it. A strong reference is retained for
    identity-keyed nodes so Python ``id()`` values cannot be recycled during a traversal. The
    original sentinel is still exported in evidence; this key is internal only.
    """

    def __init__(self) -> None:
        self._ids: set[int] = set()
        self._nodes: dict[int, Any] = {}

    def first_visit(self, node: Any) -> bool:
        try:
            identifier = node.getId()
        except Exception:
            identifier = -1
        if type(identifier) is int and identifier >= 0:
            if identifier in self._ids:
                return False
            self._ids.add(identifier)
            return True
        key = id(node)
        if key in self._nodes:
            return False
        self._nodes[key] = node
        return True


def inspect_table_targets(
    supervisor: Any, descriptor: Mapping[str, object], robot: Any
) -> dict[str, Any]:
    """Capture raw runtime rows for the five nested Table/SolidBox collision boxes.

    This function only reads fields and poses through the supported Supervisor API. It never
    substitutes source-derived geometry: unavailable fields remain ``None`` and uninspectable
    nodes remain ``missing`` or ``unsupported``.
    """
    if not isinstance(descriptor, Mapping) or set(descriptor) != _TABLE_DESCRIPTOR_KEYS:
        raise ValueError("table target descriptor has unexpected keys")
    if descriptor["schema_version"] != TABLE_TARGET_OBSERVATION_SCHEMA_VERSION:
        raise ValueError("table target descriptor schema_version is unsupported")
    source_names = descriptor["source_names"]
    if (
        not isinstance(source_names, Sequence)
        or isinstance(source_names, (str, bytes))
        or len(source_names) == 0
        or any(not isinstance(name, str) or not name for name in source_names)
        or len(set(source_names)) != len(source_names)
    ):
        raise ValueError("table target descriptor source_names are invalid")
    for key in ("parent_type", "nested_type", "bounding_shape_type"):
        if not isinstance(descriptor[key], str) or not descriptor[key]:
            raise ValueError(f"table target descriptor {key} is invalid")
    if not isinstance(descriptor["source_protos"], Mapping):
        raise ValueError("table target descriptor source_protos are invalid")
    traversal_errors: list[str] = []

    def raw_field(node: Any, name: str) -> Any:
        if node is None:
            return None
        try:
            base = node.getBaseNodeField(name) if node.isProto() else None
        except Exception:
            base = None
        if base is not None:
            return base
        try:
            return node.getField(name)
        except Exception:
            return None

    def base_field(node: Any, name: str) -> Any:
        if node is None:
            return None
        try:
            return node.getBaseNodeField(name)
        except Exception:
            return None

    def interface_field(node: Any, name: str) -> Any:
        if node is None:
            return None
        try:
            return node.getField(name)
        except Exception:
            return None

    def scalar(node: Any, name: str, method: str) -> Any:
        field_value = raw_field(node, name)
        if field_value is None or not hasattr(field_value, method):
            return None
        try:
            return getattr(field_value, method)()
        except Exception:
            return None

    def text(node: Any, name: str) -> str | None:
        value = scalar(node, name, "getSFString")
        return value if isinstance(value, str) and value else None

    def node_children(node: Any) -> list[Any]:
        result: list[Any] = []
        children_field = None
        try:
            if node is not None and node.isProto():
                children_field = node.getBaseNodeField("children")
        except Exception as exc:
            traversal_errors.append(f"children traversal failed: {type(exc).__name__}")
        if children_field is None:
            children_field = base_field(node, "children")
        if children_field is None:
            children_field = raw_field(node, "children")
        if children_field is not None and hasattr(children_field, "getCount"):
            try:
                count = children_field.getCount()
                for index in range(count):
                    child = children_field.getMFNode(index)
                    if child is not None:
                        result.append(child)
            except Exception as exc:
                traversal_errors.append(f"children traversal failed: {type(exc).__name__}")
        endpoint = base_field(node, "endPoint") or raw_field(node, "endPoint")
        endpoint_node = None
        if endpoint is not None and hasattr(endpoint, "getSFNode"):
            try:
                endpoint_node = endpoint.getSFNode()
            except Exception:
                endpoint_node = None
        if endpoint_node is not None:
            result.append(endpoint_node)
        return result

    def identity(node: Any) -> dict[str, object]:
        if node is None:
            return {}
        proto_type = None
        try:
            proto = node.getProto() if node.isProto() else None
            proto_type = proto.getTypeName() if proto is not None else None
        except Exception:
            proto_type = None

        def safe(method: str) -> object:
            try:
                return getattr(node, method)()
            except Exception:
                return None

        node_id = safe("getId")
        return {
            "id": int(node_id) if isinstance(node_id, int) else None,
            "type": safe("getTypeName"),
            "base_type": safe("getBaseTypeName"),
            "name": text(node, "name"),
            "def": safe("getDef"),
            "proto_type": proto_type,
        }

    def type_matches(node: Any, expected: str) -> bool:
        info = identity(node)
        return expected in {info.get("type"), info.get("proto_type"), info.get("base_type")}

    def walk(start_nodes: Sequence[Any]) -> list[Any]:
        result: list[Any] = []
        visited = _VisitTracker()
        queue = [(node, 0) for node in start_nodes]
        while queue:
            node, depth = queue.pop(0)
            if not visited.first_visit(node):
                continue
            result.append(node)
            if depth < 12:
                queue.extend((child, depth + 1) for child in node_children(node))
        return result

    def ancestors(node: Any) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        visited = _VisitTracker()
        visited.first_visit(node)
        try:
            parent = node.getParentNode()
        except Exception:
            return result
        while parent is not None and len(result) < 16:
            if not visited.first_visit(parent):
                break
            result.append(identity(parent))
            try:
                parent = parent.getParentNode()
            except Exception:
                break
        return result

    all_nodes = walk([supervisor.getRoot()])
    table_nodes = [node for node in all_nodes if type_matches(node, str(descriptor["parent_type"]))]
    errors: list[str] = list(dict.fromkeys(traversal_errors))
    if not table_nodes:
        errors.append("no runtime Table node was reachable through supported Supervisor traversal")
    if len(table_nodes) > 1:
        errors.append("multiple runtime Table nodes were reachable; target is ambiguous")

    nested_by_name: dict[str, Any] = {}
    if len(table_nodes) == 1:
        for node in walk(node_children(table_nodes[0])):
            if not type_matches(node, str(descriptor["nested_type"])):
                continue
            name = text(node, "name")
            if isinstance(name, str) and name in source_names and name not in nested_by_name:
                nested_by_name[name] = node

    rows: list[dict[str, Any]] = []
    for name in source_names:
        row: dict[str, Any] = {
            "source_name": name,
            "status": "missing",
            "reason": None,
            "source_protos": dict(descriptor["source_protos"]),
            "runtime_node": {},
            "hierarchy": {"reachable": False},
            "evaluated": {},
            "transforms": {},
            "unsupported_reasons": [],
        }
        node = nested_by_name.get(name)
        if node is None:
            row["reason"] = "nested runtime node was not observed"
            rows.append(row)
            continue
        row["runtime_node"] = identity(node)
        chain = ancestors(node)
        table_ancestor = next(
            (
                item
                for item in chain
                if descriptor["parent_type"]
                in {
                    item.get("type"),
                    item.get("proto_type"),
                    item.get("base_type"),
                }
            ),
            {},
        )
        bounding_node = None
        bounding_field = base_field(node, "boundingObject") or raw_field(node, "boundingObject")
        if bounding_field is not None and hasattr(bounding_field, "getSFNode"):
            try:
                bounding_node = bounding_field.getSFNode()
            except Exception:
                bounding_node = None
        row["hierarchy"] = {
            "reachable": True,
            "table_ancestor": table_ancestor,
            "parent_chain": chain,
            "children": [identity(child) for child in node_children(node)],
            "bounding_object": identity(bounding_node),
        }

        translation = _target_vector(interface_field(node, "translation"), "getSFVec3f", 3)
        rotation = _target_vector(interface_field(node, "rotation"), "getSFRotation", 4)
        size = _target_vector(interface_field(node, "size"), "getSFVec3f", 3)
        enable_raw = scalar(node, "enableBoundingObject", "getSFBool")
        bounding_type = None
        bounding_size = None
        if bounding_node is not None:
            try:
                bounding_type = bounding_node.getTypeName()
            except Exception:
                bounding_type = None
            if bounding_type == descriptor["bounding_shape_type"]:
                bounding_size = _target_vector(
                    interface_field(bounding_node, "size"), "getSFVec3f", 3
                )
        pose_world = None
        pose_base = None
        try:
            pose_world = [float(value) for value in node.getPose()]
        except Exception:
            pose_world = None
        try:
            pose_base = [float(value) for value in node.getPose(robot)]
        except Exception:
            pose_base = None
        if pose_world is not None and not all(math.isfinite(value) for value in pose_world):
            pose_world = None
        if pose_base is not None and not all(math.isfinite(value) for value in pose_base):
            pose_base = None

        row["evaluated"] = {
            "translation_m": translation,
            "rotation_axis_angle": rotation,
            "size_m": size,
            "enable_bounding_object": enable_raw,
            "bounding_object_type": bounding_type,
            "bounding_object_size_m": bounding_size,
        }
        row["transforms"] = {"T_world": pose_world, "T_base": pose_base}
        unsupported: list[str] = []
        if not type_matches(node, str(descriptor["nested_type"])):
            unsupported.append("runtime node type is not the pinned nested PROTO")
        if not table_ancestor:
            unsupported.append("runtime Table ancestry is not observed")
        if translation is None:
            unsupported.append("translation is not readable")
        if rotation is None:
            unsupported.append("rotation is not readable")
        if size is None:
            unsupported.append("size is not readable")
        if enable_raw is None:
            unsupported.append("enableBoundingObject is not readable")
        elif type(enable_raw) is not bool and not (
            type(enable_raw) is int and enable_raw in (0, 1)
        ):
            unsupported.append("enableBoundingObject is not a documented SFBool")
        if bounding_type != descriptor["bounding_shape_type"]:
            unsupported.append("boundingObject is not the pinned Box shape")
        elif bounding_size is None:
            unsupported.append("boundingObject Box.size is not readable")
        if pose_world is None or len(pose_world) != 16:
            unsupported.append("world pose is not readable")
        if pose_base is None or len(pose_base) != 16:
            unsupported.append("robot-base pose is not readable")
        row["unsupported_reasons"] = unsupported
        row["status"] = "unsupported" if unsupported else "observed"
        row["reason"] = None if not unsupported else "; ".join(unsupported)
        rows.append(row)

    return {
        "schema_version": TABLE_TARGET_OBSERVATION_SCHEMA_VERSION,
        "interface": dict(descriptor),
        "rows": rows,
        "errors": errors,
        "robot_base_node_id": int(robot.getId()),
        "table_candidates": [identity(node) for node in table_nodes],
        "traversal": "supported getBaseNodeField(children) expansion of the runtime Table PROTO",
    }


def snapshot(supervisor: Any, table_targets: Mapping[str, object] | None = None) -> dict[str, Any]:
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

    visited = _VisitTracker()
    objects = []

    def visit(node):
        if not visited.first_visit(node):
            return
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
        ancestry = _VisitTracker()
        while node is not None and ancestry.first_visit(node):
            visit(node)
            node = node.getParentNode()
    if not objects:
        raise RuntimeError("No bounding objects traversed; geometry export is incomplete")
    report: dict[str, Any] = {
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
    if table_targets is not None:
        report["table_targets"] = inspect_table_targets(supervisor, table_targets, robot)
    return report


def main() -> None:
    # Only importable inside the Webots runtime; the SDK is never present in CI.
    from controller import Supervisor  # ty: ignore[import-not-found]

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
