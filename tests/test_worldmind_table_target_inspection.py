"""Inert checks for the five nested Table/SolidBox read-only geometry targets.

No test in this module launches Webots, opens a socket, moves the robot, solves IK, scores an
episode or reads TEST/TRAIN data. Runtime observations are synthetic rows; expected geometry comes
only from the pinned source files.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vlarobot.backends.webots.geometry_probe import inspect_table_targets
from vlarobot.backends.worldmind_webots_geometry_inspection import (
    ReadOnlyGeometryBackendClient,
)
from vlarobot.eval.worldmind_webots_geometry_inspection import (
    GeometryInspectionConfig,
    GeometryInspectionSnapshot,
    _expected_table_geometry,
    load_geometry_inspection_config,
    reconcile_table_targets,
    table_target_descriptor,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "worldmind_webots_geometry_inspection.yaml"
)


def _config() -> GeometryInspectionConfig:
    return load_geometry_inspection_config(CONFIG_PATH)


def _pose16(rotation: object, translation: object) -> list[float]:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix.reshape(16).tolist()


def _valid_observation(config: GeometryInspectionConfig) -> dict[str, Any]:
    expected = _expected_table_geometry(config)
    descriptor = table_target_descriptor(config)
    rows: list[dict[str, Any]] = []
    for index, box in enumerate(expected["boxes"]):
        name = str(box["name"])
        rows.append(
            {
                "source_name": name,
                "status": "observed",
                "reason": None,
                "source_protos": copy.deepcopy(descriptor["source_protos"]),
                "runtime_node": {
                    "id": 200 + index,
                    "type": "SolidBox",
                    "base_type": "Solid",
                    "name": name,
                    "def": "",
                    "proto_type": "SolidBox",
                },
                "hierarchy": {
                    "reachable": True,
                    "table_ancestor": {
                        "id": 199,
                        "type": "Table",
                        "base_type": "Solid",
                        "name": "",
                        "def": "",
                        "proto_type": "Table",
                    },
                    "parent_chain": [{"type": "Table", "proto_type": "Table"}],
                    "children": [],
                    "bounding_object": {"type": "Box", "base_type": "Box"},
                },
                "evaluated": {
                    "translation_m": list(box["local_translation_m"]),
                    "rotation_axis_angle": list(box["local_rotation_axis_angle"]),
                    "size_m": list(box["size_m"]),
                    "enable_bounding_object": True,
                    "bounding_object_type": "Box",
                    "bounding_object_size_m": list(box["size_m"]),
                },
                "transforms": {
                    "T_world": _pose16(box["world_rotation_matrix"], box["world_translation_m"]),
                    "T_base": _pose16(box["base_rotation_matrix"], box["base_translation_m"]),
                },
                "unsupported_reasons": [],
            }
        )
    return {
        "schema_version": 1,
        "interface": copy.deepcopy(descriptor),
        "rows": rows,
        "errors": [],
        "robot_base_node_id": 77,
        "table_candidates": [{"type": "Table", "proto_type": "Table"}],
        "traversal": "supported getBaseNodeField(children) expansion of the runtime Table PROTO",
    }


def _snapshot(config: GeometryInspectionConfig, observation: object) -> GeometryInspectionSnapshot:
    return GeometryInspectionSnapshot(  # type: ignore[arg-type]
        ping={},
        observation=None,
        ground_truth=None,
        identity={},
        captured_at_utc="2026-11-18T00:00:00+00:00",
        age_s=0.0,
        dependency_availability={},
        call_counts={"inspect_geometry": 1},
        geometry={"table_targets": observation},
    )


def _reconcile(config: GeometryInspectionConfig, observation: object) -> dict[str, object]:
    report = reconcile_table_targets(config, _snapshot(config, observation))
    assert isinstance(report, dict)
    return report


def _target(report: dict[str, object], name: str) -> dict[str, object]:
    targets = report["targets"]
    assert isinstance(targets, list)
    matches = [
        item for item in targets if isinstance(item, dict) and item.get("source_name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_all_five_runtime_boxes_match_independently_derived_source_geometry():
    config = _config()
    report = _reconcile(config, _valid_observation(config))
    assert report["runtime_grounded"] is True
    assert report["status"] == "runtime_grounded"
    assert report["summary"] == {
        "Matched": 5,
        "Mismatched": 0,
        "Missing": 0,
        "Unsupported": 0,
    }
    assert report["blockers"] == []
    targets = report["targets"]
    assert isinstance(targets, list)
    assert [item["source_name"] for item in targets] == list(config.table_targets.source_names)
    assert all(item["verdict"] == "Matched" for item in targets)
    assert all(item["runtime_verification"] == "VERIFIED" for item in targets)
    source_provenance = report["source_provenance"]
    assert isinstance(source_provenance, dict)
    table_proto = source_provenance["table_proto"]
    solid_proto = source_provenance["solid_box_proto"]
    assert isinstance(table_proto, dict)
    assert isinstance(solid_proto, dict)
    assert table_proto["sha256"] == (
        "d7f2a20f3f6c3f515ce96ae10665941813b226c16442abf92d17e62259379bdb"
    )
    assert solid_proto["sha256"] == (
        "f9e18612ce325bc7abfd2d94cbd1c391a244f54ce5a2789a97fd1f901dc0dfba"
    )
    tolerances = report["tolerances"]
    assert isinstance(tolerances, dict)
    assert set(tolerances.values()) == {0.0001}


def test_missing_runtime_box_is_unverified_and_source_values_are_not_substituted():
    config = _config()
    observation = _valid_observation(config)
    observation["rows"] = [row for row in observation["rows"] if row["source_name"] != "foot 3"]
    report = _reconcile(config, observation)
    target = _target(report, "foot 3")
    assert target["verdict"] == "Missing"
    assert target["runtime_verification"] == "UNVERIFIED"
    assert "evaluated" not in target
    assert "transforms" not in target
    assert "expected" not in target
    assert report["runtime_grounded"] is False
    assert report["summary"]["Missing"] == 1
    assert any("foot 3: Missing" in str(item) for item in report["blockers"])


def test_incorrect_transform_and_dimension_are_mismatched():
    config = _config()
    observation = _valid_observation(config)
    transform_row = observation["rows"][1]
    transform_row["transforms"]["T_base"][3] += 0.05
    size_row = observation["rows"][2]
    size_row["evaluated"]["size_m"][0] += 0.02
    size_row["evaluated"]["bounding_object_size_m"][0] += 0.02
    report = _reconcile(config, observation)
    transform = _target(report, "foot 1")
    dimension = _target(report, "foot 2")
    assert transform["verdict"] == "Mismatched"
    assert dimension["verdict"] == "Mismatched"
    assert transform["runtime_verification"] == "UNVERIFIED"
    assert any(check["name"] == "T_base_translation_m" for check in transform["checks"])
    assert any(check["name"] == "size_m" for check in dimension["checks"])
    assert report["runtime_grounded"] is False


def test_disabled_bounding_object_fails_closed():
    config = _config()
    observation = _valid_observation(config)
    observation["rows"][0]["evaluated"]["enable_bounding_object"] = False
    report = _reconcile(config, observation)
    target = _target(report, "main frame")
    assert target["verdict"] == "Mismatched"
    check = next(item for item in target["checks"] if item["name"] == "enable_bounding_object")
    assert check["matched"] is False
    assert report["runtime_grounded"] is False


def test_nested_proto_expansion_failure_is_unsupported():
    config = _config()
    observation = _valid_observation(config)
    row = observation["rows"][3]
    row["status"] = "unsupported"
    row["hierarchy"]["reachable"] = False
    row["unsupported_reasons"] = ["children field could not be expanded"]
    report = _reconcile(config, observation)
    target = _target(report, "foot 3")
    assert target["verdict"] == "Unsupported"
    assert target["runtime_verification"] == "UNVERIFIED"
    assert report["runtime_grounded"] is False


def test_unsupported_runtime_node_type_is_unsupported():
    config = _config()
    observation = _valid_observation(config)
    runtime_node = observation["rows"][4]["runtime_node"]
    runtime_node.update({"type": "Box", "base_type": "Box", "proto_type": None})
    report = _reconcile(config, observation)
    target = _target(report, "foot 4")
    assert target["verdict"] == "Unsupported"
    assert target["runtime_verification"] == "UNVERIFIED"
    assert report["runtime_grounded"] is False


def test_duplicate_runtime_rows_are_unsupported():
    config = _config()
    observation = _valid_observation(config)
    observation["rows"].append(copy.deepcopy(observation["rows"][0]))
    report = _reconcile(config, observation)
    target = _target(report, "main frame")
    assert target["verdict"] == "Unsupported"
    assert target["runtime_verification"] == "UNVERIFIED"
    assert report["summary"]["Unsupported"] == 1
    assert report["runtime_grounded"] is False


def test_capture_forwards_only_selector_descriptor_and_blocks_forbidden_calls():
    config = _config()
    descriptor = table_target_descriptor(config)

    class LowLevel:
        def __init__(self) -> None:
            self.arguments: list[object] = []

        def ping(self) -> dict[str, object]:
            return {"status": "ok"}

        def inspect_geometry(self, table_targets: object = None) -> dict[str, object]:
            self.arguments.append(table_targets)
            return {}

        def close(self) -> None:
            return None

    low = LowLevel()
    client = ReadOnlyGeometryBackendClient(low)
    assert client.inspect_geometry(descriptor) == {}
    assert low.arguments == [descriptor]
    assert "expected" not in str(descriptor)
    blocked_calls = {
        "apply": (None,),
        "step": (),
        "reset": (None, 0),
        "export_kinematics": (),
        "inspect": (),
        "get_ground_truth": (),
    }
    for name, args in blocked_calls.items():
        with pytest.raises(RuntimeError, match=f"{name} is forbidden"):
            getattr(client, name)(*args)
    counts = client.call_counts()
    assert counts["inspect_geometry"] == 1
    assert all(counts[name] == 1 for name in ("apply", "step", "reset"))


class _Field:
    def __init__(self, value: object) -> None:
        self.value = value

    def getSFVec3f(self) -> object:
        return self.value

    def getSFRotation(self) -> object:
        return self.value

    def getSFString(self) -> object:
        return self.value

    def getSFBool(self) -> object:
        return self.value

    def getSFNode(self) -> object:
        return self.value

    def getCount(self) -> int:
        return len(self.value)  # type: ignore[arg-type]

    def getMFNode(self, index: int) -> object:
        return self.value[index]  # type: ignore[index]


class _Proto:
    def __init__(self, name: str) -> None:
        self.name = name

    def getTypeName(self) -> str:
        return self.name


class _Node:
    def __init__(
        self,
        identifier: int,
        type_name: str,
        base_type: str,
        *,
        proto_type: str | None = None,
        fields: dict[str, object] | None = None,
        base_fields: dict[str, object] | None = None,
        children: list[_Node] | None = None,
        parent: _Node | None = None,
        world_pose: list[float] | None = None,
        base_pose: list[float] | None = None,
        broken_children: bool = False,
    ) -> None:
        self.identifier = identifier
        self.type_name = type_name
        self.base_type = base_type
        self.proto_type = proto_type
        self.fields = dict(fields or {})
        self.base_fields = dict(base_fields or {})
        self.parent_node = parent
        self.world_pose = world_pose or np.eye(4).reshape(16).tolist()
        self.base_pose = base_pose or np.eye(4).reshape(16).tolist()
        self.broken_children = broken_children
        if children is not None:
            self.base_fields["children"] = _Field(children)

    def getId(self) -> int:
        return self.identifier

    def getDef(self) -> str:
        return ""

    def getParentNode(self) -> _Node | None:
        return self.parent_node

    def isProto(self) -> bool:
        return self.proto_type is not None

    def getProto(self) -> _Proto | None:
        return _Proto(self.proto_type) if self.proto_type is not None else None

    def getTypeName(self) -> str:
        return self.type_name

    def getBaseTypeName(self) -> str:
        return self.base_type

    def getField(self, name: str) -> object:
        return self.fields.get(name)

    def getBaseNodeField(self, name: str) -> object:
        if name == "children" and self.broken_children:
            raise RuntimeError("children field unavailable")
        return self.base_fields.get(name)

    def getPose(self, from_node: object = None) -> list[float]:
        return self.base_pose if from_node is not None else self.world_pose


class _Supervisor:
    def __init__(self, root: _Node, robot: _Node) -> None:
        self.root = root
        self.robot = robot

    def getRoot(self) -> _Node:
        return self.root

    def getSelf(self) -> _Node:
        return self.robot


def _fake_scene(
    config: GeometryInspectionConfig,
    *,
    broken_table: bool = False,
    omit_last_box: bool = False,
) -> _Supervisor:
    expected = _expected_table_geometry(config)
    definitions = expected["boxes"][:-1] if omit_last_box else expected["boxes"]
    boxes: list[_Node] = []
    for index, box in enumerate(definitions):
        bounding = _Node(
            400 + index,
            "Box",
            "Box",
            fields={"size": _Field(list(box["size_m"]))},
        )
        boxes.append(
            _Node(
                300 + index,
                "SolidBox",
                "Solid",
                proto_type="SolidBox",
                fields={
                    "translation": _Field(list(box["local_translation_m"])),
                    "rotation": _Field(list(box["local_rotation_axis_angle"])),
                    "size": _Field(list(box["size_m"])),
                    "name": _Field(str(box["name"])),
                    "enableBoundingObject": _Field(True),
                },
                base_fields={"boundingObject": _Field(bounding)},
                world_pose=_pose16(box["world_rotation_matrix"], box["world_translation_m"]),
                base_pose=_pose16(box["base_rotation_matrix"], box["base_translation_m"]),
            )
        )
    robot = _Node(100, "Tiago", "Robot")
    table = _Node(
        200,
        "Table",
        "Solid",
        proto_type="Table",
        children=boxes,
        parent=None,
        broken_children=broken_table,
    )
    for box in boxes:
        box.parent_node = table
    root = _Node(1, "Group", "Group", children=[table])
    table.parent_node = root
    return _Supervisor(root, robot)


def test_supervisor_probe_captures_all_five_boxes_without_source_substitution():
    config = _config()
    descriptor = table_target_descriptor(config)
    observation = inspect_table_targets(
        _fake_scene(config), descriptor, _Node(100, "Tiago", "Robot")
    )
    assert observation["errors"] == []
    rows = observation["rows"]
    assert [row["source_name"] for row in rows] == list(config.table_targets.source_names)
    assert all(row["status"] == "observed" for row in rows)
    assert all(row["unsupported_reasons"] == [] for row in rows)
    assert all(row["source_protos"] == descriptor["source_protos"] for row in rows)
    assert all(row["runtime_node"]["proto_type"] == "SolidBox" for row in rows)
    assert all(row["hierarchy"]["table_ancestor"]["proto_type"] == "Table" for row in rows)
    assert all(row["evaluated"]["enable_bounding_object"] is True for row in rows)
    assert all(row["evaluated"]["bounding_object_type"] == "Box" for row in rows)
    assert all(len(row["transforms"]["T_world"]) == 16 for row in rows)
    assert all(len(row["transforms"]["T_base"]) == 16 for row in rows)


def test_supervisor_probe_records_missing_and_failed_proto_expansion():
    config = _config()
    descriptor = table_target_descriptor(config)
    missing = inspect_table_targets(
        _fake_scene(config, omit_last_box=True), descriptor, _Node(100, "Tiago", "Robot")
    )
    missing_rows = {row["source_name"]: row for row in missing["rows"]}
    assert missing_rows["foot 4"]["status"] == "missing"
    assert missing_rows["foot 4"]["evaluated"] == {}
    assert all(
        row["status"] == "observed" for name, row in missing_rows.items() if name != "foot 4"
    )

    broken = inspect_table_targets(
        _fake_scene(config, broken_table=True), descriptor, _Node(100, "Tiago", "Robot")
    )
    assert any("children traversal failed" in item for item in broken["errors"])
    assert all(row["status"] == "missing" for row in broken["rows"])
    assert all(row["evaluated"] == {} for row in broken["rows"])
    assert all(row["transforms"] == {} for row in broken["rows"])


def test_capture_rejects_descriptor_that_carries_expected_geometry():
    config = _config()
    descriptor = table_target_descriptor(config)
    bad = copy.deepcopy(descriptor)
    bad["expected_geometry"] = {"foot 1": {"size_m": [1, 1, 1]}}
    with pytest.raises(ValueError, match="unexpected keys"):
        inspect_table_targets(_fake_scene(config), bad, _Node(100, "Tiago", "Robot"))
