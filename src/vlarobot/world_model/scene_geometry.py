"""Pinned Webots scene geometry reduced to conservative robot-base-frame primitives.

This is a pure, file-only loader. It never imports Webots, IKPy, ROS or MuJoCo, opens a socket,
launches a simulator, solves IK, applies an action, steps physics or accesses a data split. Every
source hash, transform and primitive bound is checked; unsupported geometry remains listed and the
result is explicitly not a motion certificate.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
import yaml

from vlarobot.core.types import Pose3D, Vec3
from vlarobot.world_model.arm_geometry import (
    CarriedObjectEnvelope,
    FloorHalfSpace,
    GripperEnvelope,
    StaticBox,
    StaticSceneGeometry,
)

SCENE_GEOMETRY_SCHEMA_VERSION = 1
_WORLD_POSE_RE = re.compile(r"\bDEF\s+TIAGO_ROBOT\s+Tiago\s*\{(.*?)\n\s*\}", re.DOTALL)
_TRANSLATION_RE = re.compile(r"\btranslation\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)")
_ROTATION_RE = re.compile(r"\brotation\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)")
_TABLE_RE = re.compile(r"\bTable\s*\{(.*?)\n\s*\}", re.DOTALL)
_SIZE_RE = re.compile(r"\bsize\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_angle_matrix(axis_angle: Sequence[float]) -> np.ndarray:
    """Convert Webots xyzw axis-angle to a proper rotation matrix."""
    values = np.asarray(axis_angle, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("invalid axis-angle pose")
    axis = values[:3]
    angle = float(values[3])
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0 or angle == 0.0:
        rotation = np.eye(3)
    else:
        unit = axis / norm
        cross = np.asarray(
            [
                [0.0, -unit[2], unit[1]],
                [unit[2], 0.0, -unit[0]],
                [-unit[1], unit[0], 0.0],
            ]
        )
        rotation = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    if not _proper_rotation(rotation):
        raise ValueError("axis-angle pose is not a proper rotation")
    return rotation


def pose_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = axis_angle_matrix(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite pose matrix")
    return matrix


def matrix_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=np.float64)
    if not _proper_rotation(rotation[:3, :3]):
        raise ValueError("cannot convert a non-rigid transform to a pose")
    trace = float(np.trace(rotation[:3, :3]))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation[:3, :3])))
        if index == 0:
            scale = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return (
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )


def _proper_rotation(rotation: np.ndarray) -> bool:
    values = np.asarray(rotation, dtype=float)
    return bool(
        values.shape == (3, 3)
        and np.isfinite(values).all()
        and np.allclose(values.T @ values, np.eye(3), atol=1e-8, rtol=0)
        and math.isclose(float(np.linalg.det(values)), 1.0, abs_tol=1e-8)
    )


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


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@overload
def _vector(value: object, name: str, size: Literal[3]) -> tuple[float, float, float]: ...


@overload
def _vector(value: object, name: str, size: Literal[4]) -> tuple[float, float, float, float]: ...


@overload
def _vector(value: object, name: str, size: int) -> tuple[float, ...]: ...


def _vector(value: object, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    values = tuple(float(item) for item in value)
    if len(values) != size or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return values


def _array3(values: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return (float(array[0]), float(array[1]), float(array[2]))


def _world_block(world: str, pattern: re.Pattern[str], name: str) -> str:
    match = pattern.search(world)
    if match is None:
        raise ValueError(f"pinned world is missing {name}")
    return match.group(1)


def _world_translation(block: str, name: str) -> tuple[float, float, float]:
    match = _TRANSLATION_RE.search(block)
    if match is None:
        raise ValueError(f"pinned world is missing {name} translation")
    values = tuple(float(value) for value in match.groups())
    if len(values) != 3:
        raise ValueError(f"pinned world {name} translation is malformed")
    return (values[0], values[1], values[2])


def _world_rotation(block: str, name: str) -> tuple[float, float, float, float]:
    match = _ROTATION_RE.search(block)
    if match is None:
        raise ValueError(f"pinned world is missing {name} rotation")
    values = tuple(float(value) for value in match.groups())
    if len(values) != 4:
        raise ValueError(f"pinned world {name} rotation is malformed")
    return (values[0], values[1], values[2], values[3])


def _verify_source(path: Path, expected: str, name: str) -> str:
    digest = sha256_file(path)
    if digest != _sha_text(expected, f"hashes.{name}"):
        raise ValueError(f"{name} source hash mismatch or missing file: {path}")
    return digest


def _load_registration_matrix(path: Path) -> np.ndarray:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, Mapping)
        or raw.get("source_split") != "train"
        or not raw.get("evidence")
    ):
        raise ValueError("pinned registration lacks TRAIN provenance")
    matrix = np.asarray(raw.get("T_robot_base_urdf_root"), dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not _proper_rotation(matrix[:3, :3])
    ):
        raise ValueError("pinned registration must be a finite proper rigid transform")
    return matrix


@dataclass(frozen=True)
class SceneCollisionCatalog:
    """Verified, conservative geometry derived only from pinned sources."""

    source_path: Path
    status: str
    hashes: Mapping[str, str]
    T_world_base: np.ndarray
    T_base_world: np.ndarray
    T_base_urdf: np.ndarray
    T_world_urdf: np.ndarray
    expected_base_pose_world: Pose3D
    base_pose_tolerance: Mapping[str, object]
    geometry: StaticSceneGeometry
    unsupported: tuple[str, ...]

    @property
    def motion_ready(self) -> bool:
        return False

    def report(self) -> dict[str, object]:
        return {
            "schema_version": SCENE_GEOMETRY_SCHEMA_VERSION,
            "status": self.status,
            "motion_ready": self.motion_ready,
            "source_hashes": dict(self.hashes),
            "T_world_base": self.T_world_base.tolist(),
            "T_base_world": self.T_base_world.tolist(),
            "T_base_urdf": self.T_base_urdf.tolist(),
            "T_world_urdf": self.T_world_urdf.tolist(),
            "expected_base_pose_world": {
                "position": list(self.expected_base_pose_world.position.as_array()),
                "orientation": list(self.expected_base_pose_world.orientation),
            },
            "base_pose_tolerance": dict(self.base_pose_tolerance),
            "static_boxes": [
                {
                    "name": box.name,
                    "center_base": list(box.center_base),
                    "half_base": list(box.half_base),
                    "source": box.source,
                }
                for box in self.geometry.boxes
            ],
            "floor_planes": [
                {
                    "name": floor.name,
                    "point_base": list(floor.point_base),
                    "normal_base": list(floor.normal_base),
                    "source": floor.source,
                }
                for floor in self.geometry.floors
            ],
            "gripper_envelope": {
                "name": self.geometry.gripper.name,
                "frame": self.geometry.gripper.frame,
                "ee_to_front_translation_m": list(self.geometry.gripper.ee_to_front_translation_m),
                "start_tool": list(self.geometry.gripper.start_tool),
                "end_tool": list(self.geometry.gripper.end_tool),
                "radius_m": self.geometry.gripper.radius_m,
                "source": self.geometry.gripper.source,
            }
            if self.geometry.gripper is not None
            else None,
            "carried_objects": [
                {
                    "name": item.name,
                    "radius_m": item.radius_m,
                    "height_m": item.height_m,
                    "source": item.source,
                }
                for item in self.geometry.carried_objects
            ],
            "supported_object_ids": sorted(self.geometry.supported_object_ids),
            "unsupported": list(self.unsupported),
        }


def load_scene_collision_catalog(path: Path) -> SceneCollisionCatalog:
    """Load and independently verify the pinned scene geometry catalog."""
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "scene collision catalog")
    if set(root) != {
        "schema_version",
        "status",
        "source",
        "sources",
        "hashes",
        "robot_world_pose",
        "base_pose_tolerance",
        "static_boxes",
        "floor_planes",
        "gripper_envelope",
        "can_envelope",
        "unsupported",
    }:
        raise ValueError("scene collision catalog has unexpected top-level keys")
    if root["schema_version"] != SCENE_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported scene collision catalog schema_version")
    status = _text(root["status"], "status")
    if status != "diagnostic_uncalibrated":
        raise ValueError("scene collision catalog status must be diagnostic_uncalibrated")

    sources = _mapping(root["sources"], "sources")
    if set(sources) != {"world", "registration", "tiago", "floor", "table", "can", "gripper"}:
        raise ValueError("scene catalog sources have unexpected keys")
    paths = {
        name: _resolve(source.parent, value, f"sources.{name}") for name, value in sources.items()
    }
    hashes = _mapping(root["hashes"], "hashes")
    if set(hashes) != {
        "world_sha256",
        "registration_sha256",
        "tiago_sha256",
        "floor_sha256",
        "table_sha256",
        "can_sha256",
        "gripper_sha256",
    }:
        raise ValueError("scene catalog hashes have unexpected keys")
    digests = {
        name: _verify_source(path, _sha_text(hashes[f"{name}_sha256"], f"hashes.{name}"), name)
        for name, path in paths.items()
    }

    world = paths["world"].read_text(encoding="utf-8")
    robot_block = _world_block(world, _WORLD_POSE_RE, "DEF TIAGO_ROBOT")
    world_translation = _world_translation(robot_block, "TIAGO_ROBOT")
    world_rotation = _world_rotation(robot_block, "TIAGO_ROBOT")
    pose = _mapping(root["robot_world_pose"], "robot_world_pose")
    if set(pose) != {"translation", "rotation"}:
        raise ValueError("robot_world_pose has unexpected keys")
    catalog_translation = _vector(pose["translation"], "robot_world_pose.translation", 3)
    catalog_rotation = _vector(pose["rotation"], "robot_world_pose.rotation", 4)
    if not np.allclose(
        catalog_translation, world_translation, atol=1e-9, rtol=0
    ) or not np.allclose(catalog_rotation, world_rotation, atol=1e-12, rtol=0):
        raise ValueError("catalog robot pose does not match the pinned world file")

    T_world_base = pose_matrix(catalog_translation, catalog_rotation)
    T_base_world = np.linalg.inv(T_world_base)
    T_base_urdf = _load_registration_matrix(paths["registration"])
    T_world_urdf = T_world_base @ T_base_urdf
    if not _proper_rotation(T_world_urdf[:3, :3]):
        raise ValueError("world/URDF registration composition is not rigid")
    expected_pose = Pose3D(Vec3(*map(float, T_world_base[:3, 3])), matrix_quaternion(T_world_base))
    raw_tolerance = _mapping(root["base_pose_tolerance"], "base_pose_tolerance")
    if set(raw_tolerance) != {"position_m", "orientation_rad", "source"}:
        raise ValueError("base_pose_tolerance has unexpected keys")
    tolerance_source = _text(raw_tolerance["source"], "base_pose_tolerance.source")
    if not tolerance_source:
        raise ValueError("base_pose_tolerance requires provenance")
    position_tolerance = _finite_scalar(
        raw_tolerance["position_m"], "base_pose_tolerance.position_m"
    )
    orientation_tolerance = _finite_scalar(
        raw_tolerance["orientation_rad"], "base_pose_tolerance.orientation_rad"
    )
    if position_tolerance < 0.0 or orientation_tolerance < 0.0:
        raise ValueError("base-pose tolerances must be finite and nonnegative")
    tolerances = {
        "position_m": position_tolerance,
        "orientation_rad": orientation_tolerance,
        "source": tolerance_source,
    }

    table_block = _world_block(world, _TABLE_RE, "Table")
    table_translation = _world_translation(table_block, "Table")
    table_size_match = _SIZE_RE.search(table_block)
    if table_size_match is None:
        raise ValueError("pinned world Table is missing size")
    table_size = _vector(table_size_match.groups(), "Table.size", 3)
    if any(value <= 0.0 for value in table_size):
        raise ValueError("table size must be positive")

    boxes: list[StaticBox] = []
    raw_boxes = root["static_boxes"]
    if not isinstance(raw_boxes, Sequence) or isinstance(raw_boxes, (str, bytes)):
        raise ValueError("static_boxes must be a list")
    for index, value in enumerate(raw_boxes):
        box = _mapping(value, f"static_boxes[{index}]")
        if set(box) != {"name", "world_center", "world_rotation", "size", "source", "supported"}:
            raise ValueError(f"static_boxes[{index}] has unexpected keys")
        if box["supported"] is not True:
            raise ValueError(f"static box {box['name']} is not supported")
        center = np.asarray(_vector(box["world_center"], "world_center", 3))
        size = np.asarray(_vector(box["size"], "size", 3))
        if np.any(size <= 0):
            raise ValueError("static box size must be positive")
        if _text(box["name"], "static box name") == "table":
            expected_center = np.asarray(table_translation) + np.asarray((0.0, 0.0, size[2] / 2))
            if not np.allclose(center, expected_center, atol=1e-7, rtol=0) or not np.allclose(
                size, table_size, atol=1e-9, rtol=0
            ):
                raise ValueError("table catalog box does not match the pinned world instance")
        axis_angle = _vector(box["world_rotation"], "world_rotation", 4)
        rotation = axis_angle_matrix(axis_angle)
        center_base = T_base_world[:3, :3] @ center + T_base_world[:3, 3]
        rotation_base = T_base_world[:3, :3] @ rotation
        half_base = np.abs(rotation_base) @ (size / 2.0)
        boxes.append(
            StaticBox(
                _text(box["name"], "static box name"),
                _array3(center_base, "static box center_base"),
                _array3(half_base, "static box half_base"),
                _text(box["source"], "static box source"),
            )
        )
    if not boxes:
        raise ValueError("static box catalog is empty")

    floors: list[FloorHalfSpace] = []
    raw_floors = root["floor_planes"]
    if not isinstance(raw_floors, Sequence) or isinstance(raw_floors, (str, bytes)):
        raise ValueError("floor_planes must be a list")
    for index, value in enumerate(raw_floors):
        floor = _mapping(value, f"floor_planes[{index}]")
        if set(floor) != {"name", "world_point", "world_normal", "source", "supported"}:
            raise ValueError(f"floor_planes[{index}] has unexpected keys")
        if floor["supported"] is not True:
            raise ValueError(f"floor {floor['name']} is not supported")
        point = np.asarray(_vector(floor["world_point"], "world_point", 3))
        normal = np.asarray(_vector(floor["world_normal"], "world_normal", 3))
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            raise ValueError("floor normal must be nonzero")
        normal = normal / norm
        point_base = T_base_world[:3, :3] @ point + T_base_world[:3, 3]
        normal_base = T_base_world[:3, :3] @ normal
        normal_base /= np.linalg.norm(normal_base)
        floors.append(
            FloorHalfSpace(
                _text(floor["name"], "floor name"),
                _array3(point_base, "floor point_base"),
                _array3(normal_base, "floor normal_base"),
                _text(floor["source"], "floor source"),
            )
        )
    if not floors:
        raise ValueError("floor catalog is empty")

    raw_gripper = _mapping(root["gripper_envelope"], "gripper_envelope")
    if set(raw_gripper) != {
        "name",
        "frame",
        "ee_to_front_translation_m",
        "start_tool",
        "end_tool",
        "radius_m",
        "max_point_speed_m_s",
        "source",
        "supported",
    }:
        raise ValueError("gripper_envelope has unexpected keys")
    gripper_frame = _text(raw_gripper["frame"], "gripper.frame")
    if gripper_frame not in {"front", "wrist_ft_tool_link"}:
        raise ValueError("gripper envelope is unsupported or uses an unexpected frame")
    if raw_gripper["supported"] is not True:
        raise ValueError("gripper envelope is unsupported")
    ee_to_front = _vector(
        raw_gripper["ee_to_front_translation_m"], "gripper.ee_to_front_translation_m", 3
    )
    if gripper_frame == "wrist_ft_tool_link" and any(abs(value) > 0.0 for value in ee_to_front):
        raise ValueError("legacy wrist-frame gripper envelope cannot carry an ee-to-front offset")
    gripper = GripperEnvelope(
        _text(raw_gripper["name"], "gripper name"),
        _vector(raw_gripper["start_tool"], "gripper.start_tool", 3),
        _vector(raw_gripper["end_tool"], "gripper.end_tool", 3),
        _finite_scalar(raw_gripper["radius_m"], "gripper.radius_m"),
        _finite_scalar(raw_gripper["max_point_speed_m_s"], "gripper.max_point_speed_m_s"),
        _text(raw_gripper["source"], "gripper source"),
        frame=gripper_frame,
        ee_to_front_translation_m=ee_to_front,
    )
    if not math.isfinite(gripper.radius_m) or gripper.radius_m <= 0:
        raise ValueError("gripper radius must be positive and finite")
    if not math.isfinite(gripper.max_point_speed_m_s) or gripper.max_point_speed_m_s <= 0:
        raise ValueError("gripper speed certificate must be positive and finite")

    raw_can = _mapping(root["can_envelope"], "can_envelope")
    if set(raw_can) != {"name", "radius_m", "height_m", "source", "supported"}:
        raise ValueError("can_envelope has unexpected keys")
    if raw_can["name"] != "CAN" or raw_can["supported"] is not True:
        raise ValueError("CAN envelope is unsupported or uses an unexpected identity")
    can = CarriedObjectEnvelope(
        "CAN",
        _finite_scalar(raw_can["radius_m"], "can.radius_m"),
        _finite_scalar(raw_can["height_m"], "can.height_m"),
        _text(raw_can["source"], "can source"),
    )
    if (
        not math.isfinite(can.radius_m)
        or not math.isfinite(can.height_m)
        or can.radius_m <= 0
        or can.height_m <= 0
    ):
        raise ValueError("CAN radius and height must be positive and finite")

    unsupported = root["unsupported"]
    if not isinstance(unsupported, Sequence) or isinstance(unsupported, (str, bytes)):
        raise ValueError("unsupported must be a list of strings")
    unsupported_items = tuple(_text(value, "unsupported item") for value in unsupported)
    if not unsupported_items:
        raise ValueError("explicit unsupported geometry list must not be empty")

    geometry = StaticSceneGeometry(
        boxes=tuple(boxes),
        floors=tuple(floors),
        gripper=gripper,
        carried_objects=(can,),
        supported_object_ids=frozenset({can.name}),
        unsupported=unsupported_items,
        catalog_sha256=sha256_file(source),
        base_pose_position_tolerance_m=position_tolerance,
        base_pose_orientation_tolerance_rad=orientation_tolerance,
    )
    return SceneCollisionCatalog(
        source_path=source,
        status=status,
        hashes=digests,
        T_world_base=T_world_base,
        T_base_world=T_base_world,
        T_base_urdf=T_base_urdf,
        T_world_urdf=T_world_urdf,
        expected_base_pose_world=expected_pose,
        base_pose_tolerance=tolerances,
        geometry=geometry,
        unsupported=unsupported_items,
    )


__all__ = [
    "SCENE_GEOMETRY_SCHEMA_VERSION",
    "SceneCollisionCatalog",
    "axis_angle_matrix",
    "load_scene_collision_catalog",
    "matrix_quaternion",
    "pose_matrix",
    "sha256_file",
]
