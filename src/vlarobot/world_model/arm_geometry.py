"""Conservative swept capsule bounds with explicit geometry coverage and speed certificates."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from vlarobot.core.types import Pose3D, WorldState
from vlarobot.world_model.rollout import rotate


@dataclass(frozen=True)
class Capsule:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius_m: float
    # Certified upper bound on any centreline point speed, from link lengths/joint limits.
    max_point_speed_m_s: float


@dataclass(frozen=True)
class StaticBox:
    """Conservative axis-aligned box in the robot base frame."""

    name: str
    center_base: tuple[float, float, float]
    half_base: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class FloorHalfSpace:
    """Conservative floor boundary: the free side has non-negative signed distance."""

    name: str
    point_base: tuple[float, float, float]
    normal_base: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class GripperEnvelope:
    """Tool-frame capsule covering the finger/body boundingObject solids."""

    name: str
    start_tool: tuple[float, float, float]
    end_tool: tuple[float, float, float]
    radius_m: float
    max_point_speed_m_s: float
    source: str


@dataclass(frozen=True)
class CarriedObjectEnvelope:
    """Cylinder envelope of the carried CAN from its pinned PROTO source."""

    name: str
    radius_m: float
    height_m: float
    source: str


@dataclass(frozen=True)
class StaticSceneGeometry:
    """Supported static geometry plus explicitly unsupported coverage gaps."""

    boxes: tuple[StaticBox, ...] = ()
    floors: tuple[FloorHalfSpace, ...] = ()
    gripper: GripperEnvelope | None = None
    carried_objects: tuple[CarriedObjectEnvelope, ...] = ()
    supported_object_ids: frozenset[str] = frozenset()
    unsupported: tuple[str, ...] = ()
    catalog_sha256: str | None = None
    base_pose_position_tolerance_m: float = 0.0
    base_pose_orientation_tolerance_rad: float = 0.0


class SweptArmCheck:
    """Fail closed on missing geometry; broad bounding boxes may conservatively reject.

    Bounds cover each entire control interval if the supplied point-speed certificates
    hold. This is not triangle-mesh contact simulation or a calibrated probability.
    """

    def __init__(
        self,
        geometry: Callable[[WorldState], tuple[Capsule, ...]],
        *,
        required_links: tuple[str, ...],
        required_objects: frozenset[str],
        scene_complete: bool,
        dt_s: float,
        margin_m: float,
        adjacent_pairs: frozenset[frozenset[str]],
        permitted_contact_objects: frozenset[str] = frozenset(),
        scene: StaticSceneGeometry | None = None,
        expected_base_pose_world: Pose3D | None = None,
        geometry_error_bound_m: float = 0.0,
        registration_uncertainty_bound_m: float = 0.0,
    ) -> None:
        if (
            not required_links
            or len(set(required_links)) != len(required_links)
            or not math.isfinite(dt_s)
            or dt_s <= 0
            or not math.isfinite(margin_m)
            or margin_m <= 0
        ):
            raise ValueError("explicit complete geometry contract, dt and positive margin required")
        self.geometry, self.required = geometry, frozenset(required_links)
        self.required_objects, self.scene_complete = required_objects, scene_complete
        self.dt, self.margin = dt_s, margin_m
        self.adjacent, self.contacts = adjacent_pairs, permitted_contact_objects
        self.scene = scene
        self.expected_base_pose_world = expected_base_pose_world
        for value, name in (
            (geometry_error_bound_m, "geometry error bound"),
            (registration_uncertainty_bound_m, "registration uncertainty bound"),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.geometry_error_bound_m = float(geometry_error_bound_m)
        self.registration_uncertainty_bound_m = float(registration_uncertainty_bound_m)
        if scene is not None:
            self._validate_scene(scene)

    @staticmethod
    def _validate_scene(scene: StaticSceneGeometry) -> None:
        if any(not item.name or not item.source for item in scene.boxes):
            raise ValueError("static boxes require names and provenance")
        if len({item.name for item in scene.boxes}) != len(scene.boxes):
            raise ValueError("static box names must be unique")
        for box in scene.boxes:
            values = np.r_[box.center_base, box.half_base]
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values[3:] <= 0):
                raise ValueError(f"invalid static box geometry: {box.name}")
        if len({item.name for item in scene.floors}) != len(scene.floors):
            raise ValueError("floor names must be unique")
        for floor in scene.floors:
            values = np.r_[floor.point_base, floor.normal_base]
            normal = np.asarray(floor.normal_base, dtype=float)
            if (
                values.shape != (6,)
                or not np.isfinite(values).all()
                or not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-8)
            ):
                raise ValueError(f"invalid floor half-space: {floor.name}")
        if scene.gripper is not None:
            values = np.r_[
                scene.gripper.start_tool,
                scene.gripper.end_tool,
                scene.gripper.radius_m,
                scene.gripper.max_point_speed_m_s,
            ]
            if not np.isfinite(values).all() or scene.gripper.radius_m <= 0:
                raise ValueError("invalid gripper envelope")
            if scene.gripper.max_point_speed_m_s <= 0:
                raise ValueError("gripper speed certificate must be positive")
        if len({item.name for item in scene.carried_objects}) != len(scene.carried_objects):
            raise ValueError("carried-object names must be unique")
        for item in scene.carried_objects:
            if (
                not math.isfinite(item.radius_m)
                or not math.isfinite(item.height_m)
                or item.radius_m <= 0
                or item.height_m <= 0
            ):
                raise ValueError(f"invalid carried-object envelope: {item.name}")
        for value in (
            scene.base_pose_position_tolerance_m,
            scene.base_pose_orientation_tolerance_rad,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("base-pose tolerances must be finite and nonnegative")

    @staticmethod
    def overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))

    @staticmethod
    def _pose_array(pose: Pose3D | None) -> np.ndarray | None:
        if pose is None:
            return None
        values = np.r_[pose.position.as_array(), np.asarray(pose.orientation, dtype=float)]
        return values if values.shape == (7,) and np.isfinite(values).all() else None

    def _gripper_capsules(self, state: WorldState) -> tuple[Capsule, ...]:
        scene = self.scene
        if scene is None or scene.gripper is None:
            return ()
        gripper = scene.gripper
        pose = state.robot.ee_pose
        values = self._pose_array(pose)
        q = np.asarray(pose.orientation, dtype=float)
        if values is None or not math.isclose(float(q @ q), 1.0, abs_tol=1e-8):
            raise ValueError("invalid tool transform for gripper envelope")
        origin = pose.position.as_array()
        points: list[tuple[float, float, float]] = []
        for point in (gripper.start_tool, gripper.end_tool):
            vector = np.asarray(point, dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("invalid tool-frame gripper point")
            world = rotate(q, vector) + origin
            if world.shape != (3,) or not np.isfinite(world).all():
                raise ValueError("nonfinite transformed gripper envelope")
            xyz = tuple(float(component) for component in world)
            points.append((xyz[0], xyz[1], xyz[2]))
        return (
            Capsule(
                gripper.name,
                points[0],
                points[1],
                gripper.radius_m,
                gripper.max_point_speed_m_s,
            ),
        )

    def __call__(self, before: WorldState, after: WorldState) -> bool | None:
        if self.scene_complete is not True or not self.required_objects.issubset(
            {o.object_id for o in before.scene.objects}
        ):
            return None
        if before.robot.base_pose != after.robot.base_pose:
            return None
        if self.expected_base_pose_world is not None:
            expected = self._pose_array(self.expected_base_pose_world)
            observed_before = self._pose_array(before.robot.base_pose_world)
            observed_after = self._pose_array(after.robot.base_pose_world)
            if expected is None or observed_before is None or observed_after is None:
                return None
            position_tolerance = (
                self.scene.base_pose_position_tolerance_m if self.scene is not None else 0.0
            )
            orientation_tolerance = (
                self.scene.base_pose_orientation_tolerance_rad if self.scene is not None else 0.0
            )
            for observed in (observed_before, observed_after):
                position_error = float(np.linalg.norm(observed[:3] - expected[:3]))
                dot = float(np.clip(np.abs(observed[3:] @ expected[3:]), -1.0, 1.0))
                orientation_error = 2.0 * math.acos(dot)
                if position_error > position_tolerance or orientation_error > orientation_tolerance:
                    return None
        try:
            sampled = [self.geometry(before), self.geometry(after)]
            if self.scene is not None and self.scene.gripper is not None:
                if self.scene.gripper.name not in self.required:
                    return None
                sampled = [
                    (*sample, *self._gripper_capsules(state))
                    for sample, state in zip(sampled, (before, after), strict=True)
                ]
        except Exception:  # noqa: BLE001 - any geometry/transform failure is fail-closed
            return None
        maps: list[Mapping[str, Capsule]] = [{c.name: c for c in sample} for sample in sampled]
        if any(
            set(m) != self.required or len(m) != len(sample)
            for m, sample in zip(maps, sampled, strict=True)
        ):
            return None
        swept: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self.required:
            a, b = maps[0][name], maps[1][name]
            points = np.asarray((a.start, a.end, b.start, b.end), dtype=float)
            if (
                points.shape != (4, 3)
                or not np.isfinite(points).all()
                or any(
                    not math.isfinite(v) or v <= 0
                    for v in (a.radius_m, b.radius_m, a.max_point_speed_m_s, b.max_point_speed_m_s)
                )
            ):
                return None
            if a.radius_m != b.radius_m or a.max_point_speed_m_s != b.max_point_speed_m_s:
                return None
            if any(
                np.linalg.norm(np.asarray(x) - y) > a.max_point_speed_m_s * self.dt
                for x, y in ((a.start, b.start), (a.end, b.end))
            ):
                return None
            # A point-speed bound alone does not justify half-interval padding: an uncertain
            # path may move up to speed * dt from either endpoint. Add explicit support-error
            # bounds instead of assuming straight, constant-speed motion.
            padding = (
                a.radius_m
                + self.margin
                + a.max_point_speed_m_s * self.dt
                + self.geometry_error_bound_m
                + self.registration_uncertainty_bound_m
            )
            swept[name] = (points.min(axis=0) - padding, points.max(axis=0) + padding)
        for i, name in enumerate(sorted(swept)):
            for other in sorted(swept)[i + 1 :]:
                if frozenset((name, other)) not in self.adjacent and self.overlap(
                    swept[name], swept[other]
                ):
                    return False

        scene = self.scene
        if scene is not None:
            if not self.required_objects.issubset(scene.supported_object_ids):
                return None
            for box in scene.boxes:
                center = np.asarray(box.center_base, dtype=float)
                half = np.asarray(box.half_base, dtype=float)
                box_bounds = (center - half, center + half)
                if any(self.overlap(bound, box_bounds) for bound in swept.values()):
                    return False
            for floor in scene.floors:
                point = np.asarray(floor.point_base, dtype=float)
                normal = np.asarray(floor.normal_base, dtype=float)
                for bound in swept.values():
                    corners = np.asarray(
                        list(itertools.product(*zip(bound[0], bound[1], strict=False)))
                    )
                    if float(np.min((corners - point) @ normal)) <= 0.0:
                        return False

        ids = [o.object_id for o in before.scene.objects]
        if len(set(ids)) != len(ids):
            return None
        if scene is not None and not set(ids).issubset(scene.supported_object_ids):
            return None
        for obj in before.scene.objects:
            envelope = (
                next((item for item in scene.carried_objects if item.name == obj.object_id), None)
                if scene is not None
                else None
            )
            if envelope is not None:
                expected_dimensions = sorted(
                    (2.0 * envelope.radius_m, 2.0 * envelope.radius_m, envelope.height_m)
                )
                if not np.allclose(
                    sorted(obj.dimensions.as_array()), expected_dimensions, atol=1e-6, rtol=0
                ):
                    return None
            if obj.object_id in self.contacts:
                continue
            center = obj.pose.position.as_array()
            q = np.asarray(obj.pose.orientation)
            dimensions = obj.dimensions.as_array()
            if (
                not np.isfinite(np.r_[center, q, dimensions]).all()
                or np.any(dimensions <= 0)
                or not np.isclose(q @ q, 1)
            ):
                return None
            rotation = np.column_stack([rotate(q, axis) for axis in np.eye(3)])
            half = np.abs(rotation) @ (dimensions / 2)
            box = (center - half, center + half)
            if any(self.overlap(bound, box) for bound in swept.values()):
                return False
        # Moving obstacles require their own certified bounds; never silently treat as static.
        if [o.object_id for o in after.scene.objects] != ids:
            return None
        for a, b in zip(before.scene.objects, after.scene.objects, strict=True):
            if a.object_id not in self.contacts and (
                a.pose != b.pose or a.dimensions != b.dimensions
            ):
                return None
        return True
