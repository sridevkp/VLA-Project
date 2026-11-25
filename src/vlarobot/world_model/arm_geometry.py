"""Conservative swept capsule bounds with explicit geometry coverage and speed certificates."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    """Pinned scene box in the robot base frame.

    ``center_base``/``half_base`` are the conservative axis-aligned bound used by the shipped
    broad phase. When the source box is not axis-aligned in the base frame, ``axes_base`` and
    ``half_extents`` carry the exact oriented box. Diagnostics may use the exact form; the
    shipped check keeps the AABB so its behaviour is unchanged.
    """

    name: str
    center_base: tuple[float, float, float]
    half_base: tuple[float, float, float]
    source: str
    axes_base: tuple[tuple[float, float, float], ...] | None = None
    half_extents: tuple[float, float, float] | None = None

    @property
    def exact(self) -> bool:
        return self.axes_base is not None and self.half_extents is not None


@dataclass(frozen=True)
class FloorHalfSpace:
    """Conservative floor boundary: the free side has non-negative signed distance."""

    name: str
    point_base: tuple[float, float, float]
    normal_base: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class BoxPrimitive:
    """Conservative oriented-box collider derived from one URDF ``<collision>`` solid.

    Boxes exist because capsule bounding is badly lossy for flat gripper plates: the finger
    solid ``0.0076 x 0.05091 x 0.08982`` becomes a capsule of radius ``0.02574 m``, i.e. 6.8x
    its true half-thickness. ``clearance_m`` is the per-primitive clearance floor that the
    declared interaction class allows; it replaces the free-space margin for that primitive
    only and is never allowed to be negative.
    """

    name: str
    body: str
    center: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], ...]
    half_extents: tuple[float, float, float]
    max_point_speed_m_s: float
    clearance_m: float

    def validate(self) -> None:
        axes = np.asarray(self.axes, dtype=float)
        values = np.r_[
            self.center,
            axes.ravel(),
            self.half_extents,
            [self.max_point_speed_m_s, self.clearance_m],
        ]
        if values.shape != (17,) or not np.isfinite(values).all():
            raise ValueError(f"invalid box primitive: {self.name}")
        if not self.name or not self.body:
            raise ValueError("box primitives require a name and a URDF body")
        if not math.isclose(float(np.linalg.det(axes)), 1.0, abs_tol=1e-6):
            raise ValueError(f"box primitive axes must be right-handed orthonormal: {self.name}")
        if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6):
            raise ValueError(f"box primitive axes must be orthonormal: {self.name}")
        if np.any(np.asarray(self.half_extents, dtype=float) <= 0.0):
            raise ValueError(f"box primitive half extents must be positive: {self.name}")
        if self.max_point_speed_m_s < 0.0 or self.clearance_m < 0.0:
            raise ValueError(f"invalid box speed/clearance bound: {self.name}")


def segment_segment_distance(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> float:
    """Exact Euclidean distance between two closed line segments (Ericson, 5.1.9)."""
    p1 = np.asarray(first_start, dtype=float)
    q1 = np.asarray(first_end, dtype=float)
    p2 = np.asarray(second_start, dtype=float)
    q2 = np.asarray(second_end, dtype=float)
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    denom = a * e - b * b
    s = 0.0
    if denom > 1e-15:
        s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
    t = float((b * s + f) / e) if e > 1e-15 else 0.0
    if t < 0.0:
        t = 0.0
        s = float(np.clip(-c / a, 0.0, 1.0)) if a > 1e-15 else 0.0
    elif t > 1.0:
        t = 1.0
        s = float(np.clip((b - c) / a, 0.0, 1.0)) if a > 1e-15 else 0.0
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def segment_segment_distance_batch(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray:
    """Vectorised exact segment/segment distance for ``N`` corresponding pairs.

    Same clamp sequence as :func:`segment_segment_distance`; used only by the offline
    compatibility regression, which needs every recorded configuration.
    """
    p1 = np.asarray(first_start, dtype=float)
    q1 = np.asarray(first_end, dtype=float)
    p2 = np.asarray(second_start, dtype=float)
    q2 = np.asarray(second_end, dtype=float)
    if (
        p1.ndim != 2
        or p1.shape[1:] != (3,)
        or any(array.shape != p1.shape for array in (q1, p2, q2))
    ):
        raise ValueError("segments must be matching (N, 3) arrays")
    if p1.shape[0] == 0:
        return np.zeros(0, dtype=float)
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    usable = denom > 1e-15
    s = np.where(usable, np.clip((b * f - c * e) / np.where(usable, denom, 1.0), 0.0, 1.0), 0.0)
    t = np.where(e > 1e-15, (b * s + f) / np.where(e > 1e-15, e, 1.0), 0.0)
    negative = t < 0.0
    s = np.where(
        negative,
        np.clip(-c / np.where(a > 1e-15, a, 1.0), 0.0, 1.0),
        s,
    )
    t = np.where(negative, 0.0, t)
    positive = t > 1.0
    s = np.where(
        positive,
        np.clip((b - c) / np.where(a > 1e-15, a, 1.0), 0.0, 1.0),
        s,
    )
    t = np.where(positive, 1.0, t)
    return np.linalg.norm(p1 + d1 * s[:, None] - (p2 + d2 * t[:, None]), axis=1)


def box_corners(box: BoxPrimitive) -> np.ndarray:
    """Return the eight world/base-frame corners of an oriented box."""
    axes = np.asarray(box.axes, dtype=float)
    half = np.asarray(box.half_extents, dtype=float)
    local = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=float) * half
    return local @ axes + np.asarray(box.center, dtype=float)


def point_box_distance(point: Sequence[float], box: BoxPrimitive) -> float:
    axes = np.asarray(box.axes, dtype=float)
    half = np.asarray(box.half_extents, dtype=float)
    local = axes @ (np.asarray(point, dtype=float) - np.asarray(box.center, dtype=float))
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def segment_box_distance(start: Sequence[float], end: Sequence[float], box: BoxPrimitive) -> float:
    """Exact distance between a segment and an oriented box via convex ternary search.

    The distance from a linearly moving point to a convex set is a convex function of the
    parameter, so a fixed 64-step ternary search converges to machine precision without any
    randomness or iteration-count dependence on the input.
    """
    low, high = 0.0, 1.0
    p = np.asarray(start, dtype=float)
    d = np.asarray(end, dtype=float) - p
    for _ in range(64):
        third = (high - low) / 3.0
        first = point_box_distance(p + (low + third) * d, box)
        second = point_box_distance(p + (high - third) * d, box)
        if first <= second:
            high -= third
        else:
            low += third
    return point_box_distance(p + 0.5 * (low + high) * d, box)


# Corner order comes from ``itertools.product((-1, 1), repeat=3)``: index bits are (x, y, z).
_BOX_EDGES: tuple[tuple[int, int], ...] = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if bin(first ^ second).count("1") == 1
)


def box_edges(box: BoxPrimitive) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    corners = box_corners(box)
    return tuple((corners[first], corners[second]) for first, second in _BOX_EDGES)


def box_box_distance(first: BoxPrimitive, second: BoxPrimitive) -> float:
    """Exact distance between two oriented boxes (0.0 when they intersect).

    For two convex polytopes the closest features are a vertex/face or an edge/edge pair, so the
    minimum over both boxes' vertices and edges against the other solid is exact. Returns 0.0 for
    any intersecting pair; use :func:`oriented_boxes_intersect` to separate touching from
    penetration.
    """
    best = float("inf")
    for corner in box_corners(first):
        best = min(best, point_box_distance(corner, second))
    for corner in box_corners(second):
        best = min(best, point_box_distance(corner, first))
    for start, end in box_edges(first):
        best = min(best, segment_box_distance(start, end, second))
    for start, end in box_edges(second):
        best = min(best, segment_box_distance(start, end, first))
    return 0.0 if best < 0.0 else float(best)


def oriented_boxes_intersect(first: BoxPrimitive, second: BoxPrimitive) -> bool:
    """Separating-axis test for two oriented boxes (touching counts as intersecting)."""
    center_first = np.asarray(first.center, dtype=float)
    center_second = np.asarray(second.center, dtype=float)
    axes_first = np.asarray(first.axes, dtype=float)
    axes_second = np.asarray(second.axes, dtype=float)
    half_first = np.asarray(first.half_extents, dtype=float)
    half_second = np.asarray(second.half_extents, dtype=float)
    candidates: list[np.ndarray] = [*axes_first, *axes_second]
    for first_axis in axes_first:
        for second_axis in axes_second:
            cross = np.cross(first_axis, second_axis)
            norm = float(np.linalg.norm(cross))
            if norm > 1e-9:
                candidates.append(cross / norm)
    delta = center_second - center_first
    for axis in candidates:
        radius_first = float(np.sum(np.abs(axes_first @ axis) * half_first))
        radius_second = float(np.sum(np.abs(axes_second @ axis) * half_second))
        if abs(float(axis @ delta)) > radius_first + radius_second + 1e-12:
            return False
    return True


def static_box_primitive(box: StaticBox) -> BoxPrimitive:
    """Collider for a pinned scene box (zero speed by construction).

    Uses the exact oriented box when the catalog carries one; otherwise falls back to the
    conservative axis-aligned bound so older/synthetic callers keep working.
    """
    if box.exact:
        axes = tuple(tuple(float(value) for value in row) for row in box.axes_base)
        half = tuple(float(value) for value in box.half_extents)
    else:
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        half = tuple(float(value) for value in box.half_base)
    return BoxPrimitive(
        name=box.name,
        body=box.name,
        center=tuple(float(value) for value in box.center_base),
        axes=axes,
        half_extents=half,
        max_point_speed_m_s=0.0,
        clearance_m=0.0,
    )


def primitive_lower_bound(first: Any, second: Any) -> float:
    """Component-wise AABB lower bound on the separation of two swept primitives."""
    low_first, high_first = primitive_bounds(first)
    low_second, high_second = primitive_bounds(second)
    return float(
        max(
            np.max(low_first - high_second),
            np.max(low_second - high_first),
        )
    )


def primitive_bounds(primitive: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(primitive, BoxPrimitive):
        corners = box_corners(primitive)
        return corners.min(axis=0), corners.max(axis=0)
    start = np.asarray(primitive.start, dtype=float)
    end = np.asarray(primitive.end, dtype=float)
    radius = np.asarray([primitive.radius_m] * 3)
    return np.minimum(start, end) - radius, np.maximum(start, end) + radius


def primitive_penetrates(first: Any, second: Any) -> bool:
    """True when two declared primitives touch or interpenetrate (no tolerance)."""
    if isinstance(first, BoxPrimitive) and isinstance(second, BoxPrimitive):
        return oriented_boxes_intersect(first, second)
    if isinstance(first, BoxPrimitive) or isinstance(second, BoxPrimitive):
        box = first if isinstance(first, BoxPrimitive) else second
        capsule = second if isinstance(first, BoxPrimitive) else first
        return segment_box_distance(capsule.start, capsule.end, box) <= capsule.radius_m
    return (
        segment_segment_distance(first.start, first.end, second.start, second.end)
        <= first.radius_m + second.radius_m
    )


def primitive_clearance(first: Any, second: Any) -> float:
    """Exact minimum surface separation of two primitives (0.0 when touching/overlapping)."""
    if isinstance(first, BoxPrimitive) and isinstance(second, BoxPrimitive):
        return box_box_distance(first, second)
    if isinstance(first, BoxPrimitive) or isinstance(second, BoxPrimitive):
        box = first if isinstance(first, BoxPrimitive) else second
        capsule = second if isinstance(first, BoxPrimitive) else first
        return max(0.0, segment_box_distance(capsule.start, capsule.end, box) - capsule.radius_m)
    return max(
        0.0,
        segment_segment_distance(first.start, first.end, second.start, second.end)
        - first.radius_m
        - second.radius_m,
    )


@dataclass(frozen=True)
class GripperEnvelope:
    """Conservative capsule for gripper finger/body solids in a declared frame.

    ``frame`` identifies the coordinate system of ``start_tool``/``end_tool``. For
    ``front``, ``ee_to_front_translation_m`` recovers the URDF ``front`` origin from the
    runtime ``ee_pose`` (which is the finger-pad tool frame). The legacy
    ``wrist_ft_tool_link`` frame keeps a zero offset and remains available for diagnostics.
    """

    name: str
    start_tool: tuple[float, float, float]
    end_tool: tuple[float, float, float]
    radius_m: float
    max_point_speed_m_s: float
    source: str
    frame: str = "wrist_ft_tool_link"
    ee_to_front_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


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
        numerical_error_bound_m: float = 0.0,
        transform_error_bound_m: float = 0.0,
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
            (numerical_error_bound_m, "numerical error bound"),
            (transform_error_bound_m, "transform error bound"),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.geometry_error_bound_m = float(geometry_error_bound_m)
        self.registration_uncertainty_bound_m = float(registration_uncertainty_bound_m)
        self.numerical_error_bound_m = float(numerical_error_bound_m)
        self.transform_error_bound_m = float(transform_error_bound_m)
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
            if box.axes_base is None and box.half_extents is None:
                continue
            if box.axes_base is None or box.half_extents is None:
                raise ValueError(f"partial oriented static box: {box.name}")
            axes = np.asarray(box.axes_base, dtype=float)
            half = np.asarray(box.half_extents, dtype=float)
            if (
                axes.shape != (3, 3)
                or half.shape != (3,)
                or not np.isfinite(np.r_[axes.ravel(), half]).all()
                or np.any(half <= 0.0)
                or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6)
                or not math.isclose(float(np.linalg.det(axes)), 1.0, abs_tol=1e-6)
            ):
                raise ValueError(f"invalid oriented static box: {box.name}")
            if np.any(np.asarray(box.half_base) + 1e-9 < np.abs(axes.T) @ half):
                raise ValueError(
                    f"axis-aligned bound does not contain oriented static box: {box.name}"
                )
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
            if scene.gripper.frame not in {"front", "wrist_ft_tool_link"}:
                raise ValueError("unsupported gripper envelope frame")
            if not np.isfinite(np.asarray(scene.gripper.ee_to_front_translation_m)).all():
                raise ValueError("invalid gripper ee-to-front translation")
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
        if gripper.frame == "front":
            offset = np.asarray(gripper.ee_to_front_translation_m, dtype=np.float64)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("invalid gripper ee-to-front translation")
            origin = rotate(q, offset) + origin
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
                + self.numerical_error_bound_m
                + self.transform_error_bound_m
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


class ExactSweptArmCheck(SweptArmCheck):
    """Narrowed, exact-distance swept check used only by the Packet-B diagnostic builder.

    This is deliberately **not** the shipped conservative broad phase. Every declared
    separation is computed exactly (segment/segment, segment/box, box/box) and each bound term
    is applied once per pair::

        required = max(clearance_a, clearance_b) + max(speed_a, speed_b) * dt + error_allowance

    ``excluded_self_pairs`` removes only pairs that are demonstrably invalid to check (same
    rigid body, directly jointed links, URDF volumes that intersect over the declared joint
    range, or the sibling gripper fingers). Every other pair stays vetoed.

    ``permitted_contact`` maps one primitive name to the scene object ids that primitive may
    touch during a declared interaction. A permitted interaction still has to be
    non-penetrating, and every other object keeps the full clearance requirement.

    The class never certifies motion. The shipped contract stays ``status: blocked`` and the
    production path keeps :class:`SweptArmCheck`.
    """

    def __init__(
        self,
        geometry: Callable[[WorldState], tuple[Capsule, ...]],
        *,
        box_geometry: Callable[[WorldState], tuple[BoxPrimitive, ...]] | None = None,
        excluded_self_pairs: frozenset[frozenset[str]] = frozenset(),
        permitted_contact: Mapping[str, frozenset[str]] | None = None,
        error_allowance_m: float = 0.0,
        **contract: Any,
    ) -> None:
        super().__init__(geometry, **contract)
        if (
            isinstance(error_allowance_m, bool)
            or not math.isfinite(error_allowance_m)
            or error_allowance_m < 0.0
        ):
            raise ValueError("error allowance must be finite and nonnegative")
        excluded = frozenset(excluded_self_pairs)
        if any(len(pair) != 2 for pair in excluded):
            raise ValueError("excluded self pairs must name exactly two primitives")
        self.box_geometry = box_geometry
        self.excluded_self_pairs = excluded
        self.permitted_contact = {
            str(name): frozenset(objects) for name, objects in (permitted_contact or {}).items()
        }
        self.error_allowance_m = float(error_allowance_m)

    @staticmethod
    def _speed(primitive: Any) -> float:
        return float(getattr(primitive, "max_point_speed_m_s", 0.0))

    def _clearance(self, primitive: Any) -> float:
        return self.margin if isinstance(primitive, Capsule) else primitive.clearance_m

    def _required(self, first: Any, second: Any) -> float:
        motion = max(self._speed(first), self._speed(second)) * self.dt
        clearance = max(self._clearance(first), self._clearance(second))
        return clearance + motion + self.error_allowance_m

    def _pair_ok(self, first: Any, second: Any, required: float) -> bool:
        if primitive_lower_bound(first, second) >= required:
            return True
        if required <= 0.0:
            return not primitive_penetrates(first, second)
        return primitive_clearance(first, second) >= required

    def _base_pose_ok(self, state: WorldState) -> bool | None:
        if self.expected_base_pose_world is None:
            return True
        expected = self._pose_array(self.expected_base_pose_world)
        observed = self._pose_array(state.robot.base_pose_world)
        if expected is None or observed is None:
            return None
        position_tolerance = (
            self.scene.base_pose_position_tolerance_m if self.scene is not None else 0.0
        )
        orientation_tolerance = (
            self.scene.base_pose_orientation_tolerance_rad if self.scene is not None else 0.0
        )
        position_error = float(np.linalg.norm(observed[:3] - expected[:3]))
        dot = float(np.clip(np.abs(observed[3:] @ expected[3:]), -1.0, 1.0))
        return (
            position_error <= position_tolerance and 2.0 * math.acos(dot) <= orientation_tolerance
        )

    @staticmethod
    def _valid_capsules(sample: Sequence[Capsule]) -> bool:
        names = [capsule.name for capsule in sample]
        if len(set(names)) != len(names):
            return False
        for capsule in sample:
            points = np.asarray((capsule.start, capsule.end), dtype=float)
            if points.shape != (2, 3) or not np.isfinite(points).all():
                return False
            if not math.isfinite(capsule.radius_m) or capsule.radius_m <= 0.0:
                return False
            if not math.isfinite(capsule.max_point_speed_m_s) or capsule.max_point_speed_m_s <= 0.0:
                return False
        return True

    @staticmethod
    def _valid_boxes(sample: Sequence[BoxPrimitive]) -> bool:
        names = [box.name for box in sample]
        if len(set(names)) != len(names):
            return False
        for box in sample:
            try:
                box.validate()
            except ValueError:
                return False
            if box.max_point_speed_m_s <= 0.0:
                return False
        return True

    def _motion_ok(
        self,
        capsules: Sequence[Sequence[Capsule]],
        boxes: Sequence[Sequence[BoxPrimitive]],
    ) -> bool:
        first_capsules = {capsule.name: capsule for capsule in capsules[0]}
        second_capsules = {capsule.name: capsule for capsule in capsules[1]}
        if set(first_capsules) != set(second_capsules):
            return False
        for name, first in first_capsules.items():
            second = second_capsules[name]
            if (
                first.radius_m != second.radius_m
                or first.max_point_speed_m_s != second.max_point_speed_m_s
            ):
                return False
            limit = first.max_point_speed_m_s * self.dt
            for start, end in ((first.start, second.start), (first.end, second.end)):
                if float(np.linalg.norm(np.asarray(start) - np.asarray(end))) > limit:
                    return False
        first_boxes = {box.name: box for box in boxes[0]}
        second_boxes = {box.name: box for box in boxes[1]}
        if set(first_boxes) != set(second_boxes):
            return False
        for name, first in first_boxes.items():
            second = second_boxes[name]
            if (
                first.half_extents != second.half_extents
                or first.clearance_m != second.clearance_m
                or first.max_point_speed_m_s != second.max_point_speed_m_s
            ):
                return False
            limit = first.max_point_speed_m_s * self.dt
            step = float(np.max(np.linalg.norm(box_corners(first) - box_corners(second), axis=1)))
            if step > limit:
                return False
        return True

    def _self_separation_ok(
        self,
        capsules: Sequence[Capsule],
        boxes: Sequence[BoxPrimitive],
    ) -> bool:
        for index, first in enumerate(capsules):
            for second in capsules[index + 1 :]:
                if frozenset((first.name, second.name)) in self.excluded_self_pairs:
                    continue
                if not self._pair_ok(first, second, self._required(first, second)):
                    return False
            for box in boxes:
                if frozenset((first.name, box.name)) in self.excluded_self_pairs:
                    continue
                if not self._pair_ok(first, box, self._required(first, box)):
                    return False
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                if frozenset((first.name, second.name)) in self.excluded_self_pairs:
                    continue
                if not self._pair_ok(first, second, self._required(first, second)):
                    return False
        return True

    @staticmethod
    def _floor_ok(primitive: Any, floor: FloorHalfSpace, required: float) -> bool:
        point = np.asarray(floor.point_base, dtype=float)
        normal = np.asarray(floor.normal_base, dtype=float)
        if isinstance(primitive, BoxPrimitive):
            signed = float(np.min((box_corners(primitive) - point) @ normal))
        else:
            signed = (
                float(
                    np.min(
                        (np.asarray((primitive.start, primitive.end), dtype=float) - point) @ normal
                    )
                )
                - primitive.radius_m
            )
        return signed >= required

    def _static_separation_ok(
        self,
        capsules: Sequence[Capsule],
        boxes: Sequence[BoxPrimitive],
    ) -> bool:
        scene = self.scene
        if scene is None:
            return True
        statics = [static_box_primitive(box) for box in scene.boxes]
        for primitive in (*capsules, *boxes):
            for static in statics:
                if not self._pair_ok(primitive, static, self._required(primitive, static)):
                    return False
            floor_required = (
                self._clearance(primitive)
                + self._speed(primitive) * self.dt
                + self.error_allowance_m
            )
            for floor in scene.floors:
                if not self._floor_ok(primitive, floor, floor_required):
                    return False
        return True

    def _object_separation_ok(
        self,
        before: WorldState,
        capsules: Sequence[Capsule],
        boxes: Sequence[BoxPrimitive],
    ) -> bool:
        ids = [obj.object_id for obj in before.scene.objects]
        if len(set(ids)) != len(ids):
            return False
        scene = self.scene
        if scene is not None and not set(ids).issubset(scene.supported_object_ids):
            return False
        for obj in before.scene.objects:
            if obj.object_id in self.contacts:
                continue
            center = obj.pose.position.as_array()
            quaternion = np.asarray(obj.pose.orientation, dtype=float)
            dimensions = obj.dimensions.as_array()
            if (
                not np.isfinite(np.r_[center, quaternion, dimensions]).all()
                or np.any(dimensions <= 0.0)
                or not math.isclose(float(quaternion @ quaternion), 1.0, abs_tol=1e-9)
            ):
                return False
            envelope = _object_box(obj)
            for primitive in (*capsules, *boxes):
                if obj.object_id in self.permitted_contact.get(primitive.name, frozenset()):
                    if primitive_penetrates(primitive, envelope):
                        return False
                    continue
                if not self._pair_ok(primitive, envelope, self._required(primitive, envelope)):
                    return False
        return True

    def __call__(self, before: WorldState, after: WorldState) -> bool | None:
        if self.scene_complete is not True or not self.required_objects.issubset(
            {obj.object_id for obj in before.scene.objects}
        ):
            return None
        if before.robot.base_pose != after.robot.base_pose:
            return None
        if self.box_geometry is None:
            return None
        if self._base_pose_ok(before) is not True or self._base_pose_ok(after) is not True:
            return None
        try:
            capsules = [self.geometry(state) for state in (before, after)]
            boxes = [self.box_geometry(state) for state in (before, after)]
        except Exception:  # noqa: BLE001 - any geometry/transform failure is fail-closed
            return None
        if not all(self._valid_capsules(sample) for sample in capsules):
            return None
        if not all(self._valid_boxes(sample) for sample in boxes):
            return None
        if {capsule.name for capsule in capsules[0]} != set(self.required):
            return None
        if {box.name for box in boxes[0]} != {box.name for box in boxes[1]}:
            return None
        if {box.name for box in boxes[0]} & set(self.required):
            return None
        if not self._motion_ok(capsules, boxes):
            return None
        if [obj.object_id for obj in after.scene.objects] != [
            obj.object_id for obj in before.scene.objects
        ]:
            return None
        try:
            for first, second in zip(before.scene.objects, after.scene.objects, strict=True):
                if first.object_id not in self.contacts and (
                    first.pose != second.pose or first.dimensions != second.dimensions
                ):
                    return None
            if not self._self_separation_ok(capsules[0], boxes[0]):
                return False
            if not self._static_separation_ok(capsules[0], boxes[0]):
                return False
            if not self._object_separation_ok(before, capsules[0], boxes[0]):
                return False
        except Exception:  # noqa: BLE001 - any distance failure is fail-closed
            return None
        return True


def _object_box(obj: Any) -> BoxPrimitive:
    """Envelope one scene object in a box aligned with its declared pose."""
    quaternion = np.asarray(obj.pose.orientation, dtype=float)
    rotation = np.column_stack([rotate(quaternion, axis) for axis in np.eye(3)])
    return BoxPrimitive(
        name=f"object:{obj.object_id}",
        body=obj.object_id,
        center=tuple(float(value) for value in obj.pose.position.as_array()),
        axes=tuple(tuple(float(value) for value in row) for row in rotation.T),
        half_extents=tuple(float(value) / 2.0 for value in obj.dimensions.as_array()),
        max_point_speed_m_s=0.0,
        clearance_m=0.0,
    )
