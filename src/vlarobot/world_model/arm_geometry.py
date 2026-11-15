"""Conservative swept capsule bounds with explicit geometry coverage and speed certificates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from vlarobot.core.types import WorldState
from vlarobot.world_model.rollout import rotate


@dataclass(frozen=True)
class Capsule:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius_m: float
    # Certified upper bound on any centreline point speed, from link lengths/joint limits.
    max_point_speed_m_s: float


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
        dt_s: float,
        margin_m: float,
        adjacent_pairs: frozenset[frozenset[str]],
        permitted_contact_objects: frozenset[str] = frozenset(),
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
        self.dt, self.margin = dt_s, margin_m
        self.adjacent, self.contacts = adjacent_pairs, permitted_contact_objects

    @staticmethod
    def overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
        return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))

    def __call__(self, before: WorldState, after: WorldState) -> bool | None:
        if before.robot.base_pose != after.robot.base_pose:
            return None
        sampled = [self.geometry(before), self.geometry(after)]
        maps: list[Mapping[str, Capsule]] = [{c.name: c for c in sample} for sample in sampled]
        if any(
            set(m) != self.required or len(m) != len(sample)
            for m, sample in zip(maps, sampled, strict=True)
        ):
            return None
        swept = {}
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
            padding = a.radius_m + self.margin + a.max_point_speed_m_s * self.dt / 2
            swept[name] = (points.min(axis=0) - padding, points.max(axis=0) + padding)
        for i, name in enumerate(sorted(swept)):
            for other in sorted(swept)[i + 1 :]:
                if frozenset((name, other)) not in self.adjacent and self.overlap(
                    swept[name], swept[other]
                ):
                    return False
        ids = [o.object_id for o in before.scene.objects]
        if len(set(ids)) != len(ids):
            return None
        for obj in before.scene.objects:
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
