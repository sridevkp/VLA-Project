"""Conservative Cartesian geometry; no simulator branching or invented success probabilities."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vlarobot.core.types import (
    ActionChunk,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    PredictionError,
    RobotState,
    SceneState,
    WorldState,
)


class ObjectWorldModel:
    """Protocol-compatible snapshots/correction with no mutable alias to caller state."""

    def __init__(self) -> None:
        self._state: WorldState | None = None

    def update(self, scene: SceneState, robot: RobotState) -> None:
        self._state = copy.deepcopy(WorldState(scene, robot))

    def snapshot(self) -> WorldState:
        if self._state is None:
            raise ValueError("world state has not been initialized")
        return copy.deepcopy(self._state)

    def correct(self, predicted: WorldState, observed: WorldState) -> PredictionError:
        expected = {o.object_id: o for o in predicted.scene.objects}
        actual = {o.object_id: o for o in observed.scene.objects}
        if (
            not expected
            or len(expected) != len(predicted.scene.objects)
            or len(actual) != len(observed.scene.objects)
            or set(expected) != set(actual)
        ):
            raise ValueError("correction requires complete matched object identities")
        positions, angles = [], []
        for name, obj in expected.items():
            truth = actual[name]
            p = obj.pose.position.as_array() - truth.pose.position.as_array()
            q, r = np.asarray(obj.pose.orientation), np.asarray(truth.pose.orientation)
            if (
                not np.all(np.isfinite(np.r_[p, q, r]))
                or np.linalg.norm(q) == 0
                or np.linalg.norm(r) == 0
            ):
                raise ValueError("nonfinite or invalid correction state")
            positions.append(float(p @ p))
            cosine = abs(float(q @ r / (np.linalg.norm(q) * np.linalg.norm(r))))
            angles.append(float((2 * np.arccos(np.clip(cosine, 0, 1))) ** 2))
        self._state = copy.deepcopy(observed)
        return PredictionError(float(np.sqrt(np.mean(positions))), float(np.sqrt(np.mean(angles))))


def segment_box_clearance(
    start: np.ndarray, end: np.ndarray, lower: np.ndarray, upper: np.ndarray, radius_m: float
) -> float:
    """Exact minimum segment/AABB distance minus sphere radius, including thin obstacles."""
    vectors = [np.asarray(v, dtype=float) for v in (start, end, lower, upper)]
    if (
        any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in vectors)
        or not np.isfinite(radius_m)
        or radius_m < 0
    ):
        raise ValueError("finite 3D geometry and nonnegative radius required")
    start, end, lower, upper = vectors
    if np.any(lower > upper):
        raise ValueError("box bounds are reversed")
    delta = end - start
    breaks = {0.0, 1.0}
    for axis in range(3):
        if delta[axis] != 0:
            for face in [lower[axis], upper[axis]]:
                t = float((face - start[axis]) / delta[axis])
                if 0 < t < 1:
                    breaks.add(t)
    candidates = sorted(breaks)
    minimum = float("inf")
    for left, right in zip(candidates[:-1], candidates[1:], strict=True):
        midpoint = start + delta * ((left + right) / 2)
        outside = (midpoint < lower) | (midpoint > upper)
        boundary = np.where(midpoint < lower, lower, upper)
        a, b = (start - boundary)[outside], delta[outside]
        t = float(np.clip(-a @ b / (b @ b), left, right)) if b @ b > 0 else left
        for value in [left, t, right]:
            point = start + delta * value
            distance = np.maximum(np.maximum(lower - point, point - upper), 0)
            minimum = min(minimum, float(np.linalg.norm(distance)))
    return minimum - radius_m


@dataclass(frozen=True)
class GeometryPrediction:
    min_clearance_m: float | None
    hard_reasons: tuple[str, ...]
    duration_s: float
    tool_path_base_m: tuple[tuple[float, float, float], ...]
    geometry_complete: bool
    ik_verified: bool
    calibrated_probabilities: bool = False


class CartesianPredictor:
    """Swept tool-sphere clearance; full-arm/self/grasp physics remain unmodeled."""

    def __init__(
        self,
        *,
        tool_radius_m: float,
        clearance_margin_m: float,
        workspace_min_m: tuple[float, float, float],
        workspace_max_m: tuple[float, float, float],
        ik_check: Callable[[WorldState, ActionChunk], bool] | None = None,
    ) -> None:
        values = np.r_[tool_radius_m, clearance_margin_m, workspace_min_m, workspace_max_m]
        if (
            not np.all(np.isfinite(values))
            or tool_radius_m <= 0
            or clearance_margin_m <= 0
            or np.any(np.asarray(workspace_min_m) >= workspace_max_m)
        ):
            raise ValueError("finite positive geometry and workspace bounds required")
        self.radius, self.margin = tool_radius_m, clearance_margin_m
        self.lower, self.upper = np.asarray(workspace_min_m), np.asarray(workspace_max_m)
        self.ik_check = ik_check

    def predict(
        self, state: WorldState, chunk: ActionChunk, *, scene_complete: bool
    ) -> GeometryPrediction:
        reasons = []
        if not np.isfinite(chunk.dt) or chunk.dt <= 0 or not chunk.actions:
            raise ValueError("prediction requires positive lockstep duration and actions")
        point = state.robot.ee_pose.position.as_array()
        if not np.all(np.isfinite(point)):
            raise ValueError("nonfinite initial tool pose")
        if np.any(point < self.lower) or np.any(point > self.upper):
            reasons.append("workspace")
        path = [point.copy()]
        clearance = None
        boxes = []
        for obj in state.scene.objects:
            q = np.asarray(obj.pose.orientation, dtype=float)
            center, dimensions = obj.pose.position.as_array(), obj.dimensions.as_array()
            if (
                not np.all(np.isfinite(np.r_[q, center, dimensions]))
                or np.any(dimensions <= 0)
                or not np.isclose(q @ q, 1, atol=1e-3)
            ):
                raise ValueError("object geometry is unavailable or malformed")
            x, y, z, w = q
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            half = np.abs(rotation) @ (dimensions / 2)
            boxes.append((center - half, center + half))
        for action in chunk.actions:
            if isinstance(action, EEDelta):
                destination = point + action.dpos.as_array()
            elif isinstance(action, EEPoseTarget):
                destination = action.pose.position.as_array()
            elif isinstance(action, GripperCmd):
                destination = point.copy()
            else:
                reasons.append("unsupported-action-geometry")
                break
            if not np.all(np.isfinite(destination)):
                reasons.append("nonfinite-tool-path")
                break
            if np.any(destination < self.lower) or np.any(destination > self.upper):
                reasons.append("workspace")
            for lower, upper in boxes:
                value = segment_box_clearance(point, destination, lower, upper, self.radius)
                clearance = value if clearance is None else min(value, clearance)
            point = destination
            path.append(point.copy())
        if not scene_complete:
            reasons.append("incomplete-scene")
        if clearance is not None and clearance < self.margin:
            reasons.append("tool-clearance")
        ik_verified = self.ik_check is not None and bool(self.ik_check(state, chunk))
        if not ik_verified:
            reasons.append("unverified-ik-or-arm-sweep")
        return GeometryPrediction(
            clearance,
            tuple(dict.fromkeys(reasons)),
            len(chunk.actions) * chunk.dt,
            tuple((float(p[0]), float(p[1]), float(p[2])) for p in path),
            scene_complete,
            ik_verified,
        )
