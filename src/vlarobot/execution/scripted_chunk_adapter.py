"""Reviewed scripted ActionChunk proposer for the production WorldMind composition.

This is a new, explicit adapter. It is **not** the frozen ``ScriptedCanPolicy`` baseline and
does not replace it. It cannot discover geometry, perform a new grasp, simulate contact, or
recover from a failed IK pose: the caller supplies a reviewed waypoint list, a reviewed pose
solver and reviewed safety limits. Every emitted command is validated before it enters a
candidate chunk; the live ``SafetyMonitor`` still vetoes again through ``execute_first``.

Jitter is deterministic from the draw seed and bounded by rejection sampling. Out-of-bound
draws are discarded and redrawn; they are never clipped or silently accepted. The same rule
applies to reviewed absolute joint limits: an endpoint outside those limits is rejected, and a
jitter draw that would cross them is redrawn. Every emitted joint target is still validated by
the injected SafetyMonitor before it enters a candidate chunk.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from vlarobot.core.types import ActionChunk, GripperCmd, JointTarget, Pose3D, TaskSpec, WorldState
from vlarobot.execution.safety_monitor import SafetyMonitor


class PoseSolution(Protocol):
    joints: dict[str, float]
    position_error_m: float
    orientation_error_rad: float


class PoseSolver(Protocol):
    def solve(self, pose: Pose3D, state: WorldState, *, seed: int = 0) -> PoseSolution: ...


@dataclass(frozen=True)
class ScriptedWaypoint:
    """One reviewed station in a scripted chunk.

    Exactly one of ``pose``, ``gripper_opening`` and ``joints`` must be set. ``hold_steps`` is
    the explicit number of control ticks for which that station is repeated. Per-station IK
    tolerances mirror the reviewed baseline's state-specific tolerances; when absent, the
    adapter-level tolerance is used. This station type does not implement settling, contact
    detection, orientation-only IK or recovery; those remain physical-validation items.
    """

    label: str
    hold_steps: int
    pose: Pose3D | None = None
    gripper_opening: float | None = None
    joints: Mapping[str, float] | None = None
    position_tolerance_m: float | None = None
    orientation_tolerance_rad: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("waypoint label must be a non-empty string")
        if isinstance(self.hold_steps, bool) or self.hold_steps < 1:
            raise ValueError("waypoint hold_steps must be a positive integer")
        if sum(item is not None for item in (self.pose, self.gripper_opening, self.joints)) != 1:
            raise ValueError("exactly one of pose, gripper_opening and joints is required")
        for name, value in (
            ("position_tolerance_m", self.position_tolerance_m),
            ("orientation_tolerance_rad", self.orientation_tolerance_rad),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"waypoint {name} must be finite and positive")
            if self.pose is None:
                raise ValueError(f"waypoint {name} is only valid for a pose station")
        if self.joints is not None:
            if not isinstance(self.joints, Mapping) or not self.joints:
                raise ValueError("joint waypoint requires a nonempty joint mapping")
            if any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for name, value in self.joints.items()
            ):
                raise ValueError("joint waypoint requires finite named joint targets")


@dataclass(frozen=True)
class JointJitter:
    """Explicit bounded joint jitter; zero values are deterministic and valid."""

    std_rad: float
    limit_rad: float
    max_attempts: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0
            for value in (self.std_rad, self.limit_rad)
        ):
            raise ValueError("jitter std and limit must be finite nonnegative radians")
        if isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise ValueError("jitter max_attempts must be a positive integer")


Draw = Callable[[TaskSpec, WorldState, object, int], ActionChunk]


class ScriptedChunkAdapter:
    """Draw a seeded H-step scripted chunk from reviewed waypoints and an injected solver."""

    def __init__(
        self,
        waypoints: Sequence[ScriptedWaypoint],
        *,
        solver: PoseSolver,
        monitor: SafetyMonitor,
        horizon_steps: int,
        control_dt_s: float,
        jitter: JointJitter,
        max_joint_step_rad: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> None:
        if not waypoints:
            raise ValueError("reviewed scripted waypoints are required")
        total = sum(int(item.hold_steps) for item in waypoints)
        if total != horizon_steps:
            raise ValueError(f"waypoint hold_steps sum {total} != horizon {horizon_steps}")
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                control_dt_s,
                max_joint_step_rad,
                position_tolerance_m,
                orientation_tolerance_rad,
            )
        ):
            raise ValueError(
                "explicit positive control tick, joint step and IK tolerances required"
            )
        self.waypoints = tuple(waypoints)
        self.solver = solver
        self.monitor = monitor
        self.horizon_steps = int(horizon_steps)
        self.control_dt_s = float(control_dt_s)
        self.jitter = jitter
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)

    def draw(self) -> Draw:
        """Return the callable consumed by ``SeededCandidateGenerator``."""

        def propose(task: TaskSpec, state: WorldState, memory: object, seed: int) -> ActionChunk:
            del task, memory
            if not isinstance(state, WorldState):
                raise TypeError("scripted adapter requires an oracle-initialized WorldState")
            if state.robot.base_pose_world is None:
                raise ValueError("scripted adapter requires an explicit world-to-base transform")
            rng = np.random.default_rng(int(seed))
            actions: list[JointTarget | GripperCmd] = []
            current_joints = {
                str(name): float(value) for name, value in state.robot.joint_pos.items()
            }
            for waypoint in self.waypoints:
                if waypoint.pose is not None:
                    simulated = replace(state, robot=replace(state.robot, joint_pos=current_joints))
                    solution = self.solver.solve(waypoint.pose, simulated, seed=int(seed))
                    residuals = (
                        float(solution.position_error_m),
                        float(solution.orientation_error_rad),
                    )
                    position_limit = (
                        self.position_tolerance_m
                        if waypoint.position_tolerance_m is None
                        else float(waypoint.position_tolerance_m)
                    )
                    orientation_limit = (
                        self.orientation_tolerance_rad
                        if waypoint.orientation_tolerance_rad is None
                        else float(waypoint.orientation_tolerance_rad)
                    )
                    if not all(math.isfinite(value) for value in residuals) or (
                        residuals[0] > position_limit or residuals[1] > orientation_limit
                    ):
                        raise ValueError("scripted pose residual outside reviewed tolerance")
                    endpoint = self._jittered_endpoint(solution.joints, rng)
                    stepped, current_joints = self._bounded_joint_actions(
                        endpoint, waypoint.hold_steps, state.robot, current_joints
                    )
                    actions.extend(stepped)
                elif waypoint.joints is not None:
                    endpoint = self._jittered_endpoint(waypoint.joints, rng)
                    stepped, current_joints = self._bounded_joint_actions(
                        endpoint, waypoint.hold_steps, state.robot, current_joints
                    )
                    actions.extend(stepped)
                else:
                    opening = waypoint.gripper_opening
                    if opening is None or not math.isfinite(float(opening)):
                        raise ValueError("reviewed gripper opening must be finite")
                    gripper = GripperCmd(float(opening))
                    verdict = self.monitor.validate(
                        gripper, replace(state.robot, joint_pos=current_joints)
                    )
                    if not verdict.accepted:
                        raise ValueError(
                            verdict.reason or "scripted waypoint failed safety validation"
                        )
                    actions.extend([gripper] * waypoint.hold_steps)
            if len(actions) != self.horizon_steps:
                raise ValueError("scripted chunk length contract violated")
            return ActionChunk(list(actions), self.control_dt_s, "scripted")

        return propose

    def _joint_position_limits(self, name: str) -> tuple[float, float]:
        limits = self.monitor.limits.joint_position.get(name)
        if (
            limits is None
            or len(limits) != 2
            or not all(math.isfinite(float(value)) for value in limits)
            or float(limits[0]) > float(limits[1])
        ):
            raise ValueError(f"joint {name} has no reviewed absolute position limits")
        return float(limits[0]), float(limits[1])

    def _jittered_value(
        self, value: float, lower: float, upper: float, rng: np.random.Generator
    ) -> float:
        if self.jitter.std_rad == 0.0:
            return value
        for _ in range(self.jitter.max_attempts):
            delta = float(rng.normal(0.0, self.jitter.std_rad))
            if abs(delta) > self.jitter.limit_rad:
                continue
            candidate = value + delta
            if lower <= candidate <= upper:
                return candidate
        raise ValueError("bounded jitter rejection budget exhausted; no clipped value emitted")

    def _jittered_endpoint(
        self, joints: Mapping[str, float], rng: np.random.Generator
    ) -> dict[str, float]:
        endpoint: dict[str, float] = {}
        for name, value in joints.items():
            raw = float(value)
            if not math.isfinite(raw):
                raise ValueError(f"endpoint for {name} must be finite")
            lower, upper = self._joint_position_limits(str(name))
            if raw < lower or raw > upper:
                raise ValueError(f"endpoint for {name} violates reviewed absolute position limits")
            endpoint[str(name)] = self._jittered_value(raw, lower, upper, rng)
        return endpoint

    def _bounded_joint_actions(
        self,
        endpoint: Mapping[str, float],
        hold_steps: int,
        robot: Any,
        current_joints: dict[str, float],
    ) -> tuple[list[JointTarget], dict[str, float]]:
        """Convert one endpoint into hold_steps executable one-tick joint transitions."""
        for name, endpoint_value in endpoint.items():
            lower, upper = self._joint_position_limits(name)
            value = float(endpoint_value)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"endpoint for {name} violates reviewed absolute position limits")
        current = dict(current_joints)
        actions: list[JointTarget] = []
        for _ in range(int(hold_steps)):
            target: dict[str, float] = {}
            for name, endpoint_value in endpoint.items():
                if name not in current or not math.isfinite(current[name]):
                    raise ValueError(f"no finite current position for bounded joint {name}")
                lower, upper = self._joint_position_limits(name)
                if current[name] < lower or current[name] > upper:
                    raise ValueError(f"current position for {name} violates reviewed joint limits")
                delta = float(endpoint_value) - current[name]
                if abs(delta) <= self.max_joint_step_rad:
                    target[name] = float(endpoint_value)
                else:
                    direction = 1.0 if delta > 0.0 else -1.0
                    target[name] = current[name] + direction * self.max_joint_step_rad
            action = JointTarget(target)
            verdict = self.monitor.validate(action, replace(robot, joint_pos=dict(current)))
            if not verdict.accepted:
                raise ValueError(verdict.reason or "scripted joint step failed safety validation")
            actions.append(action)
            current.update(target)
        return actions, current
