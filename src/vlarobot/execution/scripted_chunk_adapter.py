"""Reviewed scripted ActionChunk proposer for the production WorldMind composition.

This is a new, explicit adapter. It is **not** the frozen ``ScriptedCanPolicy`` baseline and
does not replace it. It cannot discover geometry, perform a new grasp, simulate contact, or
recover from a failed IK pose: the caller supplies a reviewed waypoint list, a reviewed pose
solver and reviewed safety limits. Every emitted command is validated before it enters a
candidate chunk; the live ``SafetyMonitor`` still vetoes again through ``execute_first``.

Jitter is deterministic from the draw seed and bounded by rejection sampling. Out-of-bound
draws are discarded and redrawn; they are never clipped or silently accepted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vlarobot.core.types import ActionChunk, GripperCmd, JointTarget, Pose3D, TaskSpec, WorldState
from vlarobot.execution.safety_monitor import SafetyMonitor


class PoseSolution(Protocol):
    joints: dict[str, float]
    position_error_m: float
    orientation_error_rad: float


class PoseSolver(Protocol):
    def solve(self, pose: Pose3D, state: WorldState) -> PoseSolution: ...


@dataclass(frozen=True)
class ScriptedWaypoint:
    """One reviewed station in a scripted chunk.

    Exactly one of ``pose`` and ``gripper_opening`` must be set. ``hold_steps`` is the explicit
    number of control ticks for which that station is repeated.
    """

    label: str
    hold_steps: int
    pose: Pose3D | None = None
    gripper_opening: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("waypoint label must be a non-empty string")
        if isinstance(self.hold_steps, bool) or self.hold_steps < 1:
            raise ValueError("waypoint hold_steps must be a positive integer")
        if (self.pose is None) == (self.gripper_opening is None):
            raise ValueError("exactly one of pose and gripper_opening is required")


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
            for value in (control_dt_s, position_tolerance_m, orientation_tolerance_rad)
        ):
            raise ValueError("explicit positive control tick and IK tolerances required")
        self.waypoints = tuple(waypoints)
        self.solver = solver
        self.monitor = monitor
        self.horizon_steps = int(horizon_steps)
        self.control_dt_s = float(control_dt_s)
        self.jitter = jitter
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
            for waypoint in self.waypoints:
                action: JointTarget | GripperCmd
                if waypoint.pose is not None:
                    solution = self.solver.solve(waypoint.pose, state)
                    residuals = (
                        float(solution.position_error_m),
                        float(solution.orientation_error_rad),
                    )
                    if not all(math.isfinite(value) for value in residuals) or (
                        residuals[0] > self.position_tolerance_m
                        or residuals[1] > self.orientation_tolerance_rad
                    ):
                        raise ValueError("scripted pose residual outside reviewed tolerance")
                    joints = {
                        str(name): float(value) + self._delta(rng)
                        for name, value in solution.joints.items()
                    }
                    action = JointTarget(joints)
                else:
                    opening = waypoint.gripper_opening
                    if opening is None or not math.isfinite(float(opening)):
                        raise ValueError("reviewed gripper opening must be finite")
                    action = GripperCmd(float(opening))
                verdict = self.monitor.validate(action, state.robot)
                if not verdict.accepted:
                    raise ValueError(verdict.reason or "scripted waypoint failed safety validation")
                actions.extend([action] * waypoint.hold_steps)
            if len(actions) != self.horizon_steps:
                raise ValueError("scripted chunk length contract violated")
            return ActionChunk(list(actions), self.control_dt_s, "scripted")

        return propose

    def _delta(self, rng: np.random.Generator) -> float:
        if self.jitter.std_rad == 0.0:
            return 0.0
        for _ in range(self.jitter.max_attempts):
            value = float(rng.normal(0.0, self.jitter.std_rad))
            if abs(value) <= self.jitter.limit_rad:
                return value
        raise ValueError("bounded jitter rejection budget exhausted; no clipped value emitted")
