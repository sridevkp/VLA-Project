"""Brain-side scripted CAN pick-and-place policy used for M1 regression checks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from vlarobot.backends.webots.config import (
    APPROACH_H,
    ARM_JOINTS,
    BASE_DRIFT_POSITION_TOLERANCE_M,
    BASE_DRIFT_YAW_TOLERANCE_DEG,
    GRASP_Z,
    HOME_SETTLE_TOLERANCE_RAD,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    JOINT_MOVE_TIMEOUT_STEPS,
    JOINT_PROGRESS_EPSILON_RAD,
    JOINT_STALL_STEPS,
    LINEAR_FINAL_SETTLE_TOLERANCE_RAD,
    LINEAR_MOVE_TIMEOUT_STEPS,
    LINEAR_SETTLE_TOLERANCE_RAD,
    LINEAR_WAYPOINT_SPACING_M,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
    PLACE_CLEAR,
    PLACE_WORLD,
    PREGRASP_ROUTE,
    PREGRASP_SETTLE_TOLERANCE_RAD,
    STOP_AFTER,
    TORSO_HEIGHT,
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import (
    GripperCmd,
    GroundTruth,
    JointTarget,
    Observation,
    Pose3D,
    ScenarioSpec,
)


@dataclass(frozen=True)
class ScriptedCanResult:
    seed: int
    success: bool
    rejections: int
    final_target_distance: float
    final_state: str
    failure_state: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ResetProbeWaypoint:
    joint_targets: tuple[tuple[str, float], ...]
    achieved_position: tuple[float, float, float]
    achieved_orientation: tuple[float, float, float, float]
    gripper_opening: float


@dataclass(frozen=True)
class ResetProbeResult:
    seed: int
    reset_duration_s: float
    probe_duration_s: float
    state_sequence: tuple[str, ...]
    commanded_pregrasp: tuple[float, float, float]
    waypoints: tuple[ResetProbeWaypoint, ...]
    achieved_position: tuple[float, float, float]
    achieved_orientation: tuple[float, float, float, float]
    gripper_opening: float
    reset_joint_positions: tuple[tuple[str, float], ...]
    reset_base_position: tuple[float, float, float]
    reset_base_orientation: tuple[float, float, float, float]
    reset_ee_position: tuple[float, float, float]
    reset_ee_orientation: tuple[float, float, float, float]
    reset_can_position: tuple[float, float, float]
    reset_can_orientation: tuple[float, float, float, float]
    rgb_at_reset: np.ndarray
    rgb_warmup_samples: tuple[tuple[int, np.ndarray], ...]
    rgb_after_warmup: np.ndarray
    failure_reason: str | None = None


class ScriptedCanPolicy:
    """Drive the CAN object through the original waypoint sequence in joint space."""

    _STATE_NAMES = {
        "pregrasp": "PREGRASP",
        "grasp": "DESCEND",
        "lift": "LIFT",
        "transit": "TRANSIT",
        "place_descend": "PLACE_DESCEND",
        "release": "RELEASE",
        "retreat": "RETREAT",
        "home": "HOME",
    }

    def __init__(
        self,
        kinematics: Kinematics | None = None,
        trace_path: Path | None = None,
        pregrasp_route: str | None = None,
        stop_after: str | None = None,
    ) -> None:
        self.kinematics = kinematics or Kinematics()
        self.trace_path = trace_path
        self.pregrasp_route = PREGRASP_ROUTE if pregrasp_route is None else pregrasp_route
        self.stop_after = STOP_AFTER if stop_after is None else stop_after
        if self.pregrasp_route not in {"interpolated", "safe_via"}:
            raise ValueError("pregrasp_route must be 'interpolated' or 'safe_via'.")
        self._seed = 0
        self._last_failure_reason: str | None = None
        self._last_failure_is_safety = False
        self._initial_can_world_xyz: np.ndarray | None = None
        self._last_ik_error: float | None = None
        self._last_ik_tilt: float | None = None
        self._last_commanded_joint_targets: dict[str, float] = {}
        self._state_sim_start: dict[str, float] = {}
        self.state_sim_durations: dict[str, float] = {}
        self.state_settle_durations: dict[str, float] = {}
        self.state_wall_durations: dict[str, float] = {}
        self._state_wall_start: dict[str, float] = {}
        self._last_observation: Observation | None = None
        self._reset_base_pose_world: Pose3D | None = None

    def execute(self, backend, seed: int) -> ScriptedCanResult:
        self.kinematics.set_seed(seed)
        self._seed = seed
        self._last_failure_reason = None
        self._last_failure_is_safety = False
        self._last_ik_error = None
        self._last_ik_tilt = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        observation = backend.reset(ScenarioSpec(), seed)
        self._last_observation = observation
        self._reset_base_pose_world = observation.robot.base_pose_world
        if self.trace_path is not None:
            self._initial_can_world_xyz = self._can_pose(backend.get_ground_truth()).copy()
        current_q = self._full_joint_vector(observation.robot.joint_pos)
        self.state_sim_durations.clear()
        self.state_settle_durations.clear()
        self.state_wall_durations.clear()
        self._state_wall_start.clear()
        self._last_commanded_joint_targets = {
            name: float(current_q[self.kinematics.idx[name]]) for name in ARM_JOINTS
        }
        home_q = current_q.copy()
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        rejections = 0
        final_state = "INIT"

        self._trace_boundary(backend, observation, "INIT", "enter")
        place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)
        can_pose = _world_to_base(self._can_pose(backend.get_ground_truth()), base_pose_world)
        grasp_tip = can_pose + np.array((0.0, 0.0, GRASP_Z), dtype=float)
        pregrasp = grasp_tip + np.array((0.0, 0.0, APPROACH_H), dtype=float)
        place_tip = place_base + np.array(
            (0.0, 0.0, GRASP_Z + PLACE_CLEAR),
            dtype=float,
        )
        place_above = place_tip + np.array((0.0, 0.0, APPROACH_H), dtype=float)

        self._trace_waypoint(backend, observation, "descend_grasp", grasp_tip)
        self._trace_waypoint(backend, observation, "lift", pregrasp)

        self._trace_boundary(backend, observation, "INIT", "exit")
        for name, target in (
            ("pregrasp", pregrasp),
            ("grasp", grasp_tip),
            ("lift", pregrasp),
            ("transit", place_above),
            ("place_descend", place_tip),
            ("release", place_tip),
            ("retreat", place_above),
            ("home", None),
        ):
            state = self._STATE_NAMES[name]
            self._state_sim_start[state] = observation.t
            self._state_wall_start[state] = perf_counter()
            trace_target = self.kinematics.forward(home_q)[:3, 3] if name == "home" else target
            self._trace_boundary(backend, observation, state, "enter", trace_target)
            if name == "home":
                movement = self._move_to_joint_target(backend, current_q, home_q, state)
            elif (
                state in {"DESCEND", "LIFT", "TRANSIT", "PLACE_DESCEND", "RETREAT"}
                and target is not None
            ):
                movement = self._move_linear(backend, observation, target, state)
            elif target is not None:
                movement = self._move_to(backend, current_q, target, state)
            else:
                raise RuntimeError(f"Regression waypoint {name!r} is missing its target.")
            if movement is None:
                rejections += int(self._last_failure_is_safety)
                failure_observation = self._last_observation or observation
                self.state_sim_durations[state] = (
                    failure_observation.t - self._state_sim_start[state]
                )
                self.state_wall_durations[state] = perf_counter() - self._state_wall_start[state]
                final_distance = self._placement_distance(backend, failure_observation)
                self._trace_boundary(
                    backend,
                    failure_observation,
                    state,
                    "exit",
                    trace_target,
                    failed=True,
                    final_target_distance_m=final_distance,
                )
                return ScriptedCanResult(
                    seed=seed,
                    success=False,
                    rejections=rejections,
                    final_target_distance=final_distance,
                    final_state="FAILED",
                    failure_state=state,
                    failure_reason=self._last_failure_reason,
                )
            current_q, observation = movement
            self.state_sim_durations[state] = observation.t - self._state_sim_start[state]
            self.state_wall_durations[state] = perf_counter() - self._state_wall_start[state]
            final_state = name
            self._trace_boundary(backend, observation, state, "exit", trace_target)

            if state == self.stop_after:
                final_distance = self._placement_distance(backend, observation)
                return ScriptedCanResult(
                    seed=seed,
                    success=False,
                    rejections=rejections,
                    final_target_distance=final_distance,
                    final_state=name,
                )

            if name == "grasp":
                self._trace_boundary(backend, observation, "CLOSE", "enter", grasp_tip)
                response = backend.apply(GripperCmd(0.0, force=50.0))
                if not response.accepted:
                    rejections += 1
                    self._last_failure_reason = response.reason or "gripper command rejected"
                    self._trace_boundary(
                        backend,
                        observation,
                        "CLOSE",
                        "exit",
                        grasp_tip,
                        failed=True,
                        final_target_distance_m=self._placement_distance(backend, observation),
                    )
                    return ScriptedCanResult(
                        seed=seed,
                        success=False,
                        rejections=rejections,
                        final_target_distance=self._placement_distance(backend, observation),
                        final_state="FAILED",
                        failure_state="CLOSE",
                        failure_reason=self._last_failure_reason,
                    )
                observation = backend.step(1)
                self._last_observation = observation
                if self._base_moved(observation, "CLOSE"):
                    final_distance = self._placement_distance(backend, observation)
                    self._trace_boundary(
                        backend,
                        observation,
                        "CLOSE",
                        "exit",
                        grasp_tip,
                        failed=True,
                        final_target_distance_m=final_distance,
                    )
                    return ScriptedCanResult(
                        seed=seed,
                        success=False,
                        rejections=rejections,
                        final_target_distance=final_distance,
                        final_state="FAILED",
                        failure_state="CLOSE",
                        failure_reason=self._last_failure_reason,
                    )
                self._trace_boundary(backend, observation, "CLOSE", "exit", grasp_tip)
            elif name == "release":
                self._trace_boundary(backend, observation, "RELEASE", "command", place_tip)
                response = backend.apply(GripperCmd(0.045, force=50.0))
                if not response.accepted:
                    rejections += 1
                    self._last_failure_reason = response.reason or "gripper command rejected"
                    self._trace_boundary(
                        backend,
                        observation,
                        "RELEASE",
                        "exit",
                        place_tip,
                        failed=True,
                        final_target_distance_m=self._placement_distance(backend, observation),
                    )
                    return ScriptedCanResult(
                        seed=seed,
                        success=False,
                        rejections=rejections,
                        final_target_distance=self._placement_distance(backend, observation),
                        final_state="FAILED",
                        failure_state="RELEASE",
                        failure_reason=self._last_failure_reason,
                    )
                observation = backend.step(1)
                self._last_observation = observation
                if self._base_moved(observation, "RELEASE"):
                    final_distance = self._placement_distance(backend, observation)
                    self._trace_boundary(
                        backend,
                        observation,
                        "RELEASE",
                        "exit",
                        place_tip,
                        failed=True,
                        final_target_distance_m=final_distance,
                    )
                    return ScriptedCanResult(
                        seed=seed,
                        success=False,
                        rejections=rejections,
                        final_target_distance=final_distance,
                        final_state="FAILED",
                        failure_state="RELEASE",
                        failure_reason=self._last_failure_reason,
                    )
                self._trace_boundary(backend, observation, "RELEASE", "exit", place_tip)

        final_distance = self._placement_distance(backend, observation)
        self._trace_boundary(
            backend,
            observation,
            "DONE",
            "exit",
            place_base,
            final_distance_m=final_distance,
        )
        return ScriptedCanResult(
            seed=seed,
            success=final_state == "home" and final_distance < 0.10,
            rejections=rejections,
            final_target_distance=final_distance,
            final_state=final_state,
        )

    def probe_initial_pregrasp(
        self,
        backend,
        seed: int,
        warmup_steps: int = 5,
        replay: ResetProbeResult | None = None,
    ) -> ResetProbeResult:
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer.")

        self._seed = seed
        self.kinematics.set_seed(seed)
        self._last_failure_reason = None
        self._last_failure_is_safety = False
        probe_started = perf_counter()
        reset_started = perf_counter()
        observation = backend.reset(ScenarioSpec(), seed)
        self._last_observation = observation
        self._reset_base_pose_world = observation.robot.base_pose_world
        reset_duration = perf_counter() - reset_started
        rgb_at_reset = observation.rgb.copy()
        reset_ground_truth = backend.get_ground_truth()
        reset_base = observation.robot.base_pose_world
        if reset_base is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        reset_can = self._can_pose3d(reset_ground_truth)
        reset_joint_positions = tuple(sorted(observation.robot.joint_pos.items()))
        reset_base_position = _vec3_tuple(reset_base.position.as_array())
        reset_ee_position = _vec3_tuple(observation.robot.ee_pose.position.as_array())
        reset_ee_orientation = observation.robot.ee_pose.orientation
        reset_can_position = _vec3_tuple(reset_can.position.as_array())
        reset_can_orientation = reset_can.orientation
        sample_steps = {step for step in (1, 3, 5) if step <= warmup_steps}
        warmup_samples = []
        for step in range(1, warmup_steps + 1):
            observation = backend.step(1)
            if step in sample_steps:
                warmup_samples.append((step, observation.rgb.copy()))
        rgb_after_warmup = observation.rgb.copy()

        current_q = self._full_joint_vector(observation.robot.joint_pos)
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        if replay is None:
            can_base = _world_to_base(self._can_pose(backend.get_ground_truth()), base_pose_world)
            pregrasp = can_base + np.array((0.0, 0.0, GRASP_Z + APPROACH_H), dtype=float)
        else:
            pregrasp = np.asarray(replay.commanded_pregrasp, dtype=float)
        waypoints: list[ResetProbeWaypoint] = []
        self._trace_boundary(backend, observation, "INIT", "enter")
        self._trace_boundary(backend, observation, "INIT", "exit")
        self._trace_boundary(backend, observation, "PREGRASP", "enter", pregrasp)
        if replay is None:
            movement = self._move_to(
                backend, current_q, pregrasp, "PREGRASP", waypoint_records=waypoints
            )
            if movement is None:
                achieved = observation.robot.ee_pose
                state_sequence = ("INIT", "PREGRASP", "FAILED")
                failure_reason = self._last_failure_reason
                self._trace_boundary(
                    backend, observation, "PREGRASP", "exit", pregrasp, failed=True
                )
            else:
                _, observation = movement
                achieved = observation.robot.ee_pose
                state_sequence = ("INIT", "PREGRASP")
                failure_reason = None
                self._trace_boundary(backend, observation, "PREGRASP", "exit", pregrasp)
        else:
            failure_reason = None
            for record in replay.waypoints:
                response = backend.apply(JointTarget(dict(record.joint_targets)))
                if not response.accepted:
                    failure_reason = response.reason or "replayed PREGRASP command was rejected"
                    break
                observation = backend.step(1)
                waypoints.append(
                    ResetProbeWaypoint(
                        joint_targets=record.joint_targets,
                        achieved_position=_vec3_tuple(
                            observation.robot.ee_pose.position.as_array()
                        ),
                        achieved_orientation=observation.robot.ee_pose.orientation,
                        gripper_opening=float(observation.robot.gripper_opening),
                    )
                )
                self._trace_boundary(
                    backend,
                    observation,
                    "PREGRASP",
                    "waypoint",
                    pregrasp,
                    joint_target=dict(record.joint_targets),
                    replay=True,
                )
            achieved = observation.robot.ee_pose
            if failure_reason is None:
                state_sequence = ("INIT", "PREGRASP")
                self._trace_boundary(backend, observation, "PREGRASP", "exit", pregrasp)
            else:
                state_sequence = ("INIT", "PREGRASP", "FAILED")
                self._trace_boundary(
                    backend, observation, "PREGRASP", "exit", pregrasp, failed=True
                )

        return ResetProbeResult(
            seed=seed,
            reset_duration_s=reset_duration,
            probe_duration_s=perf_counter() - probe_started,
            state_sequence=state_sequence,
            commanded_pregrasp=_vec3_tuple(pregrasp),
            waypoints=tuple(waypoints),
            achieved_position=_vec3_tuple(achieved.position.as_array()),
            achieved_orientation=achieved.orientation,
            gripper_opening=float(observation.robot.gripper_opening),
            reset_joint_positions=reset_joint_positions,
            reset_base_position=reset_base_position,
            reset_base_orientation=reset_base.orientation,
            reset_ee_position=reset_ee_position,
            reset_ee_orientation=reset_ee_orientation,
            reset_can_position=reset_can_position,
            reset_can_orientation=reset_can_orientation,
            rgb_at_reset=rgb_at_reset,
            rgb_warmup_samples=tuple(warmup_samples),
            rgb_after_warmup=rgb_after_warmup,
            failure_reason=failure_reason,
        )

    def _placement_distance(self, backend, observation: Observation) -> float:
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        can_world = self._can_pose(backend.get_ground_truth())
        can_base = _world_to_base(can_world, base_pose_world)
        place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)
        return float(np.linalg.norm(can_base - place_base))

    def _move_to(
        self,
        backend,
        current_q: np.ndarray,
        target: np.ndarray,
        state: str,
        waypoint_records: list[ResetProbeWaypoint] | None = None,
    ) -> tuple[np.ndarray, Observation] | None:
        self._last_failure_is_safety = False
        ik_tolerance = _ik_tolerance_for_state(state)
        try:
            q, ik_error, ik_tilt = self.kinematics.solve(
                target,
                current_q,
                restarts=True,
                position_tolerance=ik_tolerance,
            )
            self._last_ik_error, self._last_ik_tilt = ik_error, ik_tilt
        except RuntimeError as error:
            self._last_failure_reason = f"IK solve failed: {error}"
            self._trace_event(
                "ik_rejection",
                state=state,
                target_base_xyz_m=target.tolist(),
                reason=self._last_failure_reason,
            )
            return None

        if ik_error > ik_tolerance or ik_tilt > MAX_TILT_DEG:
            self._last_failure_reason = (
                f"IK rejected: error={ik_error:.6f}m (limit {ik_tolerance:.6f}m), "
                f"tilt={ik_tilt:.3f}deg (limit {MAX_TILT_DEG:.3f}deg)"
            )
            self._trace_event(
                "ik_rejection",
                state=state,
                target_base_xyz_m=target.tolist(),
                reason=self._last_failure_reason,
                ik_error_m=ik_error,
                ik_tilt_deg=ik_tilt,
                ik_tolerance_m=ik_tolerance,
            )
            return None

        observation = None
        if state == "PREGRASP" and self.pregrasp_route == "safe_via":
            via = self._find_safe_via(current_q, q, self._reset_base_pose_world)
            if via is None:
                self._last_failure_reason = "no safe_via clearance posture found"
                self._trace_event(
                    "safe_via_rejection", state=state, reason=self._last_failure_reason
                )
                return None
            first = self._move_to_joint_target(backend, current_q, via, state)
            if first is None:
                return None
            current_q, observation = first
            self._trace_event(
                "safe_via_selected",
                state=state,
                joint_target_rad={
                    name: float(via[self.kinematics.idx[name]]) for name in ARM_JOINTS
                },
            )
        for waypoint in self.kinematics.interpolate_joint_path(
            current_q, q, MAX_JOINT_STEP_RAD, min_steps=2
        ):
            joint_target = {name: float(waypoint[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            settled = self._command_and_wait(
                backend,
                joint_target,
                state,
                start_q=current_q if observation is None else self._commanded_vector(),
                start_time=self._state_sim_start.get(state, 0.0)
                if observation is None
                else observation.t,
                settle_tolerance=PREGRASP_SETTLE_TOLERANCE_RAD,
                max_steps=JOINT_MOVE_TIMEOUT_STEPS,
                target_position=target,
                ik_error=ik_error,
                ik_tilt=ik_tilt,
            )
            if settled is None:
                return None
            observation = settled
            self._last_commanded_joint_targets = joint_target.copy()
            current_q = self._commanded_vector()
            if waypoint_records is not None:
                waypoint_records.append(
                    ResetProbeWaypoint(
                        joint_targets=tuple(sorted(joint_target.items())),
                        achieved_position=_vec3_tuple(
                            observation.robot.ee_pose.position.as_array()
                        ),
                        achieved_orientation=observation.robot.ee_pose.orientation,
                        gripper_opening=float(observation.robot.gripper_opening),
                    )
                )
            self._trace_boundary(
                backend,
                observation,
                state,
                "waypoint",
                target,
                joint_target=joint_target,
                ik_error_m=ik_error,
                ik_tilt_deg=ik_tilt,
            )

        if observation is None:
            self._last_failure_reason = "no joint interpolation waypoint was executed"
            return None
        return self._commanded_vector(), observation

    def _move_linear(
        self,
        backend,
        observation: Observation,
        target: np.ndarray,
        state: str,
    ) -> tuple[np.ndarray, Observation] | None:
        """Follow the original 3 cm Cartesian path, settling each trust-region IK waypoint."""
        self._last_failure_is_safety = False
        ik_tolerance = _ik_tolerance_for_state(state)
        start = observation.robot.ee_pose.position.as_array()
        target = np.asarray(target, dtype=float)
        count = max(2, int(float(np.linalg.norm(target - start)) / LINEAR_WAYPOINT_SPACING_M))
        q = self._commanded_vector()
        observation_now = observation
        for index, alpha in enumerate(np.linspace(0.0, 1.0, count + 1)[1:], start=1):
            point = (1.0 - alpha) * start + alpha * target
            previous_q = q.copy()
            try:
                q, error, tilt = self.kinematics.solve(
                    point,
                    q,
                    restarts=True,
                    position_tolerance=ik_tolerance,
                    max_joint_step_rad=MAX_JOINT_STEP_RAD,
                )
                self._last_ik_error, self._last_ik_tilt = error, tilt
            except RuntimeError as error:
                self._last_failure_reason = f"IK solve failed: {error}"
                self._trace_event(
                    "ik_rejection",
                    state=state,
                    target_base_xyz_m=point.tolist(),
                    reason=self._last_failure_reason,
                )
                return None
            if error > ik_tolerance or tilt > MAX_TILT_DEG:
                self._last_failure_reason = (
                    f"IK rejected: error={error:.6f}m (limit {ik_tolerance:.6f}m), "
                    f"tilt={tilt:.3f}deg (limit {MAX_TILT_DEG:.3f}deg)"
                )
                return None
            joint_target = {name: float(q[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            final = index == count
            settled = self._command_and_wait(
                backend,
                joint_target,
                state,
                start_q=previous_q,
                start_time=observation_now.t,
                settle_tolerance=(
                    LINEAR_FINAL_SETTLE_TOLERANCE_RAD if final else LINEAR_SETTLE_TOLERANCE_RAD
                ),
                max_steps=LINEAR_MOVE_TIMEOUT_STEPS,
                target_position=point,
                ik_error=error,
                ik_tilt=tilt,
            )
            if settled is None:
                return None
            observation_now = settled
            self._last_commanded_joint_targets = joint_target.copy()
            self._trace_boundary(
                backend,
                settled,
                state,
                "waypoint",
                point,
                joint_target=joint_target,
                ik_error_m=error,
                ik_tilt_deg=tilt,
            )
        return self._commanded_vector(), observation_now

    def _command_and_wait(
        self,
        backend,
        joint_target: dict[str, float],
        state: str,
        start_q: np.ndarray,
        start_time: float,
        settle_tolerance: float,
        max_steps: int,
        target_position: np.ndarray,
        ik_error: float | None,
        ik_tilt: float | None,
    ) -> Observation | None:
        response = backend.apply(JointTarget(joint_target))
        if not response.accepted:
            self._last_failure_is_safety = True
            self._last_failure_reason = response.reason or "joint target rejected"
            self._trace_event(
                "safety_rejection",
                state=state,
                target_base_xyz_m=target_position.tolist(),
                joint_target=joint_target,
                reason=self._last_failure_reason,
                ik_error_m=ik_error,
                ik_tilt_deg=ik_tilt,
            )
            return None

        previous = {name: float(start_q[self.kinematics.idx[name]]) for name in ARM_JOINTS}
        stalled_steps = 0
        for _ in range(max_steps):
            observation = backend.step(1)
            self._last_observation = observation
            measured = observation.robot.joint_pos
            errors = {name: abs(joint_target[name] - measured[name]) for name in ARM_JOINTS}
            diagnostics = observation.robot.diagnostics
            contacts = diagnostics.get("contact_nodes", [])
            if state == "PREGRASP" and isinstance(contacts, list):
                for contact in contacts:
                    if isinstance(contact, dict):
                        self._trace_event(
                            "unexpected_contact",
                            state=state,
                            sim_time_s=observation.t,
                            node_name=contact.get("node_name", "unknown"),
                            node_type=contact.get("node_type"),
                            node_id=contact.get("node_id"),
                            contact_points_world_xyz_m=contact.get(
                                "contact_points_world_xyz_m", []
                            ),
                        )
            if self._base_moved(observation, state):
                self._add_settle_duration(state, observation.t - start_time)
                return None
            if max(errors.values(), default=0.0) < settle_tolerance:
                self._add_settle_duration(state, observation.t - start_time)
                return observation
            progress = max(
                (abs(measured[name] - previous[name]) for name in ARM_JOINTS), default=0.0
            )
            stalled_steps = 0 if progress > JOINT_PROGRESS_EPSILON_RAD else stalled_steps + 1
            if stalled_steps >= JOINT_STALL_STEPS:
                self._last_failure_reason = "joint stall"
                self._trace_event(
                    "joint_stall",
                    state=state,
                    target_base_xyz_m=target_position.tolist(),
                    joint_target=joint_target,
                    no_progress_steps=stalled_steps,
                )
                self._add_settle_duration(state, observation.t - start_time)
                return None
            previous = {name: float(measured[name]) for name in ARM_JOINTS}
        worst = max(ARM_JOINTS, key=lambda name: abs(joint_target[name] - previous[name]))
        self._last_failure_reason = (
            f"move timed out (possibly collision). Worst joint: {worst}, "
            f"error {abs(joint_target[worst] - previous[worst]):.3f} rad"
        )
        self._trace_event(
            "joint_timeout",
            state=state,
            target_base_xyz_m=target_position.tolist(),
            joint_target=joint_target,
            reason=self._last_failure_reason,
        )
        self._add_settle_duration(state, observation.t - start_time)
        return None

    def _add_settle_duration(self, state: str, duration: float) -> None:
        self.state_settle_durations[state] = self.state_settle_durations.get(state, 0.0) + duration

    def _commanded_vector(self) -> np.ndarray:
        vector = np.zeros(len(self.kinematics.chain.links), dtype=float)
        if TORSO_HEIGHT is not None:
            vector[self.kinematics.idx[TORSO_JOINT]] = TORSO_HEIGHT
        for name, value in self._last_commanded_joint_targets.items():
            vector[self.kinematics.idx[name]] = value
        return vector

    def _move_to_joint_target(
        self, backend, current_q: np.ndarray, target_q: np.ndarray, state: str
    ) -> tuple[np.ndarray, Observation] | None:
        self._last_failure_is_safety = False
        self._last_ik_error = 0.0
        self._last_ik_tilt = None
        target_position = self.kinematics.forward(target_q)[:3, 3]
        observation = None
        previous_q = current_q
        for waypoint in self.kinematics.interpolate_joint_path(
            current_q, target_q, MAX_JOINT_STEP_RAD, min_steps=2
        ):
            joint_target = {name: float(waypoint[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            settled = self._command_and_wait(
                backend,
                joint_target,
                state,
                start_q=previous_q,
                start_time=self._state_sim_start.get(state, 0.0)
                if observation is None
                else observation.t,
                settle_tolerance=HOME_SETTLE_TOLERANCE_RAD,
                max_steps=JOINT_MOVE_TIMEOUT_STEPS,
                target_position=target_position,
                ik_error=0.0,
                ik_tilt=None,
            )
            if settled is None:
                return None
            observation = settled
            self._last_commanded_joint_targets = joint_target.copy()
            previous_q = self._commanded_vector()
            self._trace_boundary(
                backend,
                observation,
                state,
                "waypoint",
                target_position,
                joint_target=joint_target,
                ik_error_m=0.0,
            )
        if observation is None:
            self._last_failure_reason = "no home interpolation waypoint was executed"
            return None
        return self._commanded_vector(), observation

    def _trace_boundary(
        self,
        backend,
        observation: Observation,
        state: str,
        phase: str,
        target: np.ndarray | None = None,
        failed: bool = False,
        **details: object,
    ) -> None:
        if self.trace_path is None:
            return
        record: dict[str, object] = {
            "seed": self._seed,
            "event": "state_boundary",
            "phase": phase,
            "state": state,
            "sim_time_s": observation.t,
            "commanded_base_xyz_m": None if target is None else target.tolist(),
            "commanded_ee_base_xyz_m": None if target is None else target.tolist(),
            "measured_joint_positions_rad": observation.robot.joint_pos,
            "commanded_joint_target_rad": self._last_commanded_joint_targets.copy(),
            "commanded_joint_positions_rad": observation.robot.joint_pos,
            "achieved_ee_base_xyz_m": observation.robot.ee_pose.position.as_array().tolist(),
            "joint_positions_rad": observation.robot.joint_pos,
            "contact_nodes": observation.robot.diagnostics.get("contact_nodes", []),
            "gripper_opening_m": observation.robot.gripper_opening,
            "grasp_detected": observation.robot.gripper_holding,
            "failed": failed,
        }
        record.update(self._trace_measurements(backend, observation))
        drift_m, drift_yaw_deg = self._base_drift(observation)
        record["base_drift_distance_m"] = drift_m
        record["base_drift_yaw_deg"] = drift_yaw_deg
        record["base_drift_world_xyz_m"] = observation.robot.diagnostics.get(
            "base_drift_world_xyz_m"
        )
        record["torso_lift_joint_sensor_rad"] = observation.robot.diagnostics.get(
            "torso_lift_joint_sensor_rad", observation.robot.joint_pos.get(TORSO_JOINT)
        )
        record["wheel_motors"] = observation.robot.diagnostics.get("wheel_motors", {})
        record["arm_7_joint_fk_world_xyz_m"] = observation.robot.diagnostics.get(
            "arm_7_joint_fk_world_xyz_m"
        )
        record["arm_7_joint_ground_truth_world_xyz_m"] = observation.robot.diagnostics.get(
            "arm_7_joint_ground_truth_world_xyz_m"
        )
        record["arm_7_joint_fk_ground_truth_error_m"] = observation.robot.diagnostics.get(
            "arm_7_joint_fk_ground_truth_error_m"
        )
        record.update(details)
        self._write_trace(record)

    def _trace_waypoint(
        self, backend, observation: Observation, name: str, point: np.ndarray
    ) -> None:
        if self.trace_path is None:
            return
        self._trace_event(
            "waypoint",
            state="INIT",
            waypoint=name,
            base_xyz_m=point.tolist(),
            commanded_ee_base_xyz_m=point.tolist(),
            **self._trace_measurements(backend, observation),
        )

    def _trace_measurements(self, backend, observation: Observation) -> dict[str, object]:
        can_pose = self._can_pose3d(backend.get_ground_truth())
        can_world = can_pose.position.as_array()
        initial = self._initial_can_world_xyz
        if initial is None:
            initial = can_world
        displacement = can_world - initial
        joint_limits = {
            name: [float(value) for value in self.kinematics.bounds(name)] for name in ARM_JOINTS
        }
        record: dict[str, object] = {
            "can_world_xyz_m": can_world.tolist(),
            "can_world_orientation_matrix": _quaternion_to_matrix(can_pose.orientation)
            .reshape(-1)
            .tolist(),
            "can_displacement_world_xyz_m": displacement.tolist(),
            "lift_height_m": float(displacement[2]),
            "gripper_opening_m": observation.robot.gripper_opening,
            "grasp_detected": observation.robot.gripper_holding,
            "close_stall_counter": None,
            "ik_error_m": self._last_ik_error,
            "ik_tilt_deg": self._last_ik_tilt,
            "joint_limits_rad": joint_limits,
            "measured_joint_positions_rad": observation.robot.joint_pos,
            "commanded_joint_target_rad": self._last_commanded_joint_targets.copy(),
            "joint_positions_rad": observation.robot.joint_pos,
        }
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is not None:
            record["can_base_xyz_m"] = _world_to_base(can_world, base_pose_world).tolist()
            base_rotation = _quaternion_to_matrix(base_pose_world.orientation)
            can_rotation = _quaternion_to_matrix(can_pose.orientation)
            record["can_base_orientation_matrix"] = (
                (base_rotation.T @ can_rotation).reshape(-1).tolist()
            )
            record["base_world_xyz_m"] = base_pose_world.position.as_array().tolist()
            record["base_world_orientation_xyzw"] = list(base_pose_world.orientation)
        return record

    def _base_drift(self, observation: Observation) -> tuple[float, float]:
        diagnostics = observation.robot.diagnostics
        drift = diagnostics.get("base_drift_distance_m")
        yaw = diagnostics.get("base_drift_yaw_deg")
        if isinstance(drift, (int, float)) and isinstance(yaw, (int, float)):
            return float(drift), float(yaw)
        initial = self._reset_base_pose_world
        current = observation.robot.base_pose_world
        if initial is None or current is None:
            return 0.0, 0.0
        delta = current.position.as_array() - initial.position.as_array()
        yaw_delta_raw = _yaw_from_quaternion(current.orientation) - _yaw_from_quaternion(
            initial.orientation
        )
        yaw_delta = math.atan2(
            math.sin(yaw_delta_raw),
            math.cos(yaw_delta_raw),
        )
        return float(np.linalg.norm(delta[:2])), float(np.degrees(yaw_delta))

    def _base_moved(self, observation: Observation, state: str) -> bool:
        drift_m, drift_yaw_deg = self._base_drift(observation)
        if (
            drift_m <= BASE_DRIFT_POSITION_TOLERANCE_M
            and abs(drift_yaw_deg) <= BASE_DRIFT_YAW_TOLERANCE_DEG
        ):
            return False
        self._last_failure_reason = "base moved"
        self._trace_event(
            "base_moved",
            state=state,
            sim_time_s=observation.t,
            base_drift_distance_m=drift_m,
            base_drift_yaw_deg=drift_yaw_deg,
            contact_nodes=observation.robot.diagnostics.get("contact_nodes", []),
        )
        return True

    def _find_safe_via(
        self, start: np.ndarray, goal: np.ndarray, base_pose: Pose3D | None
    ) -> np.ndarray | None:
        """Find a deterministic joint-space waypoint whose sampled route clears the table."""
        if base_pose is None:
            return None
        rng = np.random.default_rng(self._seed + 901)
        candidates = []
        for _ in range(600):
            candidate = start.copy()
            for name in ARM_JOINTS:
                lo, hi = self.kinematics.bounds(name)
                candidate[self.kinematics.idx[name]] = rng.uniform(lo, hi)
            candidates.append(candidate)
        for via in candidates:
            if self._joint_path_table_clear(start, via, base_pose) and self._joint_path_table_clear(
                via, goal, base_pose
            ):
                return via
        return None

    def _joint_path_table_clear(
        self, start: np.ndarray, goal: np.ndarray, base_pose: Pose3D
    ) -> bool:
        path = self.kinematics.interpolate_joint_path(start, goal, MAX_JOINT_STEP_RAD)
        for q in [start, *path]:
            for _point_name, point, radius in _arm_clearance_points(self.kinematics, q):
                world = _base_to_world(point, base_pose)
                if (
                    -2.48882837 - radius <= world[0] <= 2.01117163 + radius
                    and -5.5902928 - radius <= world[1] <= -4.0902928 + radius
                    and world[2] - radius < 0.80
                ):
                    return False
        return True

    def _trace_event(self, event: str, **details: object) -> None:
        if self.trace_path is None:
            return
        self._write_trace({"seed": self._seed, "event": event, **details})

    def _write_trace(self, record: dict[str, object]) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def _full_joint_vector(self, state: dict[str, float]) -> np.ndarray:
        vector = np.zeros(len(self.kinematics.chain.links), dtype=float)
        if TORSO_HEIGHT is not None:
            vector[self.kinematics.idx[TORSO_JOINT]] = TORSO_HEIGHT
        for name, value in state.items():
            if name in self.kinematics.idx:
                vector[self.kinematics.idx[name]] = float(value)
        return vector

    @staticmethod
    def _can_pose(ground_truth: GroundTruth) -> np.ndarray:
        pose = ScriptedCanPolicy._can_pose3d(ground_truth)
        return pose.position.as_array()

    @staticmethod
    def _can_pose3d(ground_truth: GroundTruth) -> Pose3D:
        can = ground_truth.object_poses.get("CAN")
        if can is None:
            raise RuntimeError("Ground truth is missing the required CAN pose.")
        return can


def _world_to_base(point_world: np.ndarray, base_pose_world: Pose3D) -> np.ndarray:
    x, y, z, w = base_pose_world.orientation
    rotation_world_from_base = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )
    base_position_world = base_pose_world.position.as_array()
    return rotation_world_from_base.T @ (point_world - base_position_world)


def _quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _base_to_world(point: np.ndarray, base_pose_world: Pose3D) -> np.ndarray:
    return (
        _quaternion_to_matrix(base_pose_world.orientation) @ np.asarray(point, dtype=float)
        + base_pose_world.position.as_array()
    )


def _arm_clearance_points(
    kinematics: Kinematics, q: np.ndarray
) -> list[tuple[str, np.ndarray, float]]:
    transforms = kinematics.chain.forward_kinematics(q, full_kinematics=True)
    names = {link.name: index for index, link in enumerate(kinematics.chain.links)}
    elbow_index = names.get("arm_4_joint", names.get("arm_3_joint", 0))
    wrist_index = names["arm_6_joint"]
    tool_index = names.get("wrist_ft_tool_link", len(transforms) - 1)
    tip_index = len(transforms) - 1
    return [
        ("elbow", np.asarray(transforms[elbow_index])[:3, 3], 0.0),
        ("wrist", np.asarray(transforms[wrist_index])[:3, 3], 0.0),
        ("tool_frame", np.asarray(transforms[tool_index])[:3, 3], 0.0),
        ("tool_tip_hand", np.asarray(transforms[tip_index])[:3, 3], 0.08),
    ]


def _yaw_from_quaternion(quaternion: tuple[float, float, float, float]) -> float:
    return math.atan2(
        2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
        1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
    )


def _ik_tolerance_for_state(state: str) -> float:
    if state == "PREGRASP":
        return IK_TOL
    if state in {"TRANSIT", "RETREAT", "HOME"}:
        return IK_TOL_TRANSIT
    return IK_TOL_GRASP


def _vec3_tuple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])
