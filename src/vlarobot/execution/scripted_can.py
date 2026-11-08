"""Brain-side scripted CAN pick-and-place policy used for M1 regression checks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np

from vlarobot.backends.webots.config import (
    ALIGN_TOL,
    APPROACH_H,
    ARM_JOINTS,
    BASE_DRIFT_FAIL_POSITION_M,
    BASE_DRIFT_FAIL_YAW_DEG,
    BASE_DRIFT_RATE_LIMIT_M_PER_S,
    BASE_DRIFT_RATE_WINDOW_S,
    BASE_DRIFT_WARN_POSITION_M,
    BASE_DRIFT_WARN_YAW_DEG,
    CAN_PLACE_Z_TOLERANCE_M,
    CAN_TILT_LIMIT_DEG,
    CLOSE_ALIGN_TOL,
    GRASP_Z,
    GRIPPER_OPEN,
    GRIPPER_OPEN_TIMEOUT_S,
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
    PLACE_TOLERANCE_M,
    PLACE_WORLD,
    PREGRASP_REALIGN,
    PREGRASP_ROUTE,
    PREGRASP_SETTLE_TOLERANCE_RAD,
    STOP_AFTER,
    TABLE_CLEARANCE_REQUIRED_M,
    TABLE_FOOTPRINT_CENTRE_XY_M,
    TABLE_FOOTPRINT_SIZE_XY_M,
    TABLE_TOP_HEIGHT_M,
    TIME_STEP,
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

# The original accepts a finger as open within 2 mm of OPEN_POS
# (``controllers/tiago_arm_api/tiago_arm_api.py:326``).
GRIPPER_OPEN_TOLERANCE_M = 0.002


class TimingSummary(TypedDict):
    """Opt-in wall-time counters aggregated by component and by state."""

    enabled: bool
    total_wall_s: float
    totals: dict[str, float]
    by_state: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ScriptedCanResult:
    seed: int
    success: bool
    rejections: int
    final_target_distance: float
    final_state: str
    failure_state: str | None = None
    failure_reason: str | None = None
    max_base_drift_m: float | None = None
    max_base_drift_yaw_deg: float | None = None
    max_base_tilt_deg: float | None = None
    torso_start_rad: float | None = None
    torso_end_rad: float | None = None
    reset_fingerprint: dict[str, object] | None = None
    pregrasp_realign: bool = True
    home_reached: bool = False
    place_distance_ok: bool = False
    can_height_ok: bool = False
    can_tilt_ok: bool = False
    can_height_error_m: float | None = None
    can_tilt_deg: float | None = None
    final_can_world_xyz_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class CanSuccessCriteria:
    """Final CAN acceptance criteria, reported separately so each can be audited."""

    distance_m: float
    can_world_xyz_m: tuple[float, float, float]
    rest_height_m: float
    height_error_m: float
    tilt_deg: float
    home_reached: bool
    place_distance_ok: bool
    can_height_ok: bool
    can_tilt_ok: bool

    @property
    def success(self) -> bool:
        return (
            self.home_reached and self.place_distance_ok and self.can_height_ok and self.can_tilt_ok
        )


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
        pregrasp_realign: bool | None = None,
        stop_after: str | None = None,
        timing_enabled: bool = False,
    ) -> None:
        self.kinematics = kinematics or Kinematics()
        self.trace_path = trace_path
        self.pregrasp_route = PREGRASP_ROUTE if pregrasp_route is None else pregrasp_route
        self.pregrasp_realign = (
            PREGRASP_REALIGN if pregrasp_realign is None else bool(pregrasp_realign)
        )
        self.stop_after = STOP_AFTER if stop_after is None else stop_after
        if self.pregrasp_route not in {"interpolated", "safe_via"}:
            raise ValueError("pregrasp_route must be 'interpolated' or 'safe_via'.")
        self.timing_enabled = timing_enabled
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
        self._reset_fingerprint: dict[str, object] | None = None
        self._max_base_drift_m = 0.0
        self._max_base_drift_yaw_deg = 0.0
        self._max_base_tilt_deg = 0.0
        self._drift_state: str | None = None
        self._drift_samples: list[tuple[float, float]] = []
        self._drift_warned_states: set[str] = set()
        self._torso_start_rad: float | None = None
        self._torso_end_rad: float | None = None
        self._torso_command_rad = float(TORSO_HEIGHT or 0.0)
        self.state_timings: dict[str, dict[str, float]] = {}

    def execute(self, backend, seed: int) -> ScriptedCanResult:
        result = self._execute_impl(backend, seed)
        return replace(
            result,
            max_base_drift_m=self._max_base_drift_m,
            max_base_drift_yaw_deg=self._max_base_drift_yaw_deg,
            max_base_tilt_deg=self._max_base_tilt_deg,
            torso_start_rad=self._torso_start_rad,
            torso_end_rad=self._torso_end_rad,
            reset_fingerprint=self._reset_fingerprint,
            pregrasp_realign=self.pregrasp_realign,
        )

    def _execute_impl(self, backend, seed: int) -> ScriptedCanResult:
        self.kinematics.set_seed(seed)
        self._seed = seed
        self._last_failure_reason = None
        self._last_failure_is_safety = False
        self._last_ik_error = None
        self._last_ik_tilt = None
        self._reset_fingerprint = None
        self._max_base_drift_m = 0.0
        self._max_base_drift_yaw_deg = 0.0
        self._max_base_tilt_deg = 0.0
        self._drift_state = None
        self._drift_samples.clear()
        self._drift_warned_states.clear()
        self._torso_start_rad = None
        self._torso_end_rad = None
        self._torso_command_rad = float(TORSO_HEIGHT or 0.0)
        self.state_timings.clear()
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        reset_started = perf_counter()
        observation = backend.reset(ScenarioSpec(), seed)
        self._add_timing("INIT", "backend_reset_s", perf_counter() - reset_started)
        self._last_observation = observation
        self._reset_base_pose_world = observation.robot.base_pose_world
        self._reset_fingerprint = observation.robot.diagnostics.get("reset_fingerprint")
        self._torso_command_rad = float(
            observation.robot.joint_pos.get(TORSO_JOINT, float(TORSO_HEIGHT or 0.0))
        )
        self._track_observation(observation, reset=True)
        # The INIT CAN pose is the table-rest reference for the final height criterion.
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
        rejections = 0
        final_state = "INIT"

        self._trace_boundary(backend, observation, "INIT", "enter")
        opened = self._open_gripper(backend, observation)
        if opened is None:
            rejections += int(self._last_failure_is_safety)
            failure_observation = self._last_observation or observation
            final_distance = self._placement_distance(backend, failure_observation)
            self._trace_boundary(
                backend,
                failure_observation,
                "INIT",
                "exit",
                failed=True,
                final_target_distance_m=final_distance,
            )
            return ScriptedCanResult(
                seed=seed,
                success=False,
                rejections=rejections,
                final_target_distance=final_distance,
                final_state="FAILED",
                failure_state="INIT",
                failure_reason=self._last_failure_reason,
            )
        observation = opened
        initial_targets = self._base_frame_targets(backend, observation)
        # World-anchored targets are refreshed from the current base pose before each state so
        # base creep cannot silently invalidate them. The INIT lift target is only provisional:
        # PREGRASP re-aligns the approach pose and then refreshes it (ADR-0016).
        lift_target = initial_targets["lift"]

        self._trace_waypoint(backend, observation, "descend_grasp", initial_targets["grasp"])
        self._trace_waypoint(backend, observation, "lift", lift_target)

        self._trace_boundary(backend, observation, "INIT", "exit")
        for name in (
            "pregrasp",
            "grasp",
            "lift",
            "transit",
            "place_descend",
            "release",
            "retreat",
            "home",
        ):
            state = self._STATE_NAMES[name]
            if name == "home":
                target = None
            elif name == "lift":
                target = lift_target
            else:
                target = self._base_frame_targets(backend, observation)[name]
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
            if self.pregrasp_realign and movement is not None and state == "PREGRASP":
                current_q, observation = movement
                realigned = self._realign_pregrasp(backend, observation, current_q)
                if realigned is not None:
                    current_q, observation = realigned
                    # LIFT rises from the pose the arm actually reached, so holding the CAN does
                    # not drag it sideways by the tilt-induced offset.
                    refreshed_targets = self._base_frame_targets(backend, observation)
                    lift_target = refreshed_targets["lift"]
                    trace_target = refreshed_targets["pregrasp"]
                    self._trace_event(
                        "lift_target_refreshed",
                        state="PREGRASP",
                        sim_time_s=observation.t,
                        target_base_xyz_m=lift_target.tolist(),
                    )
                movement = realigned
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
                if self.pregrasp_realign:
                    assert target is not None
                    guarded = self._guard_close_alignment(backend, observation, current_q, target)
                    if guarded is None:
                        final_distance = self._placement_distance(backend, observation)
                        self._trace_boundary(
                            backend,
                            observation,
                            "CLOSE",
                            "exit",
                            target,
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
                            failure_reason="misaligned at close",
                        )
                    current_q, observation, target = guarded
                self._trace_boundary(backend, observation, "CLOSE", "enter", target)
                response = backend.apply(GripperCmd(0.0, force=50.0))
                if not response.accepted:
                    rejections += 1
                    self._last_failure_reason = response.reason or "gripper command rejected"
                    self._trace_boundary(
                        backend,
                        observation,
                        "CLOSE",
                        "exit",
                        target,
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
                step_started = perf_counter() if self.timing_enabled else None
                observation = backend.step(1)
                if step_started is not None:
                    self._add_timing("CLOSE", "backend_step_s", perf_counter() - step_started)
                self._last_observation = observation
                self._track_observation(observation, state="CLOSE")
                if self._base_moved(observation, "CLOSE"):
                    final_distance = self._placement_distance(backend, observation)
                    self._trace_boundary(
                        backend,
                        observation,
                        "CLOSE",
                        "exit",
                        target,
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
                self._trace_boundary(backend, observation, "CLOSE", "exit", target)
            elif name == "release":
                self._trace_boundary(backend, observation, "RELEASE", "command", target)
                response = backend.apply(GripperCmd(GRIPPER_OPEN, force=50.0))
                if not response.accepted:
                    rejections += 1
                    self._last_failure_reason = response.reason or "gripper command rejected"
                    self._trace_boundary(
                        backend,
                        observation,
                        "RELEASE",
                        "exit",
                        target,
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
                step_started = perf_counter() if self.timing_enabled else None
                observation = backend.step(1)
                if step_started is not None:
                    self._add_timing("RELEASE", "backend_step_s", perf_counter() - step_started)
                self._last_observation = observation
                self._track_observation(observation, state="RELEASE")
                if self._base_moved(observation, "RELEASE"):
                    final_distance = self._placement_distance(backend, observation)
                    self._trace_boundary(
                        backend,
                        observation,
                        "RELEASE",
                        "exit",
                        target,
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
                self._trace_boundary(backend, observation, "RELEASE", "exit", target)

        final_distance = self._placement_distance(backend, observation)
        criteria = self._success_criteria(backend, observation, final_state)
        self._trace_boundary(
            backend,
            observation,
            "DONE",
            "exit",
            self._base_frame_targets(backend, observation)["place_base"],
            final_distance_m=final_distance,
            **asdict(criteria),
            success=criteria.success,
        )
        return ScriptedCanResult(
            seed=seed,
            success=criteria.success,
            rejections=rejections,
            final_target_distance=final_distance,
            final_state=final_state,
            home_reached=criteria.home_reached,
            place_distance_ok=criteria.place_distance_ok,
            can_height_ok=criteria.can_height_ok,
            can_tilt_ok=criteria.can_tilt_ok,
            can_height_error_m=criteria.height_error_m,
            can_tilt_deg=criteria.tilt_deg,
            final_can_world_xyz_m=criteria.can_world_xyz_m,
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

    def _open_gripper(self, backend, observation: Observation) -> Observation | None:
        """Open both fingers at INIT, mirroring the original controller.

        The original aborts when the fingers cannot open
        (``controllers/tiago_arm_api/tiago_arm_api.py:414``). With closed fingers the DESCEND
        path pushes the can away instead of straddling it, so this must finish before PREGRASP.
        """
        self._last_failure_is_safety = False
        response = backend.apply(GripperCmd(GRIPPER_OPEN))
        if not response.accepted:
            self._last_failure_is_safety = True
            self._last_failure_reason = response.reason or "gripper open command rejected"
            self._trace_event(
                "gripper_rejection",
                state="INIT",
                sim_time_s=observation.t,
                opening_m=GRIPPER_OPEN,
                reason=self._last_failure_reason,
            )
            return None

        max_steps = max(1, int(round(GRIPPER_OPEN_TIMEOUT_S * 1000.0 / TIME_STEP)))
        for _ in range(max_steps):
            observation = backend.step(1)
            self._last_observation = observation
            self._track_observation(observation, state="INIT")
            if observation.robot.gripper_opening >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE_M:
                return observation

        self._last_failure_reason = "gripper failed to open"
        self._trace_event(
            "gripper_timeout",
            state="INIT",
            sim_time_s=observation.t,
            opening_m=float(observation.robot.gripper_opening),
            target_m=GRIPPER_OPEN,
        )
        return None

    def _realign_pregrasp(
        self, backend, observation: Observation, current_q: np.ndarray
    ) -> tuple[np.ndarray, Observation] | None:
        """Re-centre the approach pose over the CAN once the base tilt has settled.

        Extending the arm rocks the base by ~1.5 deg roll/pitch even though the planar drift
        stays under 4 mm / 0.1 deg (``runs/hold-none-02.jsonl``). Because the arm is commanded
        in the base frame, that rocking moves the CAN ~28 mm in the base frame, so the fingertip
        finishes PREGRASP ~30 mm off-centre. DESCEND then approaches diagonally and clips the
        CAN (42 mm push in the successful run). The original has the same blind spot because it
        reuses the INIT ``grasp_tip``. Re-deriving the pregrasp point from the live base pose
        and correcting the XY error before the descent keeps the descent vertical over the CAN.
        """
        target = self._base_frame_targets(backend, observation)["pregrasp"]
        tip = observation.robot.ee_pose.position.as_array()
        error_m = float(np.linalg.norm((target - tip)[:2]))
        self._trace_event(
            "pregrasp_realign",
            state="PREGRASP",
            sim_time_s=observation.t,
            xy_error_m=error_m,
            threshold_m=ALIGN_TOL,
            target_base_xyz_m=target.tolist(),
            tip_base_xyz_m=tip.tolist(),
        )
        if error_m <= ALIGN_TOL:
            return current_q, observation
        moved = self._move_to(backend, current_q, target, "PREGRASP_REALIGN")
        if moved is None:
            return None
        realigned_q, realigned_observation = moved
        residual_m = float(
            np.linalg.norm((target - realigned_observation.robot.ee_pose.position.as_array())[:2])
        )
        self._trace_event(
            "pregrasp_realigned",
            state="PREGRASP",
            sim_time_s=realigned_observation.t,
            xy_error_before_m=error_m,
            xy_error_after_m=residual_m,
        )
        return realigned_q, realigned_observation

    def _tip_to_can_xy_m(self, backend, observation: Observation) -> float:
        """Planar fingertip-to-CAN error in the live base frame."""
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        can_base = _world_to_base(self._can_pose(backend.get_ground_truth()), base_pose_world)
        tip = observation.robot.ee_pose.position.as_array()
        return float(np.linalg.norm((can_base - tip)[:2]))

    def _guard_close_alignment(
        self,
        backend,
        observation: Observation,
        current_q: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, Observation, np.ndarray] | None:
        """Re-descend once if the fingertips are not centred on the CAN before closing.

        The original only warns when the CAN is misaligned at CLOSE (``SKIP_ALIGN_CHECK``).
        Here the guard is active only when ``pregrasp_realign`` is on, so the realign-off mode
        keeps the original semantics. A single refined descent is allowed; if the residual is
        still over ``close_align_tol`` the state fails with ``misaligned at close``.
        """
        error_m = self._tip_to_can_xy_m(backend, observation)
        self._trace_event(
            "close_alignment_check",
            state="CLOSE",
            sim_time_s=observation.t,
            xy_error_m=error_m,
            threshold_m=CLOSE_ALIGN_TOL,
        )
        if error_m <= CLOSE_ALIGN_TOL:
            return current_q, observation, target

        refined = self._base_frame_targets(backend, observation)["grasp"]
        moved = self._move_linear(backend, observation, refined, "DESCEND")
        if moved is None:
            self._last_failure_reason = "misaligned at close"
            self._trace_event(
                "close_misaligned",
                state="CLOSE",
                sim_time_s=observation.t,
                xy_error_m=error_m,
                threshold_m=CLOSE_ALIGN_TOL,
                reason=self._last_failure_reason,
            )
            return None
        current_q, observation = moved
        residual_m = self._tip_to_can_xy_m(backend, observation)
        self._trace_event(
            "close_realign",
            state="CLOSE",
            sim_time_s=observation.t,
            xy_error_before_m=error_m,
            xy_error_after_m=residual_m,
            threshold_m=CLOSE_ALIGN_TOL,
        )
        if residual_m > CLOSE_ALIGN_TOL:
            self._last_failure_reason = "misaligned at close"
            self._trace_event(
                "close_misaligned",
                state="CLOSE",
                sim_time_s=observation.t,
                xy_error_m=residual_m,
                threshold_m=CLOSE_ALIGN_TOL,
                reason=self._last_failure_reason,
            )
            return None
        return current_q, observation, refined

    def _success_criteria(
        self, backend, observation: Observation, final_state: str
    ) -> CanSuccessCriteria:
        """Evaluate every final criterion from the live CAN pose and base frame."""
        can_pose = self._can_pose3d(backend.get_ground_truth())
        can_world = can_pose.position.as_array()
        rest_height = (
            float(self._initial_can_world_xyz[2])
            if self._initial_can_world_xyz is not None
            else float(can_world[2])
        )
        height_error = float(can_world[2] - rest_height)
        rotation = _quaternion_to_matrix(can_pose.orientation)
        tilt_deg = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
        distance = self._placement_distance(backend, observation)
        return CanSuccessCriteria(
            distance_m=distance,
            can_world_xyz_m=_vec3_tuple(can_world),
            rest_height_m=rest_height,
            height_error_m=height_error,
            tilt_deg=tilt_deg,
            home_reached=final_state == "home",
            place_distance_ok=distance <= PLACE_TOLERANCE_M,
            can_height_ok=abs(height_error) <= CAN_PLACE_Z_TOLERANCE_M,
            can_tilt_ok=tilt_deg < CAN_TILT_LIMIT_DEG,
        )

    def _placement_distance(self, backend, observation: Observation) -> float:
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        can_world = self._can_pose(backend.get_ground_truth())
        can_base = _world_to_base(can_world, base_pose_world)
        place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)
        return float(np.linalg.norm(can_base - place_base))

    def _base_frame_targets(self, backend, observation: Observation) -> dict[str, np.ndarray]:
        """Re-derive the world-anchored targets from the current base pose.

        Mirrors the original controller's per-state ``can_base_pos()`` re-read
        (``controllers/tiago_arm_api/tiago_arm_api.py:354-355`` used at ``:436`` and ``:459``).
        LIFT is the pregrasp point, so it rises vertically from the pose the arm actually
        reached after the PREGRASP re-align (see ADR-0016).
        """
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        can_base = _world_to_base(self._can_pose(backend.get_ground_truth()), base_pose_world)
        grasp_tip = can_base + np.array((0.0, 0.0, GRASP_Z), dtype=float)
        pregrasp = grasp_tip + np.array((0.0, 0.0, APPROACH_H), dtype=float)
        place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)
        place_tip = place_base + np.array((0.0, 0.0, GRASP_Z + PLACE_CLEAR), dtype=float)
        place_above = place_tip + np.array((0.0, 0.0, APPROACH_H), dtype=float)
        return {
            "place_base": place_base,
            "pregrasp": pregrasp,
            "grasp": grasp_tip,
            "lift": pregrasp,
            "transit": place_above,
            "place_descend": place_tip,
            "release": place_tip,
            "retreat": place_above,
        }

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
        solve_started = perf_counter() if self.timing_enabled else None
        try:
            q, ik_error, ik_tilt = self.kinematics.solve(
                target,
                current_q,
                restarts=True,
                position_tolerance=ik_tolerance,
            )
            self._last_ik_error, self._last_ik_tilt = ik_error, ik_tilt
            if solve_started is not None:
                self._add_timing(state, "ik_solve_s", perf_counter() - solve_started)
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
            base_pose = self._reset_base_pose_world
            if self._last_observation is not None:
                base_pose = self._last_observation.robot.base_pose_world or base_pose
            via = self._find_safe_via(current_q, q, base_pose)
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
            solve_started = perf_counter() if self.timing_enabled else None
            try:
                q, error, tilt = self.kinematics.solve(
                    point,
                    q,
                    restarts=True,
                    position_tolerance=ik_tolerance,
                    max_joint_step_rad=MAX_JOINT_STEP_RAD,
                )
                self._last_ik_error, self._last_ik_tilt = error, tilt
                if solve_started is not None:
                    self._add_timing(state, "ik_solve_s", perf_counter() - solve_started)
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
        apply_started = perf_counter() if self.timing_enabled else None
        response = backend.apply(JointTarget(joint_target))
        if apply_started is not None:
            self._add_timing(state, "backend_apply_s", perf_counter() - apply_started)
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
            step_started = perf_counter() if self.timing_enabled else None
            observation = backend.step(1)
            if step_started is not None:
                self._add_timing(state, "backend_step_s", perf_counter() - step_started)
            self._last_observation = observation
            self._track_observation(observation, state=state)
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
        if TORSO_JOINT in self.kinematics.idx:
            vector[self.kinematics.idx[TORSO_JOINT]] = self._torso_command_rad
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
            "pregrasp_realign": self.pregrasp_realign,
            "failed": failed,
        }
        record.update(self._trace_measurements(backend, observation))
        drift_m, drift_yaw_deg = self._base_drift(observation)
        record["base_drift_distance_m"] = drift_m
        record["base_drift_yaw_deg"] = drift_yaw_deg
        tilt_roll_deg, tilt_pitch_deg = self._base_tilt_deg(observation)
        record["base_tilt_roll_deg"] = tilt_roll_deg
        record["base_tilt_pitch_deg"] = tilt_pitch_deg
        rate_m_per_s = self._drift_rate_m_per_s(state, observation.t)
        record["base_drift_rate_m_per_s"] = 0.0 if rate_m_per_s is None else rate_m_per_s
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
        record["arm_7_joint_fk_ground_truth_mismatch"] = observation.robot.diagnostics.get(
            "arm_7_joint_fk_ground_truth_mismatch"
        )
        for key in (
            "arm_7_joint_endpoint_offset_world_xyz_m",
            "arm_7_joint_endpoint_reset_offset_world_xyz_m",
            "arm_7_joint_endpoint_offset_change_world_xyz_m",
            "arm_7_joint_ground_truth_source",
            "arm_7_joint_endpoint_fk_validation_available",
        ):
            record[key] = observation.robot.diagnostics.get(key)
        arm_links = observation.robot.diagnostics.get("arm_link_world_xyz_m")
        record["supervisor_arm_link_world_xyz_m"] = arm_links
        record["supervisor_min_table_clearance_m"] = _table_clearance(arm_links)
        record["table_top_height_m"] = TABLE_TOP_HEIGHT_M
        record["table_clearance_required_m"] = TABLE_CLEARANCE_REQUIRED_M
        if state == "INIT" and phase == "enter":
            record["reset_fingerprint"] = observation.robot.diagnostics.get("reset_fingerprint")
            record["reset_torso_before_rad"] = observation.robot.diagnostics.get(
                "reset_torso_before_rad"
            )
            record["reset_torso_after_rad"] = observation.robot.diagnostics.get(
                "reset_torso_after_rad"
            )
            record["controller_reset_state"] = self._controller_reset_state()
        record.update(details)
        write_started = perf_counter() if self.timing_enabled else None
        self._write_trace(record)
        if write_started is not None:
            self._add_timing(state, "trace_write_s", perf_counter() - write_started)

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

    def _base_tilt_deg(self, observation: Observation) -> tuple[float, float]:
        """Roll/pitch change of the base since reset, in degrees.

        The planar drift metric cannot see the base rocking by ~1.5 deg while the arm
        extends. That rocking moves the CAN in the base frame by ~30 mm at arm's length, so it
        is recorded (and reported by ``analyze_base_drift.py``) even though it is not vetoed.
        """
        initial = self._reset_base_pose_world
        current = observation.robot.base_pose_world
        if initial is None or current is None:
            return 0.0, 0.0
        initial_roll, initial_pitch = _roll_pitch_deg(_quaternion_to_matrix(initial.orientation))
        roll, pitch = _roll_pitch_deg(_quaternion_to_matrix(current.orientation))
        return float(roll - initial_roll), float(pitch - initial_pitch)

    def _record_drift_sample(self, state: str, sim_time_s: float, drift_m: float) -> None:
        """Keep a per-state sliding window of base drift for the sustained-rate check."""
        if state != self._drift_state:
            self._drift_state = state
            self._drift_samples.clear()
        self._drift_samples.append((sim_time_s, drift_m))

    def _drift_rate_m_per_s(self, state: str, sim_time_s: float) -> float | None:
        """Mean drift rate over at least ``BASE_DRIFT_RATE_WINDOW_S`` seconds, if available."""
        if state != self._drift_state or not self._drift_samples:
            return None
        cutoff = sim_time_s - BASE_DRIFT_RATE_WINDOW_S
        reference: tuple[float, float] | None = None
        for sample in self._drift_samples:
            if sample[0] <= cutoff:
                reference = sample
            else:
                break
        if reference is None:
            return None
        elapsed = sim_time_s - reference[0]
        if elapsed <= 0.0:
            return None
        current_drift = self._drift_samples[-1][1]
        return (current_drift - reference[1]) / elapsed

    def _base_moved(self, observation: Observation, state: str) -> bool:
        drift_m, drift_yaw_deg = self._base_drift(observation)
        self._record_drift_sample(state, observation.t, drift_m)
        rate_m_per_s = self._drift_rate_m_per_s(state, observation.t)
        tilt_roll_deg, tilt_pitch_deg = self._base_tilt_deg(observation)

        if state not in self._drift_warned_states and (
            drift_m >= BASE_DRIFT_WARN_POSITION_M or abs(drift_yaw_deg) >= BASE_DRIFT_WARN_YAW_DEG
        ):
            self._drift_warned_states.add(state)
            self._trace_event(
                "base_drift_warning",
                state=state,
                sim_time_s=observation.t,
                base_drift_distance_m=drift_m,
                base_drift_yaw_deg=drift_yaw_deg,
                base_drift_rate_m_per_s=rate_m_per_s,
                base_tilt_roll_deg=tilt_roll_deg,
                base_tilt_pitch_deg=tilt_pitch_deg,
                warn_position_m=BASE_DRIFT_WARN_POSITION_M,
                warn_yaw_deg=BASE_DRIFT_WARN_YAW_DEG,
                fail_position_m=BASE_DRIFT_FAIL_POSITION_M,
                fail_yaw_deg=BASE_DRIFT_FAIL_YAW_DEG,
                rate_limit_m_per_s=BASE_DRIFT_RATE_LIMIT_M_PER_S,
            )

        rate_exceeded = rate_m_per_s is not None and rate_m_per_s > BASE_DRIFT_RATE_LIMIT_M_PER_S
        position_exceeded = (
            drift_m > BASE_DRIFT_FAIL_POSITION_M or abs(drift_yaw_deg) > BASE_DRIFT_FAIL_YAW_DEG
        )
        if not rate_exceeded and not position_exceeded:
            return False
        self._last_failure_reason = "base moved"
        self._trace_event(
            "base_moved",
            state=state,
            sim_time_s=observation.t,
            base_drift_distance_m=drift_m,
            base_drift_yaw_deg=drift_yaw_deg,
            base_drift_rate_m_per_s=rate_m_per_s,
            base_tilt_roll_deg=tilt_roll_deg,
            base_tilt_pitch_deg=tilt_pitch_deg,
            rate_exceeded=rate_exceeded,
            position_exceeded=position_exceeded,
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
        x_min, x_max, y_min, y_max = _table_footprint_bounds()
        required_z = TABLE_TOP_HEIGHT_M + TABLE_CLEARANCE_REQUIRED_M
        for q in [start, *path]:
            for _point_name, point, radius in _arm_clearance_points(self.kinematics, q):
                world = _base_to_world(point, base_pose)
                if (
                    x_min - radius <= world[0] <= x_max + radius
                    and y_min - radius <= world[1] <= y_max + radius
                    and world[2] - radius < required_z
                ):
                    return False
        return True

    def _controller_reset_state(self) -> dict[str, object]:
        """Brain-side caches that are reset between episodes."""
        return {
            "ik_seed": self._seed,
            "ik_rng_digest": _rng_digest(self.kinematics.rng),
            "torso_command_rad": self._torso_command_rad,
            "last_commanded_joint_targets": dict(self._last_commanded_joint_targets),
            "state_settle_durations": dict(self.state_settle_durations),
            "state_sim_durations": dict(self.state_sim_durations),
            "state_wall_durations": dict(self.state_wall_durations),
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
        }

    def _track_observation(
        self,
        observation: Observation,
        reset: bool = False,
        state: str = "INIT",
    ) -> None:
        drift_m, drift_yaw_deg = self._base_drift(observation)
        self._max_base_drift_m = max(self._max_base_drift_m, drift_m)
        self._max_base_drift_yaw_deg = max(self._max_base_drift_yaw_deg, abs(drift_yaw_deg))
        tilt_roll_deg, tilt_pitch_deg = self._base_tilt_deg(observation)
        self._max_base_tilt_deg = max(
            self._max_base_tilt_deg, abs(tilt_roll_deg), abs(tilt_pitch_deg)
        )
        torso = observation.robot.joint_pos.get(TORSO_JOINT)
        if torso is None:
            candidate = observation.robot.diagnostics.get("torso_lift_joint_sensor_rad")
            if isinstance(candidate, (int, float)):
                torso = float(candidate)
        if torso is None:
            candidate = observation.robot.diagnostics.get("reset_torso_after_rad")
            if isinstance(candidate, (int, float)):
                torso = float(candidate)
        if torso is not None:
            if reset or self._torso_start_rad is None:
                self._torso_start_rad = float(torso)
            self._torso_end_rad = float(torso)
        if self.timing_enabled:
            diagnostics_seconds = observation.robot.diagnostics.get("backend_diagnostics_compute_s")
            if isinstance(diagnostics_seconds, (int, float)):
                self._add_timing(state, "supervisor_s", diagnostics_seconds)

    def _add_timing(self, state: str, component: str, seconds: float) -> None:
        if not self.timing_enabled:
            return
        bucket = self.state_timings.setdefault(state, {})
        bucket[component] = bucket.get(component, 0.0) + float(seconds)

    def timing_summary(self) -> TimingSummary:
        """Aggregate opt-in wall-time counters by state and component."""
        totals: dict[str, float] = {}
        for bucket in self.state_timings.values():
            for component, seconds in bucket.items():
                totals[component] = totals.get(component, 0.0) + seconds
        return {
            "enabled": self.timing_enabled,
            "total_wall_s": sum(totals.values()),
            "totals": totals,
            "by_state": self.state_timings,
        }

    def _trace_event(self, event: str, **details: object) -> None:
        if self.trace_path is None:
            return
        state = details.get("state")
        write_started = perf_counter() if self.timing_enabled else None
        self._write_trace({"seed": self._seed, "event": event, **details})
        if write_started is not None and isinstance(state, str):
            self._add_timing(state, "trace_write_s", perf_counter() - write_started)

    def _write_trace(self, record: dict[str, object]) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def _full_joint_vector(self, state: dict[str, float]) -> np.ndarray:
        vector = np.zeros(len(self.kinematics.chain.links), dtype=float)
        if TORSO_JOINT in self.kinematics.idx:
            vector[self.kinematics.idx[TORSO_JOINT]] = self._torso_command_rad
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


def _roll_pitch_deg(rotation: np.ndarray) -> tuple[float, float]:
    """Roll and pitch in degrees from a rotation matrix, using the ZYX convention."""
    roll = math.degrees(math.atan2(float(rotation[2, 1]), float(rotation[2, 2])))
    pitch = math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0))))
    return roll, pitch


def _base_to_world(point: np.ndarray, base_pose_world: Pose3D) -> np.ndarray:
    return (
        _quaternion_to_matrix(base_pose_world.orientation) @ np.asarray(point, dtype=float)
        + base_pose_world.position.as_array()
    )


def _table_footprint_bounds() -> tuple[float, float, float, float]:
    """Return the table footprint as ``(x_min, x_max, y_min, y_max)`` in world metres."""
    centre_x, centre_y = TABLE_FOOTPRINT_CENTRE_XY_M
    size_x, size_y = TABLE_FOOTPRINT_SIZE_XY_M
    return (
        centre_x - 0.5 * size_x,
        centre_x + 0.5 * size_x,
        centre_y - 0.5 * size_y,
        centre_y + 0.5 * size_y,
    )


def _table_clearance(links: object) -> float | None:
    """Minimum Supervisor link height above the table top, for links inside the footprint."""
    if not isinstance(links, dict):
        return None
    x_min, x_max, y_min, y_max = _table_footprint_bounds()
    clearances = []
    for position in links.values():
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            continue
        try:
            x, y, z = (float(value) for value in position)
        except (TypeError, ValueError):
            continue
        if x_min <= x <= x_max and y_min <= y <= y_max:
            clearances.append(z - TABLE_TOP_HEIGHT_M)
    return min(clearances) if clearances else None


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


def _rng_digest(rng: object) -> str | None:
    """Stable digest of an RNG's internal state, for reset-fingerprint comparisons."""
    bit_generator = getattr(rng, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if not isinstance(state, dict):
        return None
    payload = json.dumps(state, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)
