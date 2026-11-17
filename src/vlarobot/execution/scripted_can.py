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
    CAN_HALF_HEIGHT_M,
    CAN_PLACE_Z_TOLERANCE_M,
    CAN_RADIUS_M,
    CAN_TILT_LIMIT_DEG,
    CLOSE_ALIGN_TOL,
    CLOSED_LOOP_WAYPOINTS,
    FINGER_TIP_OFFSET_M,
    GRASP_SERVO,
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
    PAD_OFFSET_ARM7_SIMFRAME_M,
    PLACE_CLEAR,
    PLACE_TOLERANCE_M,
    PLACE_WORLD,
    PREGRASP_REALIGN,
    PREGRASP_ROUTE,
    PREGRASP_SETTLE_TOLERANCE_RAD,
    PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M,
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
# A GT finger midpoint further than this from the FK tool tip means the configured TOOL_OFFSET
# does not describe the real pads; report it, never silently re-calibrate the control loop.
TOOL_OFFSET_CALIBRATION_TOLERANCE_M = 0.005


class TimingSummary(TypedDict):
    """Opt-in wall-time counters aggregated by component and by state."""

    enabled: bool
    total_wall_s: float
    totals: dict[str, float]
    by_state: dict[str, dict[str, float]]
    steps: int
    wall_per_step_s: float | None


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
    closed_loop_waypoints: bool = True
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
        closed_loop_waypoints: bool | None = None,
        grasp_servo: str | None = None,
        stop_after: str | None = None,
        timing_enabled: bool = False,
        scenario: ScenarioSpec | None = None,
    ) -> None:
        self.kinematics = kinematics or Kinematics()
        self.trace_path = trace_path
        self.pregrasp_route = PREGRASP_ROUTE if pregrasp_route is None else pregrasp_route
        self.pregrasp_realign = (
            PREGRASP_REALIGN if pregrasp_realign is None else bool(pregrasp_realign)
        )
        self.grasp_servo = GRASP_SERVO if grasp_servo is None else str(grasp_servo)
        if self.grasp_servo not in {"none", "gt_pad"}:
            raise ValueError("grasp_servo must be 'none' or 'gt_pad'.")
        self.closed_loop_waypoints = (
            CLOSED_LOOP_WAYPOINTS if closed_loop_waypoints is None else bool(closed_loop_waypoints)
        )
        self.stop_after = STOP_AFTER if stop_after is None else stop_after
        if self.pregrasp_route not in {"interpolated", "safe_via"}:
            raise ValueError("pregrasp_route must be 'interpolated' or 'safe_via'.")
        self.timing_enabled = timing_enabled
        # Scene passed to every backend.reset() in this episode (ADR-0025); None = world pose.
        self.scenario = ScenarioSpec() if scenario is None else scenario
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
        self._step_count = 0
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
            closed_loop_waypoints=self.closed_loop_waypoints,
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
        self._step_count = 0
        self.state_timings.clear()
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        reset_started = perf_counter()
        observation = backend.reset(self.scenario, seed)
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
            if state == "DESCEND":
                self._trace_gt_tip(backend, observation, "DESCEND", "enter")
            if name == "home":
                movement = self._move_to_joint_target(backend, current_q, home_q, state)
            elif state == "DESCEND" and target is not None and self.grasp_servo == "gt_pad":
                # Start descent from achieved pregrasp/hover pose (base frame)
                hover = observation.robot.ee_pose.position.as_array().copy()
                movement = self._servo_descend_to_grasp(backend, observation, current_q, hover)
            elif (
                state in {"DESCEND", "LIFT", "TRANSIT", "PLACE_DESCEND", "RETREAT"}
                and target is not None
            ):
                provider = None
                if self.closed_loop_waypoints and state == "DESCEND":
                    provider = self._waypoint_target_provider(backend, "grasp")
                elif self.closed_loop_waypoints and state == "LIFT":
                    # The CAN is grasped during LIFT, so its live pose moves with the arm. Anchor
                    # the lift point to the entry world pose and only re-express it in the live
                    # base frame, otherwise the target would chase the rising CAN.
                    can_world = self._can_pose(backend.get_ground_truth()).copy()
                    provider = self._waypoint_target_provider(backend, "lift", can_world=can_world)
                movement = self._move_linear(
                    backend, observation, target, state, target_provider=provider
                )
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
            if movement is not None and state == "PREGRASP" and self.grasp_servo == "gt_pad":
                current_q, observation = movement
                pregrasp_now = self._base_frame_targets(backend, observation)["pregrasp"]
                servoed = self._servo_correct_pregrasp_xy(
                    backend, observation, current_q, pregrasp_now, state="PREGRASP"
                )
                if servoed is None:
                    movement = None
                else:
                    current_q, observation, trace_target = servoed
                    # Keep lift target consistent with achieved hover XY.
                    lift_target = trace_target.copy()
                    lift_target = trace_target.copy()
                    movement = (current_q, observation)
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
            if state == "PREGRASP":
                self._trace_gt_tip(backend, observation, "PREGRASP", "exit")

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
                self._state_sim_start["CLOSE"] = observation.t
                self._state_wall_start["CLOSE"] = perf_counter()
                self._trace_gt_tip(backend, observation, "CLOSE", "enter")
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
                # Pace the gripper closing: step until fingers stall against the object or limit
                previous_opening = float(observation.robot.gripper_opening)
                still_count = 0
                max_close_steps = 300
                for i in range(max_close_steps):
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
                    current_opening = float(observation.robot.gripper_opening)
                    if (
                        i == 0
                        and current_opening <= 0.001
                        and abs(current_opening - previous_opening) > 0.01
                    ):
                        break
                    if abs(current_opening - previous_opening) < 2e-5:
                        still_count += 1
                    else:
                        still_count = 0
                    previous_opening = current_opening
                    if (i > 10 and still_count >= 15) or (
                        current_opening <= 0.001 and still_count >= 3
                    ):
                        break
                self._trace_boundary(backend, observation, "CLOSE", "exit", target)
                self.state_sim_durations["CLOSE"] = observation.t - self._state_sim_start["CLOSE"]
                self.state_wall_durations["CLOSE"] = (
                    perf_counter() - self._state_wall_start["CLOSE"]
                )
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
                max_release_steps = max(1, int(round(GRIPPER_OPEN_TIMEOUT_S * 1000.0 / TIME_STEP)))
                for _ in range(max_release_steps):
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
                    if observation.robot.gripper_opening >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE_M:
                        break
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
        observation = backend.reset(self.scenario, seed)
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

        ADR-0023 addition: with ``grasp_servo=gt_pad`` the alignment is measured and gated on
        the MEASURED pad midpoint (Solid-based telemetry, exact) instead of the FK chain tip,
        whose XY position carries a live-measured ~4.2 cm systematic error versus the real
        fingertip midpoint; the re-descent target is shifted by the measured pad error so the
        pad, not the virtual tip point, is centred on the CAN. ``grasp_servo=none`` keeps the
        legacy tip-based measurement unchanged.
        """
        if self.grasp_servo == "gt_pad":
            meas = self._gt_pad_xy_error_world(backend, observation)
            if meas is None:
                self._last_failure_reason = "gt_pad close guard failed: pad midpoint unavailable"
                self._trace_event(
                    "close_alignment_check",
                    state="CLOSE",
                    sim_time_s=observation.t,
                    xy_error_m=None,
                    threshold_m=CLOSE_ALIGN_TOL,
                    reason=self._last_failure_reason,
                )
                return None
            e_xy, error_m, _can_world, _pad_world, source = meas
            self._trace_event(
                "close_alignment_check",
                state="CLOSE",
                sim_time_s=observation.t,
                xy_error_m=error_m,
                threshold_m=CLOSE_ALIGN_TOL,
                source=source,
            )
            if error_m <= CLOSE_ALIGN_TOL:
                return current_q, observation, target
            base_pose_world = observation.robot.base_pose_world
            if base_pose_world is None:
                self._last_failure_reason = "gt_pad close guard failed: missing base_pose_world"
                return None
            R_wb = _quaternion_to_matrix(base_pose_world.orientation)
            step_world = e_xy.copy()
            step_norm = float(np.linalg.norm(step_world))
            if step_norm > 0.010:  # same 10 mm bound as the servo XY corrections
                step_world = step_world * (0.010 / step_norm)
            d_base = R_wb.T @ np.array([step_world[0], step_world[1], 0.0], dtype=float)
            refined = self._base_frame_targets(backend, observation)["grasp"] + d_base
            # Static corrected target: a closed-loop provider would recompute the legacy
            # tip-space target and re-introduce the FK error the correction removes.
            moved = self._move_linear(backend, observation, refined, "CLOSE", target_provider=None)
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
            meas = self._gt_pad_xy_error_world(backend, observation)
            residual_m = meas[1] if meas is not None else float("inf")
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
        provider = (
            self._waypoint_target_provider(backend, "grasp") if self.closed_loop_waypoints else None
        )
        # The re-descent belongs to CLOSE, not DESCEND, so DESCEND settle/sim accounting is not
        # inflated by a state that already exited (ADR-0018).
        moved = self._move_linear(backend, observation, refined, "CLOSE", target_provider=provider)
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

    def _can_world_pose(self, backend) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        pose = self._can_pose3d(backend.get_ground_truth())
        return pose.position.as_array(), pose.orientation

    def _base_world_rotation(self, observation: Observation) -> np.ndarray:
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        return _quaternion_to_matrix(base_pose_world.orientation)

    def _pad_midpoint_world(self, observation: Observation) -> np.ndarray | None:
        """Pad midpoint in world frame from supervisor-derived diagnostics (preferred)."""
        diag = observation.robot.diagnostics
        mid = diag.get("finger_midpoint_world_xyz_m")
        if isinstance(mid, list) and len(mid) == 3:
            arr = np.asarray(mid, dtype=float)
            if np.all(np.isfinite(arr)):
                return arr
        return None

    def _log_servo(
        self, backend, observation: Observation, state: str, phase: str, **extra
    ) -> None:
        can_xyz, _ = self._can_world_pose(backend)
        got = self._pad_midpoint_world_measured(observation)
        pad, source = (None, None) if got is None else got
        details: dict[str, object] = {
            "state": state,
            "phase": phase,
            "sim_time_s": observation.t,
            "can_world_xyz_m": can_xyz.tolist(),
            "pad_mid_world_xyz_m": None if pad is None else pad.tolist(),
            "pad_source": source,
        }
        if pad is not None:
            delta = can_xyz - pad
            details["e_xy_world_m"] = delta[:2].tolist()
            details["e_xy_norm_m"] = float(np.linalg.norm(delta[:2]))
            details["can_minus_pad_world_xyz_m"] = delta.tolist()
            details["pad_mid_z_m"] = float(pad[2])
            details["can_centre_z_m"] = float(can_xyz[2])
            base_pose_world = observation.robot.base_pose_world
            if base_pose_world is not None:
                can_base = _world_to_base(can_xyz, base_pose_world)
                pad_base = _world_to_base(pad, base_pose_world)
                delta_base = can_base - pad_base
                details["can_minus_pad_base_xyz_m"] = delta_base.tolist()
                details["can_minus_pad_base_norm_m"] = float(np.linalg.norm(delta_base))
        details.update(extra)
        self._trace_event("grasp_servo", **details)

    def _arm7_world_pose_from_telemetry(
        self, observation: Observation
    ) -> tuple[np.ndarray, np.ndarray] | None:
        telem = observation.robot.diagnostics.get("kinematics_telemetry")
        if not isinstance(telem, dict):
            return None
        solids = telem.get("joint_endpoint_solids")
        if not isinstance(solids, dict):
            return None
        arm7 = solids.get("arm_7_joint")
        if not isinstance(arm7, dict):
            return None
        pose = arm7.get("world_pose") if "world_pose" in arm7 else arm7
        if not isinstance(pose, dict):
            return None
        pos = pose.get("world_position_m") or pose.get("position")
        rot = pose.get("world_orientation_matrix_row_major") or pose.get("orientation")
        if not isinstance(pos, list) or len(pos) != 3 or not isinstance(rot, list) or len(rot) != 9:
            return None
        p_arr = np.asarray(pos, dtype=float)
        r_arr = np.asarray(rot, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(p_arr)) or not np.all(np.isfinite(r_arr)):
            return None
        return p_arr, r_arr

    def _pad_midpoint_world_measured(
        self, observation: Observation
    ) -> tuple[np.ndarray, str] | None:
        """Measured pad reference in world frame for grasp_servo=gt_pad (ADR-0022).

        Pad midpoint = arm_7 endPoint Solid pose composed with the MEASURED constant offset
        ``(0, 0, +0.202825) m`` in the sim arm_7 Solid frame (config
        ``pad_offset_arm7_simframe_m``, measured at 4 safe poses, spread <= 1.7e-7 m).
        The legacy ``finger_tip_offset_m`` (0.1741, PROTO slider geometry) is deliberately
        NOT used here. Falls back to the same composition from the Solid position and
        orientation diagnostics when the backend-composed key is absent (fake backends);
        fails loudly (None) when neither is available.
        """
        diag = observation.robot.diagnostics
        composed = diag.get("pad_midpoint_arm7_solid_measured_world_xyz_m")
        if isinstance(composed, (list, tuple)) and len(composed) == 3:
            arr = np.asarray(composed, dtype=float)
            if np.all(np.isfinite(arr)):
                return arr, "arm7_solid_plus_measured_offset"
        arm7_pos = diag.get("arm_7_joint_ground_truth_world_xyz_m")
        arm7_rows = diag.get("arm_7_joint_ground_truth_orientation_rows")
        if (
            isinstance(arm7_pos, (list, tuple))
            and len(arm7_pos) == 3
            and isinstance(arm7_rows, (list, tuple))
            and len(arm7_rows) == 3
        ):
            position = np.asarray(arm7_pos, dtype=float)
            rotation = np.asarray(arm7_rows, dtype=float).reshape(3, 3)
            if np.all(np.isfinite(position)) and np.all(np.isfinite(rotation)):
                pad_world = position + rotation @ np.asarray(
                    PAD_OFFSET_ARM7_SIMFRAME_M, dtype=float
                )
                return pad_world, "arm7_solid_plus_measured_offset(composed)"
        arm7_telemetry = self._arm7_world_pose_from_telemetry(observation)
        if arm7_telemetry is not None:
            p_arm7, r_arm7 = arm7_telemetry
            pad_world = p_arm7 + r_arm7 @ np.asarray(PAD_OFFSET_ARM7_SIMFRAME_M, dtype=float)
            return pad_world, "arm7_telemetry_plus_measured_offset"
        return None

    def _gt_pad_xy_error_world(
        self, backend, observation: Observation
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, str] | None:
        got = self._pad_midpoint_world_measured(observation)
        if got is None:
            return None
        pad_world, source = got
        can_world, _ = self._can_world_pose(backend)
        e_xy = (can_world - pad_world)[:2].copy()
        return e_xy, float(np.linalg.norm(e_xy)), can_world, pad_world, source

    def _servo_correct_pregrasp_xy(
        self,
        backend,
        observation: Observation,
        current_q: np.ndarray,
        pregrasp_target_base: np.ndarray,
        state: str = "PREGRASP",
    ) -> tuple[np.ndarray, Observation, np.ndarray] | None:
        """Bounded XY corrections at hover to align pad midpoint over can (oracle)."""
        if self.grasp_servo != "gt_pad":
            return current_q, observation, pregrasp_target_base
        target = np.asarray(pregrasp_target_base, dtype=float).copy()
        max_step = 0.010  # 10 mm
        tol = 0.004
        for iteration in range(1, 7):
            err = self._gt_pad_xy_error_world(backend, observation)
            if err is None:
                self._last_failure_reason = "gt_pad servo failed: pad midpoint unavailable"
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "pregrasp_unavailable",
                    iteration=iteration,
                    reason=self._last_failure_reason,
                )
                return None
            e_xy, e_norm, can_world, pad_world, source = err
            self._log_servo(
                backend,
                observation,
                state,
                "pregrasp_iter",
                iteration=iteration,
                source=source,
                target_base_xyz_m=target.tolist(),
            )
            if e_norm < tol:
                return current_q, observation, target
            step_world = e_xy.copy()
            step_norm = float(np.linalg.norm(step_world))
            if step_norm > max_step:
                step_world = step_world * (max_step / step_norm)
            # world XY correction -> base frame XY (ignore yaw coupling for small bounded step)
            base_pose_world = observation.robot.base_pose_world
            if base_pose_world is None:
                self._last_failure_reason = "gt_pad servo failed: missing base_pose_world"
                return None
            R_wb = _quaternion_to_matrix(base_pose_world.orientation)
            # Move pad toward can in world XY: d_world = step_world (bounded).
            # Approx target delta in base: R_wb^T * d_world.
            d_base = R_wb.T @ np.array([step_world[0], step_world[1], 0.0], dtype=float)
            target = target + d_base
            moved = self._move_to(backend, current_q, target, state)
            if moved is None:
                if self._last_failure_reason is None:
                    self._last_failure_reason = "gt_pad servo correction move failed"
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "pregrasp_move_failed",
                    iteration=iteration,
                    reason=self._last_failure_reason,
                )
                return None
            current_q, observation = moved
        # final check
        err = self._gt_pad_xy_error_world(backend, observation)
        if err is None:
            self._last_failure_reason = (
                "gt_pad servo failed: pad midpoint unavailable after corrections"
            )
            return None
        e_xy, e_norm, _can_world, _pad_world, source = err
        self._log_servo(
            backend,
            observation,
            state,
            "pregrasp_final",
            iteration=6,
            source=source,
            target_base_xyz_m=target.tolist(),
        )
        if e_norm >= tol:
            self._last_failure_reason = (
                f"gt_pad pregrasp did not converge: |e_xy|={e_norm:.6f}m >= {tol}m"
            )
            return None
        return current_q, observation, target

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

    def _base_frame_targets(
        self,
        backend,
        observation: Observation,
        can_world: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Re-derive the world-anchored targets from the current base pose.

        Mirrors the original controller's per-state ``can_base_pos()`` re-read
        (``controllers/tiago_arm_api/tiago_arm_api.py:354-355`` used at ``:436`` and ``:459``).
        LIFT is the pregrasp point, so it rises vertically from the pose the arm actually
        reached after the PREGRASP re-align (see ADR-0016).

        ``can_world`` overrides the live CAN pose. Closed-loop LIFT uses the CAN pose captured at
        LIFT entry so a grasped rising CAN cannot create a target feedback loop (ADR-0018).
        """
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is None:
            raise RuntimeError("RobotState is missing the 3D world-frame base pose.")
        can_world_live = (
            self._can_pose(backend.get_ground_truth())
            if can_world is None
            else np.asarray(can_world, dtype=float)
        )
        can_base = _world_to_base(can_world_live, base_pose_world)
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

    def _waypoint_target_provider(
        self,
        backend,
        name: str,
        can_world: np.ndarray | None = None,
    ):
        """Return a callback that re-expresses one world-anchored target in the live base frame.

        The callback is evaluated once per Cartesian waypoint, not once per state. DESCEND passes
        no ``can_world`` so it tracks the live CAN; LIFT passes the CAN pose captured at entry so
        the grasped CAN's own motion does not move the target (ADR-0018).
        """

        def provider(observation: Observation) -> np.ndarray:
            return self._base_frame_targets(backend, observation, can_world=can_world)[name]

        return provider

    def _servo_descend_to_grasp(
        self,
        backend,
        observation: Observation,
        current_q: np.ndarray,
        hover_target_base: np.ndarray,
    ) -> tuple[np.ndarray, Observation] | None:
        """GT-pad servo descent: XY corrections every 20mm, stop at the probe-measured pad
        -bottom height (lid contact height from the contact-height probe, addition 2)."""
        state = "DESCEND"
        if PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M is None:
            self._last_failure_reason = (
                "gt_pad descent stop height requires the contact-height probe offset; "
                "probe_pad_reference_to_pad_bottom_m is null. Run the contact-height probe first."
            )
            self._log_servo(
                backend,
                observation,
                state,
                "descend_unconfigured",
                reason=self._last_failure_reason,
            )
            return None
        target_xy_base = np.asarray(hover_target_base[:2], dtype=float).copy()
        z = float(hover_target_base[2])
        can_world, _ = self._can_world_pose(backend)
        # Stopping height from the live contact-height probe (not PROTO arithmetic): stop
        # when the measured pad bottom reaches the can lid (can centre z + half height).
        target_pad_bottom_z = float(can_world[2] + CAN_HALF_HEIGHT_M)
        # don't start below a safe margin above can in commanded terms; measured stop governs
        descended_since_correction = 0.0
        step_down = 0.005  # 5 mm commanded steps
        max_total = 0.30
        total_descended = 0.0
        last_meas = self._gt_pad_xy_error_world(backend, observation)
        if last_meas is None:
            self._last_failure_reason = "gt_pad descent failed: pad midpoint unavailable at start"
            self._log_servo(
                backend, observation, state, "descend_unavailable", reason=self._last_failure_reason
            )
            return None
        _e_xy, _e_norm, _cw, pad_world, source = last_meas
        self._log_servo(
            backend,
            observation,
            state,
            "descend_start",
            source=source,
            target_pad_bottom_z_m=target_pad_bottom_z,
        )
        while total_descended < max_total:
            # stopping condition on the measured pad bottom (probe offset, addition 2)
            meas = self._gt_pad_xy_error_world(backend, observation)
            if meas is None:
                self._last_failure_reason = "gt_pad descent failed: pad midpoint unavailable"
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "descend_measure_failed",
                    reason=self._last_failure_reason,
                    total_descended_m=total_descended,
                )
                return None
            e_xy, e_norm, can_world, pad_world, source = meas
            pad_bottom_z = float(pad_world[2]) - PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M
            if pad_bottom_z <= target_pad_bottom_z:
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "descend_reached",
                    source=source,
                    target_pad_bottom_z_m=target_pad_bottom_z,
                    pad_bottom_z_m=pad_bottom_z,
                    total_descended_m=total_descended,
                )
                return current_q, observation
            # re-measure/correct every 20mm of descent
            if descended_since_correction >= 0.020 - 1e-9:
                # bounded XY correction at current height (small step)
                base_pose_world = observation.robot.base_pose_world
                if base_pose_world is None:
                    self._last_failure_reason = "gt_pad descent failed: missing base_pose_world"
                    return None
                R_wb = _quaternion_to_matrix(base_pose_world.orientation)
                step_world = e_xy.copy()
                nrm = float(np.linalg.norm(step_world))
                if nrm > 0.010:
                    step_world = step_world * (0.010 / nrm)
                d_base = R_wb.T @ np.array([step_world[0], step_world[1], 0.0], dtype=float)
                target_xy_base = target_xy_base + d_base[:2]
                descended_since_correction = 0.0
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "descend_realign",
                    source=source,
                    total_descended_m=total_descended,
                )
            # command next downward waypoint at corrected XY
            z = z - step_down
            target = np.array([target_xy_base[0], target_xy_base[1], z], dtype=float)
            # limit each correction/move step; use single-step move_to
            moved = self._move_to(backend, current_q, target, state)
            if moved is None:
                if self._last_failure_reason is None:
                    self._last_failure_reason = "gt_pad descent move failed"
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "descend_move_failed",
                    reason=self._last_failure_reason,
                    total_descended_m=total_descended,
                )
                return None
            current_q, observation = moved
            descended_since_correction += step_down
            total_descended += step_down
            self._log_servo(
                backend,
                observation,
                state,
                "descend_step",
                source=source,
                target_pad_bottom_z_m=target_pad_bottom_z,
                total_descended_m=total_descended,
            )
        # final check
        meas = self._gt_pad_xy_error_world(backend, observation)
        if meas is not None:
            _e_xy, _e_norm, _cw, pad_world, source = meas
            pad_bottom_z = float(pad_world[2]) - PROBE_PAD_REFERENCE_TO_PAD_BOTTOM_M
            if pad_bottom_z <= target_pad_bottom_z:
                self._log_servo(
                    backend,
                    observation,
                    state,
                    "descend_reached",
                    source=source,
                    target_pad_bottom_z_m=target_pad_bottom_z,
                    pad_bottom_z_m=pad_bottom_z,
                    total_descended_m=total_descended,
                )
                return current_q, observation
        self._last_failure_reason = (
            "gt_pad descent failed: did not reach the probe-measured pad-bottom stop height"
        )
        self._log_servo(
            backend,
            observation,
            state,
            "descend_incomplete",
            reason=self._last_failure_reason,
            target_pad_bottom_z_m=target_pad_bottom_z,
            total_descended_m=total_descended,
        )
        return None

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
        target_provider=None,
    ) -> tuple[np.ndarray, Observation] | None:
        """Follow the original 3 cm Cartesian path, settling each trust-region IK waypoint.

        When ``target_provider`` is given, the target is re-derived from the live base pose at
        every waypoint and the remaining fraction of the path is re-interpolated from the achieved
        position. With a fixed target this telescopes to the original ``alpha`` schedule, so the
        closed-loop-off path is byte-for-byte the same arithmetic as before (ADR-0018).
        """
        self._last_failure_is_safety = False
        ik_tolerance = _ik_tolerance_for_state(state)
        start = observation.robot.ee_pose.position.as_array()
        target = np.asarray(target, dtype=float)
        count = max(2, int(float(np.linalg.norm(target - start)) / LINEAR_WAYPOINT_SPACING_M))
        q = self._commanded_vector()
        observation_now = observation
        descend_mid_index = max(1, count // 2) if state == "DESCEND" else None
        for index, alpha in enumerate(np.linspace(0.0, 1.0, count + 1)[1:], start=1):
            live_target: np.ndarray | None = None
            if target_provider is None:
                point = (1.0 - alpha) * start + alpha * target
            else:
                current = observation_now.robot.ee_pose.position.as_array()
                live_target = np.asarray(target_provider(observation_now), dtype=float)
                remaining = count - index + 1
                point = current + (live_target - current) / float(remaining)
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
            waypoint_details: dict[str, object] = {
                "joint_target": joint_target,
                "ik_error_m": error,
                "ik_tilt_deg": tilt,
                "closed_loop_waypoint": target_provider is not None,
            }
            if live_target is not None:
                waypoint_details["live_target_base_xyz_m"] = live_target.tolist()
            self._trace_boundary(
                backend,
                settled,
                state,
                "waypoint",
                point,
                failed=False,
                **waypoint_details,
            )
            if descend_mid_index is not None and index == descend_mid_index:
                # DESCEND mid (batch diagnostics: addition 3): can-minus-pad 3D vectors at
                # the mid waypoint, in world and base frames.
                self._trace_gt_tip(backend, settled, state, "mid")
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
            self._step_count += 1
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
            "arm_7_joint_ground_truth_orientation_rows",
            "arm_7_joint_endpoint_fk_validation_available",
            "pad_midpoint_arm7_solid_measured_world_xyz_m",
            "base_velocity_linear_m_s",
            "base_velocity_angular_rad_s",
            "ee_fk_world_xyz_m",
            "finger_tip_world_xyz_m",
            "finger_midpoint_world_xyz_m",
            "finger_midpoint_source",
            "finger_midpoint_fk_offset_world_xyz_m",
            "finger_midpoint_fk_offset_m",
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

    def _gt_finger_payload(
        self,
        backend,
        observation: Observation,
        can_world: np.ndarray,
    ) -> dict[str, object]:
        """Real finger telemetry beside the FK tool tip, including per-side CAN clearance.

        Everything comes from Supervisor diagnostics (``finger_midpoint_world_xyz_m``,
        ``finger_tip_world_xyz_m``, ``ee_fk_world_xyz_m``) and the GT CAN pose. When diagnostics
        lack the FK world tip (fake backends), it is re-derived from the base-frame FK EE and the
        live base pose, so the offset itself stays a real computed number.
        """
        diagnostics = observation.robot.diagnostics
        gt_midpoint = diagnostics.get("finger_midpoint_world_xyz_m")
        finger_tips = diagnostics.get("finger_tip_world_xyz_m")
        ee_fk_world = diagnostics.get("ee_fk_world_xyz_m")
        base_pose_world = observation.robot.base_pose_world
        if ee_fk_world is None and base_pose_world is not None:
            base_rot = _quaternion_to_matrix(base_pose_world.orientation)
            base_pos = np.asarray(base_pose_world.position.as_array(), dtype=float)
            ee_fk_world = (
                base_rot @ observation.robot.ee_pose.position.as_array() + base_pos
            ).tolist()

        payload: dict[str, object] = {
            "gt_midpoint_world_xyz_m": gt_midpoint,
            "finger_tip_world_xyz_m": finger_tips,
            "finger_midpoint_source": diagnostics.get("finger_midpoint_source"),
            "finger_midpoint_fk_offset_world_xyz_m": diagnostics.get(
                "finger_midpoint_fk_offset_world_xyz_m"
            ),
            "ee_fk_world_xyz_m": ee_fk_world,
            "can_radius_m": CAN_RADIUS_M,
            "clearance_left_m": None,
            "clearance_right_m": None,
        }
        # Measured pad reference (ADR-0022 Solid composition) vs can centre, 3D, world+base
        # (batch diagnostics: addition 3). Independent of the legacy 0.1741-based midpoint.
        pad_ref = diagnostics.get("pad_midpoint_arm7_solid_measured_world_xyz_m")
        if not (isinstance(pad_ref, (list, tuple)) and len(pad_ref) == 3):
            pad_ref_arr = None
        else:
            candidate = np.asarray(pad_ref, dtype=float)
            pad_ref_arr = candidate if np.all(np.isfinite(candidate)) else None
        if pad_ref_arr is not None:
            payload["pad_ref_world_xyz_m"] = pad_ref_arr.tolist()
            can_arr = np.asarray(can_world, dtype=float)
            delta_world = can_arr - pad_ref_arr
            payload["can_minus_padref_world_xyz_m"] = delta_world.tolist()
            payload["can_minus_padref_norm_m"] = float(np.linalg.norm(delta_world))
            if base_pose_world is not None:
                delta_base = _world_to_base(can_arr, base_pose_world) - _world_to_base(
                    pad_ref_arr, base_pose_world
                )
                payload["can_minus_padref_base_xyz_m"] = delta_base.tolist()
                payload["can_minus_padref_base_norm_m"] = float(np.linalg.norm(delta_base))
        if not isinstance(gt_midpoint, (list, tuple)) or len(gt_midpoint) != 3:
            return payload
        gt_midpoint_arr = np.asarray(gt_midpoint, dtype=float)
        fk_tip_arr = (
            np.asarray(ee_fk_world, dtype=float)
            if isinstance(ee_fk_world, (list, tuple)) and len(ee_fk_world) == 3
            else None
        )
        if fk_tip_arr is not None:
            fk_to_can = fk_tip_arr - np.asarray(can_world, dtype=float)
            payload["fk_to_can_world_m"] = float(np.linalg.norm(fk_to_can))
            payload["fk_to_can_xy_m"] = float(np.linalg.norm(fk_to_can[:2]))
            offset_vector = gt_midpoint_arr - fk_tip_arr
            payload["finger_midpoint_fk_offset_world_xyz_m"] = offset_vector.tolist()
            payload["fk_gt_offset_m"] = float(np.linalg.norm(offset_vector))
        midpoint_to_can = gt_midpoint_arr - np.asarray(can_world, dtype=float)
        payload["gt_to_can_world_m"] = float(np.linalg.norm(midpoint_to_can))
        payload["gt_to_can_xy_m"] = float(np.linalg.norm(midpoint_to_can[:2]))
        if base_pose_world is not None:
            midpoint_base = _world_to_base(gt_midpoint_arr, base_pose_world)
            can_base = _world_to_base(np.asarray(can_world, dtype=float), base_pose_world)
            payload["finger_midpoint_base_xyz_m"] = midpoint_base.tolist()
            payload["finger_midpoint_to_can_base_xy_m"] = float(
                np.linalg.norm((midpoint_base - can_base)[:2])
            )
        if isinstance(finger_tips, dict):
            can_arr = np.asarray(can_world, dtype=float)
            for finger_name, tip_pos in finger_tips.items():
                if not isinstance(tip_pos, (list, tuple)) or len(tip_pos) != 3:
                    continue
                tip_arr = np.asarray(tip_pos, dtype=float)
                dist_xy = float(np.linalg.norm((tip_arr - can_arr)[:2]))
                clearance = dist_xy - CAN_RADIUS_M
                if "left" in finger_name:
                    payload["clearance_left_m"] = clearance
                elif "right" in finger_name:
                    payload["clearance_right_m"] = clearance
        return payload

    def _trace_gt_tip(
        self,
        backend,
        observation: Observation,
        state: str,
        phase: str,
    ) -> dict[str, object] | None:
        """Log real finger midpoint vs FK tip, XY offset, and per-side clearance to the CAN."""
        if self.trace_path is None:
            return None
        can_world = self._can_pose(backend.get_ground_truth())
        payload: dict[str, object] = {
            "state": state,
            "phase": phase,
            "sim_time_s": observation.t,
            **self._gt_finger_payload(backend, observation, can_world),
        }
        offset_m = payload.get("fk_gt_offset_m")
        tool_offset_error = (
            isinstance(offset_m, (int, float)) and offset_m > TOOL_OFFSET_CALIBRATION_TOLERANCE_M
        )
        payload["tool_offset_error"] = tool_offset_error
        if tool_offset_error:
            assert isinstance(offset_m, (int, float))
            payload["calibration_warning"] = (
                f"TOOL_OFFSET calibration error (|GT midpoint - FK tip| = {offset_m:.4f}m > "
                f"{TOOL_OFFSET_CALIBRATION_TOLERANCE_M:.4f}m)"
            )
            arm7_gt_world = observation.robot.diagnostics.get(
                "arm_7_joint_ground_truth_world_xyz_m"
            )
            gt_midpoint = payload.get("gt_midpoint_world_xyz_m")
            if (
                isinstance(arm7_gt_world, (list, tuple))
                and isinstance(gt_midpoint, (list, tuple))
                and len(arm7_gt_world) == 3
                and len(gt_midpoint) == 3
            ):
                span = float(
                    np.linalg.norm(
                        np.asarray(gt_midpoint, dtype=float)
                        - np.asarray(arm7_gt_world, dtype=float)
                    )
                )
                proposal: dict[str, object] = {
                    "report_only": True,
                    "configured_finger_tip_offset_m": FINGER_TIP_OFFSET_M,
                    "proposed_finger_tip_offset_m": span,
                    "proposal_note": (
                        "Pad-centre distance from the arm_7 endPoint Solid; direction must be "
                        "confirmed against the tool axis. Never applied automatically."
                    ),
                }
                payload["tool_offset_calibration_proposal"] = proposal
        self._trace_event("gt_tip_telemetry", **payload)
        return payload

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
        record.update(self._gt_finger_payload(backend, observation, can_world))
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
        if not reset:
            self._step_count += 1
        drift_m, drift_yaw_deg = self._base_drift(observation)
        self._max_base_drift_m = max(self._max_base_drift_m, drift_m)
        self._max_base_drift_yaw_deg = max(self._max_base_drift_yaw_deg, abs(drift_yaw_deg))
        tilt_roll_deg, tilt_pitch_deg = self._base_tilt_deg(observation)
        self._max_base_tilt_deg = max(
            self._max_base_tilt_deg, abs(tilt_roll_deg), abs(tilt_pitch_deg)
        )
        if not reset and self._step_count == 1 and self.trace_path is not None:
            # Batch diagnostics (addition 3): base state at the first post-reset command,
            # so failures can be attributed to tilt/velocity without re-running.
            diagnostics = observation.robot.diagnostics
            self._trace_event(
                "first_command_state",
                state=state,
                sim_time_s=observation.t,
                base_tilt_roll_deg=tilt_roll_deg,
                base_tilt_pitch_deg=tilt_pitch_deg,
                base_drift_distance_m=drift_m,
                base_drift_yaw_deg=drift_yaw_deg,
                base_velocity_linear_m_s=diagnostics.get("base_velocity_linear_m_s"),
                base_velocity_angular_rad_s=diagnostics.get("base_velocity_angular_rad_s"),
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
        total_wall_s = sum(totals.values())
        wall_per_step = (total_wall_s / self._step_count) if self._step_count > 0 else None
        return {
            "enabled": self.timing_enabled,
            "total_wall_s": total_wall_s,
            "totals": totals,
            "by_state": self.state_timings,
            "steps": self._step_count,
            "wall_per_step_s": wall_per_step,
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
    transforms = kinematics.chain.forward_kinematics(list(q), full_kinematics=True)
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
