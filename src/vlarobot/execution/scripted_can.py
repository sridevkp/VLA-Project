"""Brain-side scripted CAN pick-and-place policy used for M1 regression checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from vlarobot.backends.webots.config import (
    APPROACH_H,
    ARM_JOINTS,
    GRASP_Z,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
    PLACE_CLEAR,
    PLACE_WORLD,
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
        self, kinematics: Kinematics | None = None, trace_path: Path | None = None
    ) -> None:
        self.kinematics = kinematics or Kinematics()
        self.trace_path = trace_path
        self._seed = 0
        self._last_failure_reason: str | None = None
        self._last_failure_is_safety = False

    def execute(self, backend, seed: int) -> ScriptedCanResult:
        self.kinematics.set_seed(seed)
        self._seed = seed
        self._last_failure_reason = None
        self._last_failure_is_safety = False
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        observation = backend.reset(ScenarioSpec(), seed)
        current_q = self._full_joint_vector(observation.robot.joint_pos)
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
            trace_target = self.kinematics.forward(home_q)[:3, 3] if name == "home" else target
            self._trace_boundary(backend, observation, state, "enter", trace_target)
            if name == "home":
                movement = self._move_to_joint_target(backend, current_q, home_q, state)
            elif target is not None:
                movement = self._move_to(backend, current_q, target, state)
            else:
                raise RuntimeError(f"Regression waypoint {name!r} is missing its target.")
            if movement is None:
                rejections += int(self._last_failure_is_safety)
                final_distance = self._placement_distance(backend, observation)
                self._trace_boundary(
                    backend,
                    observation,
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
            final_state = name
            self._trace_boundary(backend, observation, state, "exit", trace_target)

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
        for waypoint in self.kinematics.interpolate_joint_path(
            current_q, q, MAX_JOINT_STEP_RAD, min_steps=2
        ):
            joint_target = {name: float(waypoint[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            response = backend.apply(JointTarget(joint_target))
            if not response.accepted:
                self._last_failure_is_safety = True
                self._last_failure_reason = response.reason or "joint target rejected"
                self._trace_event(
                    "safety_rejection",
                    state=state,
                    target_base_xyz_m=target.tolist(),
                    joint_target=joint_target,
                    reason=self._last_failure_reason,
                    ik_error_m=ik_error,
                    ik_tilt_deg=ik_tilt,
                )
                return None
            observation = backend.step(1)
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
        return self._full_joint_vector(observation.robot.joint_pos), observation

    def _move_to_joint_target(
        self, backend, current_q: np.ndarray, target_q: np.ndarray, state: str
    ) -> tuple[np.ndarray, Observation] | None:
        self._last_failure_is_safety = False
        target_position = self.kinematics.forward(target_q)[:3, 3]
        observation = None
        for waypoint in self.kinematics.interpolate_joint_path(
            current_q, target_q, MAX_JOINT_STEP_RAD, min_steps=2
        ):
            joint_target = {name: float(waypoint[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            response = backend.apply(JointTarget(joint_target))
            if not response.accepted:
                self._last_failure_is_safety = True
                self._last_failure_reason = response.reason or "home joint target rejected"
                self._trace_event(
                    "safety_rejection",
                    state=state,
                    target_base_xyz_m=target_position.tolist(),
                    joint_target=joint_target,
                    reason=self._last_failure_reason,
                )
                return None
            observation = backend.step(1)
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
        return self._full_joint_vector(observation.robot.joint_pos), observation

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
            "achieved_ee_base_xyz_m": observation.robot.ee_pose.position.as_array().tolist(),
            "joint_positions_rad": observation.robot.joint_pos,
            "gripper_opening_m": observation.robot.gripper_opening,
            "grasp_detected": observation.robot.gripper_holding,
            "failed": failed,
        }
        can_world = self._can_pose(backend.get_ground_truth())
        record["can_world_xyz_m"] = can_world.tolist()
        base_pose_world = observation.robot.base_pose_world
        if base_pose_world is not None:
            record["can_base_xyz_m"] = _world_to_base(can_world, base_pose_world).tolist()
            record["base_world_xyz_m"] = base_pose_world.position.as_array().tolist()
            record["base_world_orientation_xyzw"] = list(base_pose_world.orientation)
        record.update(details)
        self._write_trace(record)

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


def _ik_tolerance_for_state(state: str) -> float:
    if state == "PREGRASP":
        return IK_TOL
    if state in {"TRANSIT", "RETREAT", "HOME"}:
        return IK_TOL_TRANSIT
    return IK_TOL_GRASP


def _vec3_tuple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])
