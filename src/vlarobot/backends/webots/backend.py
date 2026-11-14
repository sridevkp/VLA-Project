"""RobotBackend adapter for a running Webots TIAGo world."""

from __future__ import annotations

import logging
import math
import os
from time import perf_counter
from typing import Any, cast

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    CAN_DEF,
    FINGER_TIP_OFFSET_M,
    FK_GROUND_TRUTH_TOLERANCE_M,
    HEAD_JOINTS,
    HEAD_PAN,
    HEAD_TILT,
    OBSERVE_IMAGES,
    PAD_OFFSET_ARM7_SIMFRAME_M,
    PROBE_MAX_STEPS_PER_JOINT,
    PROBE_STATIONARITY_STEPS,
    PROBE_STATIONARY_DELTA_RAD,
    PROBE_TOTAL_STEP_CAP,
    RESET_HEAD_SETTLE_MS,
    RESET_POST_SETTLE_ANG_VELOCITY_THRESHOLD_RAD_S,
    RESET_POST_SETTLE_CONSECUTIVE_STEPS,
    RESET_POST_SETTLE_ENABLED,
    RESET_POST_SETTLE_MAX_STEPS,
    RESET_POST_SETTLE_TILT_DELTA_DEG,
    RESET_POST_SETTLE_VELOCITY_THRESHOLD_M_S,
    RESET_TORSO_SETTLE_TOLERANCE_M,
    RESET_TORSO_TIMEOUT_STEPS,
    SAFETY_PROFILE,
    SETTINGS,
    TORSO_JOINT,
)
from vlarobot.backends.webots.frames import Frames, compose_can_reset_pose
from vlarobot.backends.webots.gripper import Gripper
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.robot_io import RobotIO
from vlarobot.backends.webots.supervisor_kinematics import (
    _capture_joint_entry,
    child_solid,
    joint_contexts,
    parent_solid,
    pin_base_pose,
    posture_reference,
    read_joint_node,
    sim_time_s,
    solid_pose,
    unpin_base_pose,
    webots_version,
)
from vlarobot.backends.webots.visualization import Visualization
from vlarobot.core.types import (
    Action,
    ApplyResult,
    BaseGoal,
    CameraInfo,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    GroundTruth,
    JointTarget,
    Observation,
    Pose2D,
    Pose3D,
    RobotState,
    ScenarioSpec,
    Vec3,
)
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor

LOGGER = logging.getLogger(__name__)


class WebotsBackend:
    """Adapt the existing TIAGo controller devices to the core backend interface."""

    def __init__(self) -> None:
        self.io = RobotIO()
        can_node = self.io.initialize_can(CAN_DEF)
        if can_node is None:
            raise RuntimeError(f"No node with DEF {CAN_DEF} in the world.")
        self.can_node = can_node
        self.io.enable_contact_tracking()
        self.kinematics = Kinematics()
        self.frames = Frames(self.io, self.kinematics)
        self.visualization = Visualization(self.io, self.frames)
        self.gripper = Gripper(self.io, self.visualization)
        self.camera = self.io.robot.getDevice("Astra rgb")
        if self.camera is None:
            raise RuntimeError("Required Webots camera not found: Astra rgb")
        self.camera.enable(self.io.ts)
        LOGGER.info("Webots RGB camera noise field: %s", self.io.camera_noise_value(self.camera))
        self._camera_width = int(self.camera.getWidth())
        self._camera_height = int(self.camera.getHeight())
        self._camera_fov = float(self.camera.getFov())
        initial_positions = self.io.joint_positions_from_fields()
        self._initial_joint_pos = {
            name: initial_positions[name] for name in ARM_JOINTS + [TORSO_JOINT] + HEAD_JOINTS
        }
        self._initial_finger_positions = [
            initial_positions[motor.getName()] for motor, _ in self.io.fingers
        ]
        self._initial_can_translation = self.can_node.getField("translation").getSFVec3f()
        self._initial_can_rotation = self.can_node.getField("rotation").getSFRotation()
        self._initial_base_translation = self.io.me.getField("translation").getSFVec3f()
        self._initial_base_rotation = self.io.me.getField("rotation").getSFRotation()
        self._reset_base_position = np.asarray(self._initial_base_translation, dtype=float)
        self._reset_base_yaw = 0.0
        self._last_joint_pos = self._joint_positions()
        self._last_time = float(self.io.robot.getTime())
        self._episode_start_time = self._last_time
        self._safety = SafetyMonitor(self._safety_limits())
        self._stopped = False
        self._reset_fingerprint_pending: dict[str, object] | None = None
        self._sim_kinematics_snapshot_pending: dict[str, object] | None = None
        self._physics_step_count = 0
        self._first_command_step_count: int | None = None
        self._export_job: dict[str, object] | None = None
        self._arm7_endpoint_offset_world: np.ndarray | None = None
        self._arm7_reset_link_rotation: np.ndarray | None = None

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        if self._stopped:
            raise RuntimeError("Cannot reset a stopped backend.")
        if SETTINGS.reset_mode != "soft":
            raise NotImplementedError(
                "reset_mode='reload' requires restarting Webots and reconnecting the simulator "
                "transport; only 'soft' reset is supported in the current server lifecycle."
            )
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        torso_before_rad = float(self.io.sensors[TORSO_JOINT].getValue())
        positions = dict(self._initial_joint_pos)
        positions[TORSO_JOINT] = float(SETTINGS.torso_height or 0.0)
        if all(name in self.io.motors for name in HEAD_JOINTS):
            positions[HEAD_JOINTS[0]] = HEAD_PAN
            positions[HEAD_JOINTS[1]] = HEAD_TILT
        positions.update(
            {
                motor.getName(): position
                for (motor, _), position in zip(
                    self.io.fingers, self._initial_finger_positions, strict=True
                )
            }
        )
        self.io.set_joint_positions(positions)
        can_translation = self._initial_can_translation
        can_rotation = self._initial_can_rotation
        if scenario.can_xy_yaw_offset is not None:
            # Scene perturbation for M1-07(b)/M2 ScenarioGenerator (ADR-0025): move/rotate
            # the CAN relative to the world-file pose at every reset.
            can_translation, can_rotation = compose_can_reset_pose(
                can_translation, can_rotation, scenario.can_xy_yaw_offset
            )
        self.can_node.getField("translation").setSFVec3f(can_translation)
        self.can_node.getField("rotation").setSFRotation(can_rotation)
        self.io.me.getField("translation").setSFVec3f(self._initial_base_translation)
        self.io.me.getField("rotation").setSFRotation(self._initial_base_rotation)
        self._apply_scenario_objects(scenario)
        self.io.robot.simulationResetPhysics()
        can_node = self.io.robot.getFromDef(CAN_DEF)
        if can_node is None:
            raise RuntimeError(f"No node with DEF {CAN_DEF} after resetting physics.")
        self.can_node = can_node
        self.io.can_node = can_node
        # Ensure CAN and tracked obstacles have no residual velocities after global reset.
        for node in (self.can_node,):
            if node is not None and hasattr(node, "resetPhysics"):
                try:
                    node.resetPhysics()
                except Exception:
                    pass
        for def_name in tuple(getattr(self, "_initial_object_poses", ())):
            obj = self.io.robot.getFromDef(def_name)
            if obj is not None and hasattr(obj, "resetPhysics"):
                try:
                    obj.resetPhysics()
                except Exception:
                    pass
        if self.io.robot.step(self.io.ts) == -1:
            raise RuntimeError("Webots simulation stopped while restoring the reset pose.")
        self.io.apply_hold_base()
        if all(name in self.io.motors for name in HEAD_JOINTS):
            self._settle_reset_posture()
        # Re-assert after the reset settle so the hold survives any posture motion.
        self.io.apply_hold_base()
        reset_position = self.io.me.getPosition()
        reset_rotation = np.asarray(self.io.me.getOrientation()).reshape((3, 3))
        self._reset_base_position = np.asarray(reset_position, dtype=float)
        self._reset_base_yaw = math.atan2(reset_rotation[1, 0], reset_rotation[0, 0])
        self.gripper.last_close_stalled = None
        self.gripper.last_close_opening = None
        self._episode_start_time = float(self.io.robot.getTime())
        self._last_joint_pos = self._joint_positions()
        self._last_time = self._episode_start_time
        self._physics_step_count = 0
        self._first_command_step_count = None
        torso_after_rad = float(self.io.sensors[TORSO_JOINT].getValue())
        settle_info = None
        if RESET_POST_SETTLE_ENABLED:
            settle_info = self._settle_post_reset()
        fingerprint_lookup = getattr(self.io, "reset_fingerprint", None)
        fingerprint = fingerprint_lookup() if fingerprint_lookup is not None else None
        if isinstance(fingerprint, dict):
            fingerprint = dict(fingerprint)
            fingerprint["torso_sensor_after_rad"] = torso_after_rad
            fingerprint["torso_commanded_rad"] = float(SETTINGS.torso_height or 0.0)
            if settle_info is not None:
                fingerprint["post_reset_settle"] = settle_info
            fingerprint["physics_step_count_at_reset"] = self._physics_step_count
        self._reset_fingerprint_pending = fingerprint
        self._reset_torso_before_rad = torso_before_rad
        self._reset_torso_after_rad = torso_after_rad
        endpoint_offset = self._arm7_endpoint_offset()
        if endpoint_offset is None:
            self._arm7_endpoint_offset_world = None
            self._arm7_reset_link_rotation = None
        else:
            self._arm7_endpoint_offset_world, self._arm7_reset_link_rotation = endpoint_offset
        # Re-assert scenario object poses after reset settling: posture/hold steps can
        # perturb physics objects in a reused session, so restore and clear dynamics.
        self._apply_scenario_objects(scenario)
        for def_name in tuple(getattr(self, "_initial_object_poses", ())):
            obj = self.io.robot.getFromDef(def_name)
            if obj is not None and hasattr(obj, "resetPhysics"):
                try:
                    obj.resetPhysics()
                except Exception:
                    pass
        if hasattr(self.can_node, "resetPhysics"):
            try:
                self.can_node.resetPhysics()
            except Exception:
                pass
        if self.io.robot.step(self.io.ts) == -1:
            raise RuntimeError("Webots simulation stopped while finalizing reset object poses.")
        # ADR-0020 addendum: reset stays fast (< 5 s) and performs no probes.
        # Kinematics export is served on demand via the dedicated export_kinematics
        # request (chunked polling) so probing cannot block a normal reset.
        self._export_job: dict[str, object] | None = None
        return self._observation()

    def step(self, n_steps: int = 1) -> Observation:
        if self._stopped:
            raise RuntimeError("Cannot step a stopped backend.")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        for _ in range(n_steps):
            if self.visualization.step() == -1:
                raise RuntimeError("Webots simulation stopped while stepping.")
            self._physics_step_count += 1
        return self._observation()

    def apply(self, action: Action) -> ApplyResult:
        if self._stopped:
            raise RuntimeError("Cannot apply an action to a stopped backend.")
        if self._first_command_step_count is None:
            self._first_command_step_count = self._physics_step_count
        state = self._robot_state()
        validation = self._safety.validate(action, state)
        if not validation.accepted:
            return validation

        if isinstance(action, JointTarget):
            for name, target in action.q.items():
                self.io.motors[name].setPosition(target)
            return ApplyResult(True)
        if isinstance(action, (EEPoseTarget, EEDelta)):
            if isinstance(action, EEPoseTarget):
                position = action.pose.position.as_array()
                target_orientation = action.pose.orientation
                gripper = action.gripper
            else:
                if any(value != 0.0 for value in (action.drot.x, action.drot.y, action.drot.z)):
                    return ApplyResult(
                        False, "Webots adapter currently supports translation-only deltas."
                    )
                current = state.ee_pose.position
                position = np.asarray(
                    (
                        current.x + action.dpos.x,
                        current.y + action.dpos.y,
                        current.z + action.dpos.z,
                    )
                )
                target_orientation = state.ee_pose.orientation
                gripper = action.gripper

            joints, error, _ = self.kinematics.solve(position, self.frames.current_q())
            if error > 0.015:
                return ApplyResult(False, "end-effector target is outside the IK tolerance")
            orientation = _rotation_to_quaternion(self.kinematics.forward(joints)[:3, :3])
            if _quaternion_distance(orientation, target_orientation) > math.radians(5):
                return ApplyResult(False, "end-effector orientation is outside the supported range")
            target_joints = {name: float(joints[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            joint_validation = self._safety.validate(JointTarget(target_joints), state)
            if not joint_validation.accepted:
                return joint_validation
            for name in ARM_JOINTS:
                self.io.motors[name].setPosition(float(joints[self.kinematics.idx[name]]))
            if gripper is not None:
                self._set_gripper(gripper)
            return ApplyResult(True)
        if isinstance(action, GripperCmd):
            self._set_gripper(action.opening, action.force)
            return ApplyResult(True)
        if isinstance(action, BaseGoal):
            return ApplyResult(False, "TIAGo base navigation is not implemented in M1.")
        return ApplyResult(False, f"unsupported action type: {type(action).__name__}")

    def _apply_scenario_objects(self, scenario: ScenarioSpec) -> None:
        """Place scenario obstacles at reset; restore nodes a later scenario omits (ADR-0029).

        Each entry must be a mapping with an existing ``def``, an absolute world ``xyz``,
        and an optional ``yaw_offset`` composed onto the world-file rotation (the same
        world-Z convention as ``can_xy_yaw_offset``). The world-file pose of every touched
        node is cached on first contact, so a subsequent default reset restores it — the
        CAN/CAN-less path above is unaffected when ``scenario.objects`` is empty and no
        node has ever been placed. Spawning new nodes is not supported.
        """
        placements: dict[str, tuple[list[float], float]] = {}
        for entry in scenario.objects:
            if not isinstance(entry, dict) or "def" not in entry:
                raise NotImplementedError(
                    "scenario objects must reference existing DEF nodes via a 'def' key; "
                    "spawning new nodes is not supported."
                )
            def_name = str(entry["def"])
            if def_name == CAN_DEF:
                raise ValueError(
                    f"scenario objects must not place {CAN_DEF!r}; "
                    "use ScenarioSpec.can_xy_yaw_offset for the CAN."
                )
            xyz = entry.get("xyz")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise ValueError(f"scenario object {def_name!r} needs 'xyz' with 3 world values.")
            placements[def_name] = (
                [float(value) for value in xyz],
                float(entry.get("yaw_offset", 0.0)),
            )

        cache = getattr(self, "_initial_object_poses", None)
        if cache is None:
            cache = {}
            self._initial_object_poses = cache

        def node_for(def_name: str) -> Any:
            node = self.io.robot.getFromDef(def_name)
            if node is None:
                raise RuntimeError(f"No node with DEF {def_name} for scenario object placement.")
            return node

        # Restore nodes placed by an earlier scenario but absent from this one.
        for def_name in tuple(cache):
            if def_name in placements:
                continue
            node = node_for(def_name)
            initial_translation, initial_rotation = cache[def_name]
            node.getField("translation").setSFVec3f(list(initial_translation))
            node.getField("rotation").setSFRotation(list(initial_rotation))

        for def_name, (xyz, yaw_offset) in placements.items():
            node = node_for(def_name)
            if def_name not in cache:
                cache[def_name] = (
                    [float(value) for value in node.getField("translation").getSFVec3f()],
                    [float(value) for value in node.getField("rotation").getSFRotation()],
                )
            initial_translation, initial_rotation = cache[def_name]
            node.getField("translation").setSFVec3f(xyz)
            if yaw_offset != 0.0:
                _, composed = compose_can_reset_pose(
                    initial_translation, initial_rotation, (0.0, 0.0, yaw_offset)
                )
            else:
                composed = list(initial_rotation)
            node.getField("rotation").setSFRotation(composed)
            # Clear residual dynamics before the global physics reset settles contacts.
            if hasattr(node, "resetPhysics"):
                try:
                    node.resetPhysics()
                except Exception:
                    pass

    def get_ground_truth(self) -> GroundTruth:
        position = self.can_node.getPosition()
        rotation = np.asarray(self.can_node.getOrientation()).reshape((3, 3))
        pose = Pose3D(
            Vec3(float(position[0]), float(position[1]), float(position[2])),
            _rotation_to_quaternion(rotation),
        )
        object_poses = {CAN_DEF: pose}
        # Obstacles placed by a scenario are tracked for the displacement/tilt collision
        # proxy (ADR-0029): their poses are read here so eval can diff reset vs end-of-run.
        for def_name in tuple(getattr(self, "_initial_object_poses", ())):
            node = self.io.robot.getFromDef(def_name)
            if node is None:
                continue
            obstacle_position = node.getPosition()
            obstacle_rotation = np.asarray(node.getOrientation()).reshape((3, 3))
            object_poses[def_name] = Pose3D(
                Vec3(
                    float(obstacle_position[0]),
                    float(obstacle_position[1]),
                    float(obstacle_position[2]),
                ),
                _rotation_to_quaternion(obstacle_rotation),
            )
        return GroundTruth(object_poses=object_poses)

    def stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            self.io.robot.simulationQuit(0)

    # ------------------------------------------------------------------
    # ADR-0020 addendum: on-demand, chunked kinematics export (no reset probes).
    # ------------------------------------------------------------------

    def _probe_settings(self) -> dict[str, float | int]:
        return {
            "stationary_delta": float(PROBE_STATIONARY_DELTA_RAD),
            "stationarity_steps": int(PROBE_STATIONARITY_STEPS),
            "max_steps_per_joint": int(PROBE_MAX_STEPS_PER_JOINT),
        }

    def export_kinematics(self, action: str = "poll") -> dict[str, object]:
        """Chunked export: ``start`` resets the job, ``poll`` probes one joint step.

        Each ``poll`` advances by probing at most the next joint (or finalizing),
        returning progress/heartbeat status and, when complete, the snapshot.
        This keeps reset fast and lets clients use per-poll timeouts.
        """
        if action == "start":
            pin_base_pose(self.io)
            self._export_job = self._new_export_job()
            return self._export_status(cast(dict[str, Any], self._export_job))
        if action == "cancel":
            self._export_job = None
            unpin_base_pose(self.io)
            return {"state": "cancelled", "done": True, "snapshot": None, "progress": {}}
        if action not in {"poll", "status"}:
            raise ValueError(f"Unknown export_kinematics action: {action!r}")
        job = self._export_job
        if job is None:
            pin_base_pose(self.io)
            job = self._new_export_job()
            self._export_job = job
        job_typed = cast(dict[str, Any], job)
        if not bool(job_typed.get("done")) and action == "poll":
            self._advance_export_job(job_typed)
        return self._export_status(job_typed)

    def _new_export_job(self) -> dict[str, object]:
        joint_order = (TORSO_JOINT, *ARM_JOINTS)
        contexts = list(joint_contexts(self.io, joint_order))
        return {
            "joint_order": joint_order,
            "contexts": contexts,
            "index": 0,
            "joints": [],
            "steps_used_total": 0,
            "done": False,
            "error": None,
            "snapshot": None,
            "per_joint": [],
            "probe_settings": self._probe_settings(),
        }

    def _advance_export_job(self, job: dict[str, Any]) -> None:
        contexts = cast(list[Any], job.get("contexts") or [])
        index = int(job.get("index", 0) or 0)
        if index >= len(contexts):
            job["snapshot"] = self._finalize_export_snapshot(job)
            job["done"] = True
            self._export_job = job
            unpin_base_pose(self.io)
            return
        if int(job.get("steps_used_total", 0) or 0) >= int(PROBE_TOTAL_STEP_CAP):
            job["error"] = "total_probe_step_cap_reached"
            job["snapshot"] = self._finalize_export_snapshot(job)
            job["done"] = True
            unpin_base_pose(self.io)
            return
        name, kind, motor = contexts[index]
        entry = _capture_joint_entry(
            self.io,
            name,
            kind,
            motor,
            probe_axes=True,
            probe_settings=cast(dict[str, float | int], job.get("probe_settings") or {}),
        )
        cast(list[dict[str, object]], job.setdefault("joints", [])).append(entry)
        status_obj = entry.get("probe_status") or {}
        status = status_obj if isinstance(status_obj, dict) else {}
        steps = int(status.get("steps_used", 0) or 0)
        job["steps_used_total"] = int(job.get("steps_used_total", 0) or 0) + steps
        cast(list[dict[str, object]], job.setdefault("per_joint", [])).append(
            {
                "name": name,
                "kind": kind,
                "steps_used": steps,
                "measured_angle_moved_rad": status.get("measured_angle_moved_rad"),
                "reason": status.get("reason"),
                "settled": status.get("settled"),
                "axis_source": entry.get("axis_source"),
            }
        )
        job["index"] = index + 1
        if int(job.get("index", 0) or 0) >= len(contexts):
            job["snapshot"] = self._finalize_export_snapshot(job)
            job["done"] = True
            unpin_base_pose(self.io)

    def _finalize_export_snapshot(self, job: dict[str, Any]) -> dict[str, object]:
        joints = list(cast(list[Any], job.get("joints") or []))
        # Enforce parent-chain consistency for arm joints: arm_i parent == arm_{i-1} child.
        previous_child: dict[str, object] | None = None
        for idx, entry in enumerate(joints):
            if not isinstance(entry, dict):
                continue
            child = entry.get("child")
            if idx >= 2 and previous_child is not None and isinstance(entry.get("parent"), dict):
                entry["parent"] = previous_child
            if isinstance(child, dict):
                previous_child = child
        joint_order = tuple(cast(Any, job.get("joint_order") or (TORSO_JOINT, *ARM_JOINTS)))
        readable: list[str] = []
        pose_inferred: list[str] = []
        unreadable: list[str] = []
        for entry in joints:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("axis_source"))
            name = str(entry.get("name"))
            if source == "posture_probe":
                pose_inferred.append(name)
                if entry.get("readable_axis") is not None:
                    readable.append(name)
            else:
                unreadable.append(name)
        fingers = {
            motor.getName(): child_solid(self.io, motor.getName()).to_payload()
            for motor, _sensor in getattr(self.io, "fingers", [])
        }
        snapshot: dict[str, object] = {
            "schema_version": 1,
            "units": {"length": "m", "angle": "rad"},
            "source": "Webots Supervisor, Pose-derived Solid world poses",
            "webots_version": webots_version(self.io),
            "sim_time_s": sim_time_s(self.io),
            "timestamp_utc": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "base_frame": solid_pose(self.io.me, "RobotNode").to_payload(),
            "joints": joints,
            "finger_endpoint_solids": fingers,
            "readability_report": {
                "jointParameters": readable,
                "posture_probe": pose_inferred,
                "unreadable": unreadable,
            },
            "posture_reference": posture_reference(self.io, cast(tuple[str, ...], joint_order)),
            "export_progress": {
                "steps_used_total": job.get("steps_used_total", 0),
                "total_step_cap": PROBE_TOTAL_STEP_CAP,
                "per_joint": job.get("per_joint", []),
                "error": job.get("error"),
            },
        }
        return snapshot

    def _export_status(self, job: dict[str, Any]) -> dict[str, object]:
        contexts = cast(list[Any], job.get("contexts") or [])
        index = int(job.get("index", 0) or 0)
        total = len(contexts)
        done = bool(job.get("done"))
        return {
            "state": "done" if done else "in_progress",
            "done": done,
            "current_index": index,
            "total_joints": total,
            "steps_used_total": job.get("steps_used_total", 0),
            "total_step_cap": PROBE_TOTAL_STEP_CAP,
            "per_joint": list(cast(list[Any], job.get("per_joint") or [])),
            "error": job.get("error"),
            "snapshot": job.get("snapshot") if done else None,
        }

    def _observation(self) -> Observation:
        width = int(getattr(self, "_camera_width", getattr(self.camera, "getWidth", lambda: 64)()))
        height = int(
            getattr(self, "_camera_height", getattr(self.camera, "getHeight", lambda: 48)())
        )
        fov = float(getattr(self, "_camera_fov", getattr(self.camera, "getFov", lambda: 1.0)()))
        if OBSERVE_IMAGES:
            image = np.asarray(self.camera.getImageArray(), dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] < 3:
                raise RuntimeError(f"Webots camera returned an invalid image shape: {image.shape}.")
            image = image[:, :, :3].transpose((1, 0, 2)).copy()
            width, height = image.shape[1], image.shape[0]
            fov = float(getattr(self.camera, "getFov", lambda: fov)())
        else:
            # bench-steps: image reads can be disabled without changing observation shape.
            image = np.zeros((height, width, 3), dtype=np.uint8)
        focal = width / (2.0 * math.tan(fov / 2.0)) if fov else 0.0
        camera_info = CameraInfo(
            K=(
                (focal, 0.0, width / 2.0),
                (0.0, focal, height / 2.0),
                (0.0, 0.0, 1.0),
            ),
            T_base_cam=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            width=width,
            height=height,
        )
        return Observation(
            t=float(self.io.robot.getTime()) - self._episode_start_time,
            rgb=image,
            depth=None,
            cam_info=camera_info,
            robot=self._robot_state(),
        )

    def _robot_state(self) -> RobotState:
        joint_pos = self._joint_positions()
        sim_time = float(self.io.robot.getTime())
        elapsed = sim_time - self._last_time
        joint_vel = {
            name: (position - self._last_joint_pos[name]) / elapsed if elapsed > 0 else 0.0
            for name, position in joint_pos.items()
        }
        self._last_joint_pos = joint_pos
        self._last_time = sim_time

        timing_enabled = _backend_timings_enabled()
        diagnostics_started = perf_counter() if timing_enabled else None
        base_position = self.io.me.getPosition()
        base_rotation = np.asarray(self.io.me.getOrientation()).reshape((3, 3))
        base_yaw = math.atan2(base_rotation[1, 0], base_rotation[0, 0])
        end_effector = self.kinematics.forward(self.frames.current_q())
        base_position_array = np.asarray(base_position, dtype=float)
        base_drift = base_position_array - self._reset_base_position
        base_yaw_delta = _wrap_angle(base_yaw - self._reset_base_yaw)
        full_q = self.frames.current_q()
        arm7_pose = self._arm7_fk_world_pose(full_q, base_position_array, base_rotation)
        arm7_world = None if arm7_pose is None else arm7_pose[0]
        arm7_link_rotation = None if arm7_pose is None else arm7_pose[1]
        endpoint_lookup = getattr(self.io, "joint_endpoint_world_position", None)
        arm7_endpoint = endpoint_lookup("arm_7_joint") if endpoint_lookup else None
        orientation_lookup = getattr(self.io, "joint_endpoint_world_orientation", None)
        arm7_orientation = orientation_lookup("arm_7_joint") if orientation_lookup else None
        # MEASURED pad reference (ADR-0022): arm_7 Solid pose composed with the measured
        # constant offset in the sim arm_7 Solid frame. This is the gt_pad servo reference.
        pad_midpoint_measured = None
        if arm7_endpoint is not None and arm7_orientation is not None:
            pad_midpoint_measured = (
                np.asarray(arm7_endpoint, dtype=float)
                + np.asarray(arm7_orientation, dtype=float)
                @ np.asarray(PAD_OFFSET_ARM7_SIMFRAME_M, dtype=float)
            ).tolist()
        # Base speed norms at this observation (batch diagnostics: addition 3).
        base_lin_speed = None
        base_ang_speed = None
        velocity_lookup = getattr(self.io.me, "getVelocity", None)
        if velocity_lookup is not None:
            raw_velocity = velocity_lookup()
            if isinstance(raw_velocity, (list, tuple)) and len(raw_velocity) >= 6:
                velocity_array = np.asarray(raw_velocity, dtype=float)
                if np.all(np.isfinite(velocity_array[:6])):
                    base_lin_speed = float(np.linalg.norm(velocity_array[:3]))
                    base_ang_speed = float(np.linalg.norm(velocity_array[3:6]))
        arm7_offset = (
            None
            if arm7_endpoint is None or arm7_world is None
            else np.asarray(arm7_endpoint, dtype=float) - arm7_world
        )
        reset_offset = getattr(self, "_arm7_endpoint_offset_world", None)
        reset_rotation = getattr(self, "_arm7_reset_link_rotation", None)
        if (
            arm7_offset is None
            or reset_offset is None
            or arm7_link_rotation is None
            or reset_rotation is None
        ):
            arm7_offset_change = None
        else:
            predicted_offset = (arm7_link_rotation @ reset_rotation.T) @ reset_offset
            arm7_offset_change = arm7_offset - predicted_offset
        fk_error_m = (
            None if arm7_offset_change is None else float(np.linalg.norm(arm7_offset_change))
        )
        fk_mismatch = None if fk_error_m is None else fk_error_m > FK_GROUND_TRUTH_TOLERANCE_M
        ee_fk_world = base_rotation @ end_effector[:3, 3] + base_position_array
        finger_lookup = getattr(self.io, "finger_tip_world_positions", None)
        finger_tips = finger_lookup(FINGER_TIP_OFFSET_M) if finger_lookup is not None else None
        finger_midpoint = None
        finger_offset = None
        if isinstance(finger_tips, dict) and len(finger_tips) == 2:
            finger_midpoint = np.mean(np.asarray(list(finger_tips.values()), dtype=float), axis=0)
            finger_offset = finger_midpoint - ee_fk_world
        arm_links: dict[str, object] = {
            "elbow": endpoint_lookup("arm_4_joint") if endpoint_lookup else None,
            "wrist": endpoint_lookup("arm_6_joint") if endpoint_lookup else None,
            "tool": arm7_endpoint,
        }
        contact_lookup = getattr(self.io, "contact_nodes", None)
        contacts = contact_lookup() if contact_lookup else []
        wheel_lookup = getattr(self.io, "wheel_status", None)
        diagnostics: dict[str, object] = {
            "physics_step_count": getattr(self, "_physics_step_count", 0),
            "first_command_step_count": getattr(self, "_first_command_step_count", None),
            "torso_lift_joint_sensor_rad": float(self.io.sensors[TORSO_JOINT].getValue()),
            "base_drift_world_xyz_m": base_drift.tolist(),
            "base_drift_distance_m": float(np.linalg.norm(base_drift[:2])),
            "base_drift_yaw_rad": base_yaw_delta,
            "base_drift_yaw_deg": math.degrees(base_yaw_delta),
            "wheel_motors": wheel_lookup() if wheel_lookup else {},
            "contact_nodes": contacts,
            "arm_link_world_xyz_m": arm_links,
            "arm_7_joint_fk_world_xyz_m": arm7_world.tolist() if arm7_world is not None else None,
            "arm_7_joint_ground_truth_world_xyz_m": arm7_endpoint,
            "arm_7_joint_ground_truth_source": "endPointSolid",
            "arm_7_joint_ground_truth_orientation_rows": arm7_orientation,
            "pad_midpoint_arm7_solid_measured_world_xyz_m": pad_midpoint_measured,
            "pad_offset_arm7_simframe_m": list(PAD_OFFSET_ARM7_SIMFRAME_M),
            "base_velocity_linear_m_s": base_lin_speed,
            "base_velocity_angular_rad_s": base_ang_speed,
            "arm_7_joint_endpoint_offset_world_xyz_m": (
                arm7_offset.tolist() if arm7_offset is not None else None
            ),
            "arm_7_joint_endpoint_reset_offset_world_xyz_m": (
                reset_offset.tolist() if reset_offset is not None else None
            ),
            "arm_7_joint_endpoint_offset_change_world_xyz_m": (
                arm7_offset_change.tolist() if arm7_offset_change is not None else None
            ),
            "arm_7_joint_fk_ground_truth_error_m": fk_error_m,
            "arm_7_joint_fk_ground_truth_mismatch": fk_mismatch,
            "arm_7_joint_endpoint_fk_validation_available": fk_error_m is not None,
            "ee_fk_world_xyz_m": ee_fk_world.tolist(),
            "finger_tip_world_xyz_m": finger_tips,
            "finger_midpoint_world_xyz_m": (
                finger_midpoint.tolist() if finger_midpoint is not None else None
            ),
            "finger_midpoint_source": "endPointSolid",
            "finger_midpoint_fk_offset_world_xyz_m": (
                finger_offset.tolist() if finger_offset is not None else None
            ),
            "finger_midpoint_fk_offset_m": (
                float(np.linalg.norm(finger_offset)) if finger_offset is not None else None
            ),
        }
        pending_fingerprint = getattr(self, "_reset_fingerprint_pending", None)
        if pending_fingerprint is not None:
            diagnostics["reset_fingerprint"] = pending_fingerprint
            diagnostics["reset_torso_before_rad"] = getattr(self, "_reset_torso_before_rad", None)
            diagnostics["reset_torso_after_rad"] = getattr(self, "_reset_torso_after_rad", None)
            self._reset_fingerprint_pending = None
        # Attach live GT Solid telemetry for the validation sweep when the
        # kinematics export/telemetry path is enabled (reads poses only, no probes).
        if SETTINGS.export_sim_kinematics:
            try:
                diagnostics["kinematics_telemetry"] = self._kinematics_telemetry()
            except Exception:
                diagnostics["kinematics_telemetry"] = {"error": "unavailable"}
        if diagnostics_started is not None:
            diagnostics["backend_diagnostics_compute_s"] = perf_counter() - diagnostics_started
        opening = self.gripper.opening()
        return RobotState(
            joint_pos={
                **joint_pos,
                TORSO_JOINT: float(self.io.sensors[TORSO_JOINT].getValue()),
            },
            joint_vel=joint_vel,
            base_pose=Pose2D(float(base_position[0]), float(base_position[1]), base_yaw),
            ee_pose=Pose3D(
                Vec3(*map(float, end_effector[:3, 3])),
                _rotation_to_quaternion(end_effector[:3, :3]),
            ),
            gripper_opening=opening,
            base_pose_world=Pose3D(
                Vec3(*map(float, base_position)),
                _rotation_to_quaternion(base_rotation),
            ),
            diagnostics=diagnostics,
        )

    def _joint_positions(self) -> dict[str, float]:
        return {
            **{name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS},
            TORSO_JOINT: float(self.io.sensors[TORSO_JOINT].getValue()),
        }

    def _kinematics_telemetry(self) -> dict[str, object]:
        """Live GT Solid poses for the ADR-0020 validation sweep (flag-gated).

        Mirrors the export schema field names so the same loader functions parse
        both payloads: full world pose (position + row-major rotation) of the
        Robot-node base frame, of every torso/arm joint's parent and endPoint
        child Solid, and of the two finger endPoint Solids.
        """
        joint_order = (TORSO_JOINT, *ARM_JOINTS)
        telemetry: dict[str, object] = {
            "schema_version": 1,
            "units": {"length": "m", "angle": "rad"},
            "base_frame": solid_pose(self.io.me, "RobotNode").to_payload(),
            "joint_endpoint_solids": {
                name: child_solid(self.io, name).to_payload() for name in joint_order
            },
            "joint_parent_solids": {
                name: parent_solid(read_joint_node(self.io, name), name).to_payload()
                for name in joint_order
            },
            "joint_sensors_rad": self._joint_positions(),
            "finger_endpoint_solids": {
                motor.getName(): child_solid(self.io, motor.getName()).to_payload()
                for motor, _sensor in self.io.fingers
            },
        }
        return telemetry

    def _settle_reset_posture(self) -> None:
        """Reproduce the original INIT ordering: torso first, then head.

        Original sources:
        ``controllers/tiago_arm_api/tiago_arm_api.py:357-364`` commands
        ``torso_lift_joint`` and steps up to 1000 times until the sensor is within 0.002 m.
        ``controllers/tiago_arm_api/tiago_arm_api.py:366-374`` then clamps, sets head
        pan/tilt at velocity 0.5, and waits ``int(2000 / ts)`` steps. The packaged copy is
        ``src/vlarobot/backends/webots/pick_place.py:121-145``.
        """
        if TORSO_JOINT not in self.io.sensors:
            return
        torso_target = float(SETTINGS.torso_height or 0.0)
        self.io.motors[TORSO_JOINT].setPosition(torso_target)
        torso_settled = False
        for _ in range(RESET_TORSO_TIMEOUT_STEPS):
            if self.visualization.step() == -1:
                raise RuntimeError("Webots stopped while initializing the torso posture.")
            if (
                abs(self.io.sensors[TORSO_JOINT].getValue() - torso_target)
                < RESET_TORSO_SETTLE_TOLERANCE_M
            ):
                torso_settled = True
                break
        if not torso_settled:
            raise RuntimeError(
                "Torso did not settle within "
                f"{RESET_TORSO_SETTLE_TOLERANCE_M} m of the configured height."
            )

        for name, target in zip(HEAD_JOINTS, [HEAD_PAN, HEAD_TILT], strict=True):
            motor = self.io.motors[name]
            minimum, maximum = motor.getMinPosition(), motor.getMaxPosition()
            if maximum > minimum:
                target = min(max(target, minimum), maximum)
            motor.setVelocity(0.5)
            motor.setPosition(target)
        head_steps = max(1, int(RESET_HEAD_SETTLE_MS / self.io.ts))
        for _ in range(head_steps):
            if self.visualization.step() == -1:
                raise RuntimeError("Webots stopped while initializing the head posture.")

    def _settle_post_reset(self) -> dict[str, object]:
        """Step until base velocities and tilt-change are stable, or fail explicitly.

        Campaign requirement (MEASURED intent): before first command, wait until base linear and
        angular velocity magnitudes < thresholds and tilt change < threshold for N consecutive
        steps, capped. Tilt is roll/pitch magnitude from base orientation; "tilt change" is
        |tilt-step - tilt-prev| in degrees.
        """
        consecutive = 0
        steps_used = 0
        last_tilt_deg: float | None = None
        reason = "max_steps"
        for _ in range(int(RESET_POST_SETTLE_MAX_STEPS)):
            if self.visualization.step() == -1:
                raise RuntimeError("Webots simulation stopped while post-reset settling.")
            self._physics_step_count += 1
            steps_used += 1
            vel = self.io.me.getVelocity()
            if vel is None:
                raise RuntimeError("Post-reset settle failed: base velocity unavailable.")
            v = np.asarray(vel, dtype=float)
            if v.shape[0] < 6 or not np.all(np.isfinite(v[:6])):
                raise RuntimeError("Post-reset settle failed: base velocity is non-finite.")
            lin = float(np.linalg.norm(v[:3]))
            ang = float(np.linalg.norm(v[3:6]))
            rot = np.asarray(self.io.me.getOrientation(), dtype=float).reshape((3, 3))
            roll = math.atan2(rot[2, 1], rot[2, 2])
            pitch = math.atan2(-rot[2, 0], math.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
            tilt_deg = math.degrees(math.sqrt(roll * roll + pitch * pitch))
            tilt_delta = 0.0 if last_tilt_deg is None else abs(tilt_deg - last_tilt_deg)
            last_tilt_deg = tilt_deg
            stable = (
                lin < float(RESET_POST_SETTLE_VELOCITY_THRESHOLD_M_S)
                and ang < float(RESET_POST_SETTLE_ANG_VELOCITY_THRESHOLD_RAD_S)
                and tilt_delta < float(RESET_POST_SETTLE_TILT_DELTA_DEG)
            )
            if stable:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= int(RESET_POST_SETTLE_CONSECUTIVE_STEPS):
                reason = "settled"
                break
        info = {
            "enabled": True,
            "steps_used": steps_used,
            "consecutive_stable_steps": consecutive,
            "required_consecutive_steps": int(RESET_POST_SETTLE_CONSECUTIVE_STEPS),
            "reason": reason,
            "thresholds": {
                "linear_m_s": float(RESET_POST_SETTLE_VELOCITY_THRESHOLD_M_S),
                "angular_rad_s": float(RESET_POST_SETTLE_ANG_VELOCITY_THRESHOLD_RAD_S),
                "tilt_delta_deg": float(RESET_POST_SETTLE_TILT_DELTA_DEG),
            },
            "final": {
                "linear_speed_m_s": lin if steps_used else None,
                "angular_speed_rad_s": ang if steps_used else None,
                "tilt_deg": last_tilt_deg,
            },
            # ADR-0022: non-settling is recorded, not fatal. MEASURED: the base retains
            # persistent angular jitter (~1.0-1.6e-4 rad/s) around the 1e-4 rad/s threshold,
            # so a knife-edge share of fresh worlds never accumulate 20 stable steps; failing
            # reset (RuntimeError -> error reply -> runner kill) aborted otherwise healthy
            # episodes. Thresholds are unchanged; the physical-state guard still vetoes
            # genuinely contaminated state.
            "fatal": False,
        }
        if reason != "settled":
            LOGGER.warning("post_reset_settle: not settled (recorded, non-fatal): %s", info)
            return info
        LOGGER.info("post_reset_settle: %s", info)
        return info

    def _arm7_fk_world_pose(
        self,
        q: np.ndarray,
        base_position: np.ndarray,
        base_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """World position and rotation of the IKPy ``arm_7_joint`` frame."""
        try:
            transforms = self.kinematics.chain.forward_kinematics(list(q), full_kinematics=True)
            transform = np.asarray(transforms[self.kinematics.idx["arm_7_joint"]])
        except (AttributeError, IndexError, KeyError):
            return None
        local_position = transform[:3, 3]
        local_rotation = transform[:3, :3]
        return (
            base_rotation @ local_position + base_position,
            base_rotation @ local_rotation,
        )

    def _arm7_fk_world(
        self,
        q: np.ndarray,
        base_position: np.ndarray,
        base_rotation: np.ndarray,
    ) -> np.ndarray | None:
        """World position of the IKPy ``arm_7_joint`` frame origin."""
        pose = self._arm7_fk_world_pose(q, base_position, base_rotation)
        return None if pose is None else pose[0]

    def _arm7_endpoint_offset(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Reset-time ``(world offset, arm_7 link world rotation)`` to the endPoint Solid.

        The offset is stored in the arm_7 link frame, so it is rotated by the link's current
        rotation before comparing it with a later sample. A constant tool offset therefore
        cancels even as the wrist moves.
        """
        lookup = getattr(self.io, "joint_endpoint_world_position", None)
        endpoint = lookup("arm_7_joint") if lookup is not None else None
        if endpoint is None:
            return None
        base_position = np.asarray(self.io.me.getPosition(), dtype=float)
        base_rotation = np.asarray(self.io.me.getOrientation()).reshape((3, 3))
        pose = self._arm7_fk_world_pose(self.frames.current_q(), base_position, base_rotation)
        if pose is None:
            return None
        fk_world, link_rotation = pose
        return np.asarray(endpoint, dtype=float) - fk_world, link_rotation

    def _set_gripper(self, opening: float, force: float | None = None) -> None:
        full_stroke = max(self.io.full_stroke, 1e-9)
        position = (
            self.io.closed_pos + (self.io.open_pos - self.io.closed_pos) * opening / full_stroke
        )
        for motor, _ in self.io.fingers:
            if force is not None:
                max_force = (
                    float(motor.getMaxForce()) if hasattr(motor, "getMaxForce") else float(force)
                )
                motor.setAvailableForce(min(float(force), max_force))
            motor.setPosition(position)

    def _safety_limits(self) -> SafetyLimits:
        return SafetyLimits(
            joint_position={
                name: (
                    float(self.io.motors[name].getMinPosition()),
                    float(self.io.motors[name].getMaxPosition()),
                )
                for name in ARM_JOINTS
            },
            joint_velocity={
                name: float(self.io.motors[name].getMaxVelocity()) for name in ARM_JOINTS
            },
            workspace_min=SAFETY_PROFILE.workspace_min,
            workspace_max=SAFETY_PROFILE.workspace_max,
            gripper_opening=SETTINGS.gripper_opening_limits,
            gripper_force_max=SETTINGS.gripper_force_max,
            command_horizon=SETTINGS.safety_command_horizon,
        )


def _backend_timings_enabled() -> bool:
    """Backend diagnostics timing is opt-in so normal runs stay unchanged."""
    return os.environ.get("VLAROBOT_BACKEND_TIMINGS") == "1"


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
        w = 0.25 * scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
        w = (rotation[2, 1] - rotation[1, 2]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
        w = (rotation[0, 2] - rotation[2, 0]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
        w = (rotation[1, 0] - rotation[0, 1]) / scale
    return float(x), float(y), float(z), float(w)


def _quaternion_distance(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    dot = abs(sum(a * b for a, b in zip(first, second, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))
