"""Deterministic fake backend for core and integration tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np

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

_ARM_JOINTS = tuple(f"arm_{index}_joint" for index in range(1, 8))
_CAMERA_INFO = CameraInfo(
    K=((1.0, 0.0, 32.0), (0.0, 1.0, 24.0), (0.0, 0.0, 1.0)),
    T_base_cam=(
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    width=64,
    height=48,
)
# Mirrors configs/scenarios.yaml world.can_xyz (drift-guarded by tests/test_scenarios.py).
CAN_WORLD_XYZ = (-0.799879, -4.38998, 0.810949)
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """World-Z rotation as an (x, y, z, w) quaternion."""
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


class FakeBackend:
    """Small deterministic backend with the same API as simulator clients."""

    def __init__(
        self,
        safety_monitor: SafetyMonitor | None = None,
        step_duration: float = 0.032,
        *,
        can_world_xyz: tuple[float, float, float] = CAN_WORLD_XYZ,
        gripper_rate_limited: bool = False,
        finger_speed_m_s: float | None = None,
    ) -> None:
        if step_duration <= 0.0:
            raise ValueError("step_duration must be positive.")
        if gripper_rate_limited and (
            finger_speed_m_s is None
            or not math.isfinite(float(finger_speed_m_s))
            or float(finger_speed_m_s) <= 0.0
        ):
            raise ValueError("rate-limited gripper requires a positive finite finger speed")
        self.safety_monitor = safety_monitor or SafetyMonitor(
            SafetyLimits(
                joint_position={name: (-3.14, 3.14) for name in _ARM_JOINTS},
                joint_velocity={name: 1.0 for name in _ARM_JOINTS},
            )
        )
        self.step_duration = step_duration
        self.can_world_xyz = tuple(float(value) for value in can_world_xyz)
        self.gripper_rate_limited = bool(gripper_rate_limited)
        self.finger_speed_m_s = None if finger_speed_m_s is None else float(finger_speed_m_s)
        self._gripper_target: float | None = None
        self._time = 0.0
        self._seq = 0
        self._stopped = False
        self._scenario = ScenarioSpec()
        self._rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        self._depth = np.zeros((48, 64), dtype=np.float32)
        self._robot_state = self._initial_robot_state()
        self._gripper_target = self._robot_state.gripper_opening

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if self._stopped:
            raise RuntimeError("Cannot reset a stopped backend.")
        self._scenario = deepcopy(scenario)
        self._time = 0.0
        self._seq = 0
        self._rgb.fill(0)
        self._depth.fill(0.0)
        self._robot_state = self._initial_robot_state()
        self._gripper_target = self._robot_state.gripper_opening
        return self._observation()

    def step(self, n_steps: int = 1) -> Observation:
        if self._stopped:
            raise RuntimeError("Cannot step a stopped backend.")
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        self._time += n_steps * self.step_duration
        self._seq += n_steps
        self._robot_state.joint_vel = {name: 0.0 for name in self._robot_state.joint_pos}
        if self.gripper_rate_limited and self._gripper_target is not None:
            current = float(self._robot_state.gripper_opening)
            target = float(self._gripper_target)
            travel = float(self.finger_speed_m_s) * self.step_duration * n_steps
            if current < target:
                self._robot_state.gripper_opening = min(target, current + travel)
            elif current > target:
                self._robot_state.gripper_opening = max(target, current - travel)
        return self._observation()

    def apply(self, action: Action) -> ApplyResult:
        if self._stopped:
            raise RuntimeError("Cannot apply an action to a stopped backend.")
        validation = self.safety_monitor.validate(action, self._robot_state)
        if not validation.accepted:
            return validation
        if isinstance(action, JointTarget):
            self._robot_state.joint_pos.update(action.q)
        elif isinstance(action, EEPoseTarget):
            self._robot_state.ee_pose = action.pose
            if action.gripper is not None:
                self._robot_state.gripper_opening = action.gripper
        elif isinstance(action, EEDelta):
            position = self._robot_state.ee_pose.position
            self._robot_state.ee_pose = Pose3D(
                Vec3(
                    position.x + action.dpos.x,
                    position.y + action.dpos.y,
                    position.z + action.dpos.z,
                ),
                self._robot_state.ee_pose.orientation,
            )
            self._robot_state.gripper_opening = action.gripper
        elif isinstance(action, BaseGoal):
            self._robot_state.base_pose = action.pose
        elif isinstance(action, GripperCmd):
            if self.gripper_rate_limited:
                self._gripper_target = float(action.opening)
            else:
                self._robot_state.gripper_opening = action.opening
                self._gripper_target = float(action.opening)
        else:
            return ApplyResult(False, f"unsupported action type: {type(action).__name__}")
        return ApplyResult(True)

    def get_ground_truth(self) -> GroundTruth:
        object_poses: dict[str, Pose3D] = {}
        can_xyz = self.can_world_xyz
        can_quat = _IDENTITY_QUAT
        offset = self._scenario.can_xy_yaw_offset
        if offset is not None:
            dx, dy, yaw = (float(value) for value in offset)
            can_xyz = (can_xyz[0] + dx, can_xyz[1] + dy, can_xyz[2])
            can_quat = _yaw_to_quat(yaw)
        object_poses["CAN"] = Pose3D(Vec3(*can_xyz), can_quat)
        for entry in self._scenario.objects:
            def_name = str(entry.get("def", ""))
            xyz = entry.get("xyz")
            if not def_name or xyz is None:
                continue
            values = tuple(float(value) for value in cast("Sequence[float]", xyz))
            yaw_offset = float(cast(float, entry.get("yaw_offset", 0.0)))
            object_poses[def_name] = Pose3D(Vec3(*values), _yaw_to_quat(yaw_offset))
        return GroundTruth(object_poses=object_poses)

    def stop(self) -> None:
        self._stopped = True

    def export_kinematics(self, action: str = "poll") -> dict[str, object]:
        """Fake chunked export (no Webots): progresses per poll and returns a snapshot."""
        if not hasattr(self, "_export_job") or self._export_job is None or action == "start":
            self._export_job = {"index": 0, "total": 3, "per_joint": [], "done": False}
        if action == "cancel":
            self._export_job = {"index": 0, "total": 3, "per_joint": [], "done": True}
            return {"state": "cancelled", "done": True, "snapshot": None, "progress": {}}
        job_any = self._export_job
        assert isinstance(job_any, dict)
        job: dict[str, Any] = job_any
        per_joint_any = job.setdefault("per_joint", [])
        per_joint: list[dict[str, Any]] = per_joint_any if isinstance(per_joint_any, list) else []
        job["per_joint"] = per_joint
        if not bool(job.get("done")):
            idx = int(job.get("index", 0) or 0)
            total = int(job.get("total", 0) or 0)
            if idx < total:
                name = f"arm_{idx + 1}_joint"
                per_joint.append(
                    {
                        "name": name,
                        "kind": "hinge",
                        "steps_used": 5,
                        "measured_angle_moved_rad": 0.05,
                        "reason": "ok",
                        "settled": True,
                        "axis_source": "posture_probe",
                    }
                )
                job["index"] = idx + 1
            if int(job.get("index", 0) or 0) >= total:
                job["done"] = True
        done = bool(job.get("done"))
        snapshot = (
            {
                "schema_version": 1,
                "units": {"length": "m", "angle": "rad"},
                "source": "fake",
                "valid": True,
                "joints": [],
                "finger_endpoint_solids": {},
                "base_frame": {},
                "export_progress": {
                    "steps_used_total": len(per_joint) * 5,
                    "per_joint": list(per_joint),
                    "error": None,
                },
            }
            if done
            else None
        )
        return {
            "state": "done" if done else "in_progress",
            "done": done,
            "current_index": int(job.get("index", 0) or 0),
            "total_joints": int(job.get("total", 0) or 0),
            "steps_used_total": len(per_joint) * 5,
            "total_step_cap": 2000,
            "per_joint": list(per_joint),
            "error": None,
            "snapshot": snapshot,
        }

    def _observation(self) -> Observation:
        return Observation(
            t=self._time,
            rgb=self._rgb.copy(),
            depth=self._depth.copy(),
            cam_info=_CAMERA_INFO,
            robot=deepcopy(self._robot_state),
        )

    @staticmethod
    def _initial_robot_state() -> RobotState:
        return RobotState(
            joint_pos={name: 0.0 for name in _ARM_JOINTS},
            joint_vel={name: 0.0 for name in _ARM_JOINTS},
            base_pose=Pose2D(0.0, 0.0, 0.0),
            ee_pose=Pose3D(Vec3(0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
            gripper_opening=0.045,
            base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )
