"""Deterministic action-chunk robot/object trajectories with explicit contact assumptions."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from vlarobot.core.types import (
    Action,
    ActionChunk,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    JointTarget,
    Outcome,
    Pose3D,
    RobotState,
    Vec3,
    WorldState,
)
from vlarobot.world_model.kinematic import VelocityLimitedKinematics


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose normalized xyzw rotations; reject malformed state rather than repair silently."""
    for q in (a, b):
        if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(q @ q, 1):
            raise ValueError("unit xyzw quaternion required")
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]


def rotate(q: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Rotate a base-independent vector with an xyzw quaternion."""
    return point + 2 * np.cross(q[:3], np.cross(q[:3], point) + q[3] * point)


def compose(a: Pose3D, b: Pose3D) -> Pose3D:
    qa, qb = np.asarray(a.orientation), np.asarray(b.orientation)
    q = quaternion_product(qa, qb)
    position = a.position.as_array() + rotate(qa, b.position.as_array())
    return Pose3D(Vec3(*map(float, position)), (float(q[0]), float(q[1]), float(q[2]), float(q[3])))


@dataclass(frozen=True)
class Attachment:
    """Observed attachment, never inferred as a successful grasp from a close command."""

    object_id: str
    tool_to_object: Pose3D
    release_half_aperture_m: float


@dataclass(frozen=True)
class Trajectory:
    states: tuple[WorldState, ...]
    dt_s: float
    hard_reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    released_objects: tuple[str, ...]
    calibration_id: str | None = None


class AnalyticSurrogate:
    """Velocity-limited FK/IK, rigid observed attachment and gravity-to-support release.

    No new grasp is asserted. Unknown contacts/supports and arm-sweep checks stay explicit.
    `predict_trajectory` provides continuous targets without invented probabilities;
    Protocol `rollout` additionally requires a validation-derived probability calibrator.
    """

    def __init__(
        self,
        tracking: VelocityLimitedKinematics,
        *,
        inverse: Callable[[Pose3D, RobotState], JointTarget | None],
        gripper_speed_m_s: float,
        gravity_m_s2: float,
        support_z_m: Mapping[str, float],
        validate: Callable[[Action, RobotState], bool],
        prepare_inverse: Callable[[RobotState], None] | None = None,
        check_transition: Callable[[WorldState, WorldState], bool | None] | None = None,
        calibrate: Callable[[Trajectory], Outcome] | None = None,
        calibration_id: str | None = None,
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (gripper_speed_m_s, gravity_m_s2)):
            raise ValueError("explicit positive physical tracking parameters required")
        if not all(math.isfinite(v) for v in support_z_m.values()):
            raise ValueError("finite base-frame support elevations required")
        if (calibrate is None) != (calibration_id is None):
            raise ValueError("probability calibration requires provenance")
        self.tracking, self.inverse, self.validate = tracking, inverse, validate
        self.gripper_speed, self.gravity = gripper_speed_m_s, gravity_m_s2
        self.supports = dict(support_z_m)
        self.calibrate, self.calibration_id = calibrate, calibration_id
        self.check_transition = check_transition
        self.prepare_inverse = prepare_inverse

    def predict_trajectory(
        self, state: WorldState, chunk: ActionChunk, *, attachment: Attachment | None = None
    ) -> Trajectory:
        if not chunk.actions or not math.isfinite(chunk.dt) or chunk.dt <= 0:
            raise ValueError("nonempty positive-duration action chunk required")
        ids = [o.object_id for o in state.scene.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("unique object identities required")
        if attachment is not None and (
            attachment.object_id not in ids
            or state.robot.gripper_holding is not True
            or not math.isfinite(attachment.release_half_aperture_m)
            or attachment.release_half_aperture_m <= 0
        ):
            raise ValueError("attachment requires an explicitly observed held object")
        current = copy.deepcopy(state)
        if self.prepare_inverse is not None:
            self.prepare_inverse(current.robot)
        states, reasons, released = [copy.deepcopy(current)], [], []
        velocities = {name: np.zeros(3) for name in ids}
        assumptions = [
            "nominal velocity tracking",
            "no new grasp inferred",
            "no full-arm contact physics",
            "inelastic horizontal support; released angular dynamics unmodeled",
        ]
        if attachment:
            assumptions.append("rigid observed attachment until aperture release")
        for action in chunk.actions:
            previous = copy.deepcopy(current)
            just_released = None
            if not self.validate(action, current.robot):
                reasons.append("safety-monitor-veto")
                break
            command = None
            grip = None
            if isinstance(action, JointTarget):
                command = action
            elif isinstance(action, (EEDelta, EEPoseTarget)):
                if isinstance(action, EEDelta):
                    r = action.drot.as_array()
                    angle = np.linalg.norm(r)
                    delta = np.r_[
                        r * (np.sin(angle / 2) / angle if angle else 0.5), np.cos(angle / 2)
                    ]
                    q = quaternion_product(delta, np.asarray(current.robot.ee_pose.orientation))
                    position = current.robot.ee_pose.position.as_array() + action.dpos.as_array()
                    target = Pose3D(
                        Vec3(*map(float, position)),
                        (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                    )
                    grip = action.gripper
                else:
                    target, grip = action.pose, action.gripper
                command = self.inverse(target, current.robot)
                if command is None:
                    reasons.append("ik-unreachable")
                    break
                if not self.validate(command, current.robot):
                    reasons.append("ik-joint-safety-veto")
                    break
            elif isinstance(action, GripperCmd):
                grip = action.opening
            else:
                reasons.append("unsupported-action")
                break
            if command is not None:
                current.robot = self.tracking.step(current.robot, command, chunk.dt)
            else:
                current.robot.joint_vel = {name: 0.0 for name in current.robot.joint_pos}
            if grip is not None:
                if not math.isfinite(grip):
                    reasons.append("nonfinite-gripper")
                    break
                bound = self.gripper_speed * chunk.dt
                current.robot.gripper_opening += float(
                    np.clip(grip - current.robot.gripper_opening, -bound, bound)
                )
            if attachment and current.robot.gripper_opening >= attachment.release_half_aperture_m:
                name = attachment.object_id
                released.append(name)
                release_pose = compose(current.robot.ee_pose, attachment.tool_to_object)
                old_object = next(o for o in previous.scene.objects if o.object_id == name)
                velocities[name] = (
                    release_pose.position.as_array() - old_object.pose.position.as_array()
                ) / chunk.dt
                just_released = (name, release_pose)
                attachment = None
                current.robot.gripper_holding = False
            objects = []
            for obj in current.scene.objects:
                if attachment and obj.object_id == attachment.object_id:
                    pose = compose(current.robot.ee_pose, attachment.tool_to_object)
                elif just_released and obj.object_id == just_released[0]:
                    pose = just_released[1]
                elif obj.object_id in self.supports:
                    floor = self.supports[obj.object_id] + obj.dimensions.z / 2
                    velocity = velocities[obj.object_id]
                    position = obj.pose.position.as_array() + velocity * chunk.dt
                    position[2] -= self.gravity * chunk.dt**2 / 2
                    if position[2] <= floor:
                        position[2] = floor
                        velocities[obj.object_id] = np.zeros(3)
                    else:
                        velocities[obj.object_id][2] -= self.gravity * chunk.dt
                    pose = replace(obj.pose, position=Vec3(*map(float, position)))
                else:
                    pose = obj.pose
                    reasons.append("unknown-support:" + obj.object_id)
                objects.append(replace(obj, pose=pose))
            current.scene.objects = objects
            verdict = self.check_transition(previous, current) if self.check_transition else None
            if verdict is not True:
                reasons.append(
                    "arm-sweep-collision" if verdict is False else "unverified-arm-sweep"
                )
            states.append(copy.deepcopy(current))
        return Trajectory(
            tuple(states),
            chunk.dt,
            tuple(dict.fromkeys(reasons)),
            tuple(assumptions),
            tuple(released),
            self.calibration_id,
        )

    def rollout(self, state: WorldState, chunk: ActionChunk, n_samples: int) -> list[Outcome]:
        """Protocol adapter: deterministic samples are identical, not independent evidence."""
        if n_samples < 1:
            raise ValueError("positive sample count required")
        if self.calibrate is None:
            raise ValueError(
                "probabilistic outcomes require validation calibration; use predict_trajectory"
            )
        trajectory = self.predict_trajectory(state, chunk)
        result = self.calibrate(trajectory)
        if any(
            not math.isfinite(p) or not 0 <= p <= 1
            for p in (
                result.success_probability,
                result.collision_probability,
                result.drop_probability,
            )
        ):
            raise ValueError("invalid calibrated outcome probability")
        return [result for _ in range(n_samples)]
