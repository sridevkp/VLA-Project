"""RobotBackend adapter for a running Webots TIAGo world."""

from __future__ import annotations

import logging
import math

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    CAN_DEF,
    HEAD_JOINTS,
    HEAD_PAN,
    HEAD_TILT,
    SAFETY_PROFILE,
    SETTINGS,
    TORSO_JOINT,
)
from vlarobot.backends.webots.frames import Frames
from vlarobot.backends.webots.gripper import Gripper
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.robot_io import RobotIO
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
        if scenario.objects:
            raise NotImplementedError("Webots scenario spawning is scheduled for M2.")

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
        self.can_node.getField("translation").setSFVec3f(self._initial_can_translation)
        self.can_node.getField("rotation").setSFRotation(self._initial_can_rotation)
        self.io.me.getField("translation").setSFVec3f(self._initial_base_translation)
        self.io.me.getField("rotation").setSFRotation(self._initial_base_rotation)
        self.io.robot.simulationResetPhysics()
        can_node = self.io.robot.getFromDef(CAN_DEF)
        if can_node is None:
            raise RuntimeError(f"No node with DEF {CAN_DEF} after resetting physics.")
        self.can_node = can_node
        self.io.can_node = can_node
        if self.io.robot.step(self.io.ts) == -1:
            raise RuntimeError("Webots simulation stopped while restoring the reset pose.")
        if all(name in self.io.motors for name in HEAD_JOINTS):
            self._settle_reset_posture()
        reset_position = self.io.me.getPosition()
        reset_rotation = np.asarray(self.io.me.getOrientation()).reshape((3, 3))
        self._reset_base_position = np.asarray(reset_position, dtype=float)
        self._reset_base_yaw = math.atan2(reset_rotation[1, 0], reset_rotation[0, 0])
        self.gripper.last_close_stalled = None
        self.gripper.last_close_opening = None
        self._episode_start_time = float(self.io.robot.getTime())
        self._last_joint_pos = self._joint_positions()
        self._last_time = self._episode_start_time
        return self._observation()

    def step(self, n_steps: int = 1) -> Observation:
        if self._stopped:
            raise RuntimeError("Cannot step a stopped backend.")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        for _ in range(n_steps):
            if self.visualization.step() == -1:
                raise RuntimeError("Webots simulation stopped while stepping.")
        return self._observation()

    def apply(self, action: Action) -> ApplyResult:
        if self._stopped:
            raise RuntimeError("Cannot apply an action to a stopped backend.")
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

    def get_ground_truth(self) -> GroundTruth:
        position = self.can_node.getPosition()
        rotation = np.asarray(self.can_node.getOrientation()).reshape((3, 3))
        pose = Pose3D(
            Vec3(float(position[0]), float(position[1]), float(position[2])),
            _rotation_to_quaternion(rotation),
        )
        return GroundTruth(object_poses={CAN_DEF: pose})

    def stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            self.io.robot.simulationQuit(0)

    def _observation(self) -> Observation:
        image = np.asarray(self.camera.getImageArray(), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"Webots camera returned an invalid image shape: {image.shape}.")
        image = image[:, :, :3].transpose((1, 0, 2)).copy()
        width, height = image.shape[1], image.shape[0]
        fov = float(self.camera.getFov())
        focal = width / (2.0 * math.tan(fov / 2.0))
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

        base_position = self.io.me.getPosition()
        base_rotation = np.asarray(self.io.me.getOrientation()).reshape((3, 3))
        base_yaw = math.atan2(base_rotation[1, 0], base_rotation[0, 0])
        end_effector = self.kinematics.forward(self.frames.current_q())
        base_position_array = np.asarray(base_position, dtype=float)
        base_drift = base_position_array - self._reset_base_position
        base_yaw_delta = _wrap_angle(base_yaw - self._reset_base_yaw)
        full_q = self.frames.current_q()
        arm7_world = None
        try:
            transforms = self.kinematics.chain.forward_kinematics(full_q, full_kinematics=True)
            arm7_local = np.asarray(transforms[self.kinematics.idx["arm_7_joint"]])[:3, 3]
            arm7_world = base_rotation @ arm7_local + base_position_array
        except (AttributeError, IndexError, KeyError):
            pass
        endpoint_lookup = getattr(self.io, "joint_endpoint_world_position", None)
        arm7_ground_truth = endpoint_lookup("arm_7_joint") if endpoint_lookup else None
        contact_lookup = getattr(self.io, "contact_nodes", None)
        contacts = contact_lookup() if contact_lookup else []
        wheel_lookup = getattr(self.io, "wheel_status", None)
        diagnostics: dict[str, object] = {
            "torso_lift_joint_sensor_rad": float(self.io.sensors[TORSO_JOINT].getValue()),
            "base_drift_world_xyz_m": base_drift.tolist(),
            "base_drift_distance_m": float(np.linalg.norm(base_drift[:2])),
            "base_drift_yaw_rad": base_yaw_delta,
            "base_drift_yaw_deg": math.degrees(base_yaw_delta),
            "wheel_motors": wheel_lookup() if wheel_lookup else {},
            "contact_nodes": contacts,
            "arm_7_joint_fk_world_xyz_m": arm7_world.tolist() if arm7_world is not None else None,
            "arm_7_joint_ground_truth_world_xyz_m": arm7_ground_truth,
            "arm_7_joint_fk_ground_truth_error_m": (
                float(np.linalg.norm(arm7_world - arm7_ground_truth))
                if arm7_ground_truth is not None and arm7_world is not None
                else None
            ),
        }
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

    def _settle_reset_posture(self) -> None:
        if TORSO_JOINT not in self.io.sensors:
            return
        for name in HEAD_JOINTS:
            self.io.motors[name].setVelocity(0.5)
        target = {
            TORSO_JOINT: float(SETTINGS.torso_height or 0.0),
            HEAD_JOINTS[0]: HEAD_PAN,
            HEAD_JOINTS[1]: HEAD_TILT,
        }
        for name, value in target.items():
            self.io.motors[name].setPosition(value)
        # The legacy initialize_posture waits for torso error < 0.002 m and
        # gives the commanded head pose two seconds to settle.
        head_steps = max(1, int(math.ceil(2.0 * 1000.0 / self.io.ts)))
        torso_settled = False
        for step in range(max(1000, head_steps)):
            if self.visualization.step() == -1:
                raise RuntimeError("Webots stopped while initializing torso/head posture.")
            torso_settled = (
                abs(self.io.sensors[TORSO_JOINT].getValue() - target[TORSO_JOINT]) < 0.002
            )
            if torso_settled and step + 1 >= head_steps:
                break
        if not torso_settled:
            raise RuntimeError("Torso did not settle within 0.002 of the configured height.")

    def _set_gripper(self, opening: float, force: float | None = None) -> None:
        full_stroke = max(self.io.full_stroke, 1e-9)
        position = (
            self.io.closed_pos + (self.io.open_pos - self.io.closed_pos) * opening / full_stroke
        )
        for motor, _ in self.io.fingers:
            if force is not None:
                motor.setAvailableForce(force)
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
