"""Concrete, inert backend registration for the production WorldMind composition.

This module is the only place that imports the existing Webots-backed geometry and safety
objects for WorldMind. Registering them does not launch Webots, solve IK, run inference, train,
actuate or access the TEST split. Geometry is constructed lazily when a reviewed config actually
calls a factory.

Eq. 3.8 component formulas, weights, ranking direction and calibrated risk thresholds are
deliberately absent: no authoritative source supplies them. Their registry stays empty so
``validate_worldmind_bindings`` fails closed until owner-reviewed values are supplied.
"""

from __future__ import annotations

import hashlib
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.webots.config import (
    APPROACH_H,
    ARM_JOINTS,
    GRASP_Z,
    PLACE_CLEAR,
    SAFETY_PROFILE,
    SETTINGS,
    SETTINGS_PATH,
    URDF_PATH,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.pose_solver import BoundedPoseSolver
from vlarobot.backends.webots.urdf_registration import load_registration
from vlarobot.core.types import (
    GripperCmd,
    JointTarget,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_bindings import (
    ANALYTIC_SURROGATE_BINDING,
    CORRECTED_POSE_SOLVER_BINDING,
    CORRECTED_WAYPOINTS_BINDING,
    FINITE_CHUNK_CHECK,
    PLACEMENT_VERIFIER_BINDING,
    RUNTIME_SAFETY_BINDING,
)
from vlarobot.eval.worldmind_config import WorldMindBindings, WorldMindConfig
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.planning.verification import PlacementVerifier
from vlarobot.vla.action_adapter import PoseSolution
from vlarobot.world_model.kinematic import MemoizedForward, VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate, compose, rotate

CORRECTED_REGISTRATION_RELATIVE = Path("calibration") / "urdf-base-train-v1.yaml"
CORRECTED_REGISTRATION_SHA256 = "dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0"

SURROGATE_INPUTS = ("forward", "inverse", "validate", "tracking")
SAFETY_INPUTS = ("monitor", "limits")
VERIFICATION_INPUTS: tuple[str, ...] = ()

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
# A scripted station cadence, not a scientific threshold. It is explicit so every chunk is
# reviewable and its hold steps sum to the frozen H=50 control horizon.
_CORRECTED_STATIONS: tuple[tuple[str, int, str, float], ...] = (
    ("PREGRASP", 12, "target", GRASP_Z + APPROACH_H),
    ("GRASP_APPROACH", 8, "target", GRASP_Z),
    ("LIFT", 10, "target", GRASP_Z + APPROACH_H),
    ("TRANSIT", 10, "destination", GRASP_Z + APPROACH_H),
    ("PLACE_DESCEND", 8, "destination", GRASP_Z + PLACE_CLEAR),
    ("PLACE_RETREAT", 1, "destination", GRASP_Z + APPROACH_H),
)


def corrected_registration_path() -> Path:
    """Return the frozen, hash-pinned corrected registration path."""
    path = (SETTINGS_PATH.parent / CORRECTED_REGISTRATION_RELATIVE).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CORRECTED_REGISTRATION_SHA256:
        raise ValueError("corrected URDF registration bytes differ from the frozen provenance")
    load_registration(path)
    return path


@lru_cache(maxsize=1)
def corrected_kinematics() -> Kinematics:
    """Build the existing URDF chain with the corrected registration; no IK solve is performed."""
    corrected_registration_path()
    previous = os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE")
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = CORRECTED_REGISTRATION_RELATIVE.as_posix()
    try:
        kinematics = Kinematics("urdf")
    finally:
        if previous is None:
            os.environ.pop("VLAROBOT_URDF_REGISTRATION_FILE", None)
        else:
            os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = previous
    if kinematics.registration_sha256 != CORRECTED_REGISTRATION_SHA256:
        raise ValueError("constructed kinematics did not use the corrected registration")
    return kinematics


@lru_cache(maxsize=1)
def _urdf_velocities() -> dict[str, float]:
    tree = ET.parse(URDF_PATH)
    velocities: dict[str, float] = {}
    for name in ARM_JOINTS:
        limit = tree.find(f'./joint[@name="{name}"]/limit')
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"URDF velocity limit unavailable for {name}")
        velocities[name] = float(limit.attrib["velocity"])
    if not all(math.isfinite(value) and value > 0 for value in velocities.values()):
        raise ValueError("URDF velocity limits must be finite and positive")
    return velocities


def runtime_safety_limits() -> SafetyLimits:
    """Use existing effective robot/URDF/safety settings, never default-only limits."""
    kin = corrected_kinematics()
    return SafetyLimits(
        joint_position={name: kin.bounds(name) for name in ARM_JOINTS},
        joint_velocity=dict(_urdf_velocities()),
        workspace_min=SAFETY_PROFILE.workspace_min,
        workspace_max=SAFETY_PROFILE.workspace_max,
        gripper_opening=SETTINGS.gripper_opening_limits,
        gripper_force_max=SETTINGS.gripper_force_max,
        command_horizon=SETTINGS.safety_command_horizon,
    )


def runtime_safety_factory(config: WorldMindConfig, inputs: Mapping[str, Any]) -> SafetyMonitor:
    """Bind the existing live SafetyMonitor to the reviewed effective limits."""
    del config, inputs
    return SafetyMonitor(runtime_safety_limits())


class _LazyCorrectedPoseSolver:
    """Defer geometry construction until a reviewed solve is actually requested."""

    def __init__(self) -> None:
        self._inner: BoundedPoseSolver | None = None

    @property
    def inner(self) -> BoundedPoseSolver:
        if self._inner is None:
            kin = corrected_kinematics()
            self._inner = BoundedPoseSolver(kin, max_joint_step_rad=SETTINGS.max_joint_step_rad)
        return self._inner

    def solve(self, pose: Pose3D, state: RobotState) -> PoseSolution:
        return self.inner.solve(pose, state)


class WorldStatePoseSolver:
    """Adapt the existing BoundedPoseSolver to the WorldState adapter protocol."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def solve(self, pose: Pose3D, state: WorldState) -> PoseSolution:
        if not isinstance(state, WorldState):
            raise TypeError("corrected solver requires a WorldState")
        return self.inner.solve(pose, state.robot)


def corrected_pose_solver() -> WorldStatePoseSolver:
    """Return the existing bounded IK solver with corrected URDF registration."""
    return WorldStatePoseSolver(_LazyCorrectedPoseSolver())


def _rotation_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a finite proper rotation matrix to an explicit xyzw quaternion."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("finite 3x3 rotation required for quaternion conversion")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation matrix must be orthonormal")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x, y, z = (
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            )
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x, y, z = (
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x, y, z = (
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            )
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))


def _full_joint_vector(joints: Mapping[str, float]) -> np.ndarray:
    kin = corrected_kinematics()
    vector = np.zeros(len(kin.chain.links), dtype=float)
    for name, value in joints.items():
        if name in kin.idx:
            vector[kin.idx[name]] = float(value)
    return vector


def _corrected_forward(joints: Mapping[str, float]) -> Pose3D:
    matrix = corrected_kinematics().forward(_full_joint_vector(joints))
    return Pose3D(Vec3(*map(float, matrix[:3, 3])), _rotation_to_quaternion(matrix[:3, :3]))


def analytic_surrogate_factory(
    config: WorldMindConfig, inputs: Mapping[str, Any]
) -> AnalyticSurrogate:
    """Bind AnalyticSurrogate to corrected FK/IK and the real safety veto.

    ``check_transition`` is intentionally absent. No full TIAGo swept-arm certificate exists, so
    ``full_system`` hard checks must reject every prediction until one is supplied and reviewed.
    """
    del inputs
    forward = MemoizedForward(
        _corrected_forward, geometry_id=f"corrected-urdf:{CORRECTED_REGISTRATION_SHA256}"
    )
    solver = corrected_pose_solver()
    monitor = runtime_safety_factory(config, {})
    tracking = VelocityLimitedKinematics(dict(_urdf_velocities()), forward)
    return AnalyticSurrogate(
        tracking,
        inverse=lambda pose, robot: JointTarget(
            solver.solve(pose, WorldState(SceneState([]), robot)).joints
        ),
        gripper_speed_m_s=float(SETTINGS.finger_speed),
        gravity_m_s2=9.81,
        support_z_m={config.task.target: config.verification.destination_world[2]},
        validate=lambda action, robot: monitor.validate(action, robot).accepted,
        check_transition=None,
    )


def placement_verifier_factory(
    config: WorldMindConfig,
    destination_base: Pose3D,
    rest_center_base: Pose3D,
    inputs: Mapping[str, Any],
) -> PlacementVerifier:
    """Bind the existing observed-only verifier with frozen tolerances from config."""
    del inputs
    return PlacementVerifier(
        destination_base=destination_base,
        rest_center_base=rest_center_base,
        up_axis_base=config.verification.up_axis_base,
        home_joints=config.verification.home_joints,
        distance_tolerance_m=config.verification.distance_tolerance_m,
        height_tolerance_m=config.verification.height_tolerance_m,
        tilt_limit_rad=config.verification.tilt_limit_rad,
        home_tolerance_rad=config.verification.home_tolerance_rad,
    )


def finite_chunk_check(state: WorldState, chunk: Any, trajectory: Any) -> bool:
    """Mechanical validity check; it claims no model, safety margin or scientific result."""
    del state, trajectory
    actions = getattr(chunk, "actions", None)
    if not isinstance(actions, Sequence) or not actions:
        return False
    for action in actions:
        if isinstance(action, JointTarget):
            if not all(isinstance(name, str) and name for name in action.q) or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in action.q.values()
            ):
                return False
        elif isinstance(action, GripperCmd):
            if not isinstance(action.opening, (int, float)) or not math.isfinite(action.opening):
                return False
        else:
            return False
    return True


def _inverse_pose(pose: Pose3D) -> Pose3D:
    quaternion = np.asarray(pose.orientation, dtype=np.float64)
    position = pose.position.as_array()
    if quaternion.shape != (4,) or not np.isfinite(np.r_[quaternion, position]).all():
        raise ValueError("finite xyzw pose required for frame conversion")
    if not math.isclose(float(quaternion @ quaternion), 1.0, abs_tol=1e-8):
        raise ValueError("unit xyzw pose required for frame conversion")
    inverse = np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]))
    translated = rotate(inverse, -position)
    return Pose3D(
        Vec3(*map(float, translated)),
        (float(inverse[0]), float(inverse[1]), float(inverse[2]), float(inverse[3])),
    )


def corrected_scripted_waypoints(
    config: WorldMindConfig, state: WorldState, task: Any
) -> tuple[ScriptedWaypoint, ...]:
    """Derive baseline geometry from the oracle-initialized target and world-anchored destination.

    The target pose is supplied by the declared oracle state adapter; this function only converts
    it and the frozen destination into the base frame. It never calls the backend.
    """
    if state.robot.base_pose_world is None:
        raise ValueError("corrected waypoints require an explicit world-to-base transform")
    target = next((obj for obj in state.scene.objects if obj.object_id == task.target), None)
    if target is None:
        raise ValueError(f"corrected waypoints require {task.target} in the observed scene")
    base_inverse = _inverse_pose(state.robot.base_pose_world)
    destination_base = compose(
        base_inverse,
        Pose3D(Vec3(*config.verification.destination_world), _IDENTITY_QUAT),
    )
    anchors = {"target": target.pose.position, "destination": destination_base.position}
    waypoints: list[ScriptedWaypoint] = []
    for label, hold_steps, anchor, z_offset in _CORRECTED_STATIONS:
        position = anchors[anchor]
        waypoints.append(
            ScriptedWaypoint(
                label,
                hold_steps,
                pose=Pose3D(
                    Vec3(
                        float(position.x),
                        float(position.y),
                        float(position.z + z_offset),
                    ),
                    _IDENTITY_QUAT,
                ),
            )
        )
    waypoints.append(
        ScriptedWaypoint("RELEASE", 1, gripper_opening=float(SETTINGS.gripper_opening_limits[1]))
    )
    if sum(item.hold_steps for item in waypoints) != config.pipeline.horizon_steps:
        raise ValueError("corrected waypoint cadence does not sum to the frozen horizon")
    return tuple(waypoints)


def runtime_preflight_probe(config: WorldMindConfig, bindings: WorldMindBindings) -> dict[str, str]:
    """Report inert backend-boundary facts; no Webots process or IK solve is started."""

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    limits = runtime_safety_limits()
    solver = bindings.pose_solvers.get(config.adapter.pose_solver_binding)
    waypoint_factory = bindings.waypoint_factories.get(config.adapter.waypoints_binding)
    if not callable(getattr(solver, "solve", None)):
        raise ValueError("registered pose solver cannot solve")
    if not callable(waypoint_factory):
        raise ValueError("registered waypoint factory is not callable")
    return {
        "robot_config_sha256": digest(SETTINGS_PATH),
        "safety_config_sha256": digest(config.safety.config_path),
        "urdf_sha256": digest(URDF_PATH),
        "corrected_registration_sha256": digest(corrected_registration_path()),
        "pose_solver": type(solver).__name__,
        "safety_limits": (
            f"joints={len(limits.joint_position)} "
            f"workspace_min={limits.workspace_min} workspace_max={limits.workspace_max}"
        ),
        "waypoint_factory": getattr(waypoint_factory, "__name__", type(waypoint_factory).__name__),
        "sweep_certificate": "absent",
    }


def production_worldmind_bindings() -> WorldMindBindings:
    """Return every concrete binding that exists without inventing a scientific component.

    Eq. 3.8 component registrations, weights, ranking direction, risk thresholds and calibration
    evaluators are absent. A ready config referencing them therefore still fails closed.
    """
    return WorldMindBindings(
        components={},
        hard_checks={FINITE_CHUNK_CHECK: finite_chunk_check},
        surrogates={ANALYTIC_SURROGATE_BINDING: analytic_surrogate_factory},
        safety={RUNTIME_SAFETY_BINDING: runtime_safety_factory},
        verification={PLACEMENT_VERIFIER_BINDING: placement_verifier_factory},
        pose_solvers={CORRECTED_POSE_SOLVER_BINDING: corrected_pose_solver()},
        waypoint_sets={},
        waypoint_factories={CORRECTED_WAYPOINTS_BINDING: corrected_scripted_waypoints},
        calibration_evaluators={},
        inputs={
            "surrogate": {name: name for name in SURROGATE_INPUTS},
            "safety": {name: name for name in SAFETY_INPUTS},
            "verification": {name: name for name in VERIFICATION_INPUTS},
        },
    )
