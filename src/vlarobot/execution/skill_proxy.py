"""Owner-scoped V4 tool-envelope audit over a lockstep nominal skill trajectory."""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vlarobot.backends.webots.config import (
    CAN_HALF_HEIGHT_M,
    CAN_RADIUS_M,
    PLACE_TOLERANCE_M,
    PLACE_WORLD,
    SETTINGS,
    TIME_STEP,
)
from vlarobot.backends.worldmind_bindings import _corrected_forward
from vlarobot.core.types import (
    Action,
    ActionChunk,
    ApplyResult,
    EEPoseTarget,
    GripperCmd,
    GroundTruth,
    JointTarget,
    ObjectState,
    Observation,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_can import _ik_tolerance_for_state
from vlarobot.execution.skills import Skill
from vlarobot.execution.worldmind_pipeline import _inverse_pose
from vlarobot.execution.worldmind_proxy_binding import (
    _MODEL_PINS,
    _matched_proxy_evidence,
    proxy_scene,
)
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate, Attachment, Trajectory, compose, rotate

PROFILE_ID = "v4-train-tool-envelope-proxy-v1"
V1_CLEARANCE_SHA256 = "6ea88763ad27a769943aa52132c60e785e4aa4e811899e34473c6f6cec17bed4"
_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_PATH = _ROOT / "configs/v4_skill_proxy.json"


def _clearance_configuration() -> tuple[dict, str]:
    raw = _PROFILE_PATH.read_bytes()
    profile = json.loads(raw)
    if (
        profile["identity"] != PROFILE_ID
        or profile["split"] != "train"
        or profile["full_arm_sweep_coverage"] != "unverified"
        or profile["clearance_sha256"] != V1_CLEARANCE_SHA256
    ):
        raise ValueError("unexpected V4 proxy scope")
    clearance = (_ROOT / profile["clearance_file"]).read_bytes()
    if hashlib.sha256(clearance).hexdigest() != V1_CLEARANCE_SHA256:
        raise ValueError("V4 proxy clearance profile hash mismatch")
    world = (_ROOT / profile["world_file"]).read_bytes()
    if hashlib.sha256(world).hexdigest() != profile["world_sha256"]:
        raise ValueError("V4 proxy world hash mismatch")
    for name, digest in _MODEL_PINS.items():
        model = _ROOT / "assets/webots/protos/r2025a" / name
        if hashlib.sha256(model.read_bytes()).hexdigest() != digest:
            raise ValueError(f"V4 proxy model hash mismatch: {name}")
    return json.loads(clearance), hashlib.sha256(raw).hexdigest()


# Recovery is local to the measured transit-height deficit.
CLEARANCE_MAX_RISE_M = 0.020
CLEARANCE_TARGET_MARGIN_M = 0.006
CLEARANCE_MAX_LATERAL_M = 0.004
CLEARANCE_TRACKING_SLACK_M = 0.001


def transit_floor_m() -> float:
    settings, _ = _clearance_configuration()
    return float(settings["table_top_world_z_m"] + settings["table_margin_m"][Skill.TRANSFER.value])


def clearance_path_passes(
    points: np.ndarray, origin: np.ndarray, floor: float, terminal: bool
) -> bool:
    """Check a bounded lift under LIFT geometry and require transit height at its endpoint."""
    return bool(
        np.all(np.isfinite(points))
        and np.all(np.linalg.norm((points - origin)[:, :2], axis=1) <= CLEARANCE_MAX_LATERAL_M)
        and np.min(points[:, 2]) >= origin[2] - CLEARANCE_TRACKING_SLACK_M
        and np.max(points[:, 2]) <= origin[2] + CLEARANCE_MAX_RISE_M
        and np.min(points[:, 2]) >= points[0, 2] - CLEARANCE_TRACKING_SLACK_M
        and (not terminal or points[-1, 2] >= floor)
    )


def placement_clearance_offset_m() -> float:
    """Use half the pinned terminal tolerance for a higher placement alternative."""
    settings, _ = _clearance_configuration()
    return float(settings["contact_height_m"]["PLACE_DESCEND"]["tolerance_m"]) / 2


@dataclass(frozen=True)
class SkillProxyAssessment:
    profile: str
    profile_sha256: str
    clearance_sha256: str
    full_arm_sweep_verified: bool
    strict_reasons: tuple[str, ...]
    checks: tuple[tuple[str, bool], ...]
    audited_joint_target: tuple[tuple[str, float], ...] | None = None
    audited_gripper_opening_m: float | None = None
    predicted_tool_base_m: tuple[Vec3, ...] = ()


def assess_skill_proxy(
    state: WorldState,
    truth: GroundTruth,
    skill: Skill,
    chunk: ActionChunk,
    surrogate: AnalyticSurrogate,
    monitor: SafetyMonitor,
    *,
    attachment: Attachment | None = None,
) -> SkillProxyAssessment:
    """Audit an aligned, dense skill path under the V1 tool-envelope scope only."""
    trajectory = surrogate.predict_trajectory(state, chunk, attachment=attachment)
    return _assess_path(state, truth, skill, chunk, trajectory, monitor)


def _assess_path(
    state: WorldState,
    truth: GroundTruth,
    skill: Skill,
    chunk: ActionChunk,
    trajectory: Trajectory,
    monitor: SafetyMonitor,
    *,
    terminal: bool = True,
    exact_joint_commands: bool = False,
    clearance_origin_world: np.ndarray | None = None,
) -> SkillProxyAssessment:
    settings, profile_sha256 = _clearance_configuration()
    command_type = GripperCmd if skill in (Skill.OPEN, Skill.CLOSE, Skill.RELEASE) else EEPoseTarget
    if exact_joint_commands and command_type is EEPoseTarget:
        command_type = JointTarget
    complete = (
        bool(chunk.actions)
        and len(trajectory.states) == len(chunk.actions) + 1
        and math.isclose(chunk.dt, TIME_STEP / 1000.0)
        and math.isclose(trajectory.dt_s, chunk.dt)
        and all(isinstance(action, command_type) for action in chunk.actions)
    )
    checks = [
        ("dense_skill_prediction", complete),
        (
            "predictive_model_constraints",
            complete and not (set(trajectory.hard_reasons) - {"unverified-arm-sweep"}),
        ),
    ]
    if complete:
        config = SimpleNamespace(verification=SimpleNamespace(destination_world=PLACE_WORLD))
        scene, _, goal = proxy_scene(state, truth, config)
        phase = Skill.LIFT.value if skill is Skill.CLEARANCE_LIFT else skill.value
        evidence = _matched_proxy_evidence(
            scene,
            chunk,
            trajectory,
            monitor,
            settings,
            active_phase=phase,
            phase_terminal=terminal,
        )
        checks.extend(
            (
                (
                    "independent_command_limits",
                    all(
                        monitor.validate(action, prior.robot).accepted
                        for action, prior in zip(chunk.actions, trajectory.states[:-1], strict=True)
                    ),
                ),
                ("joint_limits", evidence.joint_limits_pass is True),
                (
                    "ik_and_terminal_tracking",
                    evidence.ik_pass is True
                    and bool(
                        command_type is not EEPoseTarget
                        or np.linalg.norm(
                            trajectory.states[-1].robot.ee_pose.position.as_array()
                            - chunk.actions[-1].pose.position.as_array()
                        )
                        <= _ik_tolerance_for_state(phase)
                    ),
                ),
                ("table", evidence.geometric_hard_pass[0] is True),
                ("modeled_obstacles", evidence.geometric_hard_pass[1] is True),
            )
        )
        if skill is Skill.CLEARANCE_LIFT:
            base = state.robot.base_pose_world
            valid = False
            if base is not None and clearance_origin_world is not None:
                world_points = np.array(
                    [
                        base.position.as_array()
                        + rotate(
                            np.asarray(base.orientation), item.robot.ee_pose.position.as_array()
                        )
                        for item in trajectory.states
                    ]
                )
                floor = (
                    settings["table_top_world_z_m"]
                    + settings["table_margin_m"][Skill.TRANSFER.value]
                )
                valid = clearance_path_passes(world_points, clearance_origin_world, floor, terminal)
            checks.append(("bounded_vertical_clearance", valid))
        if skill is Skill.LOWER and terminal:
            error = np.linalg.norm(
                trajectory.states[-1].robot.ee_pose.position.as_array() - goal.as_array()
            )
            checks.append(("placement", bool(error <= PLACE_TOLERANCE_M)))
    return SkillProxyAssessment(
        PROFILE_ID,
        profile_sha256,
        V1_CLEARANCE_SHA256,
        False,
        trajectory.hard_reasons,
        tuple(checks),
    )


def command_proxy(
    monitor: SafetyMonitor,
    skill: Skill,
    record: Callable[[SkillProxyAssessment], None],
    *,
    record_accepted_forecast: Callable[[Trajectory], None] | None = None,
    clearance_origin_world: np.ndarray | None = None,
) -> Callable[[Action, Observation, GroundTruth, bool], ApplyResult]:
    """Reforecast the exact current command before dispatch under the separate proxy scope."""
    dt = TIME_STEP / 1000.0
    velocities = monitor.limits.joint_velocity
    tracking = VelocityLimitedKinematics(velocities, _corrected_forward)

    def check(
        action: Action,
        observation: Observation,
        truth: GroundTruth,
        terminal: bool,
    ) -> ApplyResult:
        state_robot = observation.robot
        base = state_robot.base_pose_world
        if base is None or set(truth.object_poses) != {"CAN", "HAM", "APP"}:
            return ApplyResult(False, "missing proxy scene geometry")
        if isinstance(action, JointTarget):
            seconds = max(
                abs(target - state_robot.joint_pos[name]) / velocities[name]
                for name, target in action.q.items()
            )
        elif isinstance(action, GripperCmd):
            seconds = abs(action.opening - state_robot.gripper_opening) / SETTINGS.finger_speed
        else:
            return ApplyResult(False, "unsupported skill command for proxy")
        count = max(1, math.ceil(seconds / dt))
        can_pose = compose(_inverse_pose(base), truth.object_poses["CAN"])
        can = ObjectState(
            "CAN",
            "can",
            can_pose,
            Vec3(2 * CAN_RADIUS_M, 2 * CAN_RADIUS_M, 2 * CAN_HALF_HEIGHT_M),
            1.0,
        )
        state = WorldState(SceneState([can]), replace(state_robot))
        surrogate = AnalyticSurrogate(
            tracking,
            inverse=lambda pose, robot: None,
            gripper_speed_m_s=SETTINGS.finger_speed,
            gravity_m_s2=9.81,
            support_z_m={"CAN": can_pose.position.z - CAN_HALF_HEIGHT_M},
            validate=lambda request, robot: monitor.validate(request, robot).accepted,
        )
        chunk = ActionChunk([action] * count, dt, "vla")
        trajectory = surrogate.predict_trajectory(state, chunk)
        assessment = _assess_path(
            state,
            truth,
            skill,
            chunk,
            trajectory,
            monitor,
            terminal=terminal,
            exact_joint_commands=True,
            clearance_origin_world=clearance_origin_world,
        )
        record(
            replace(
                assessment,
                audited_joint_target=tuple(sorted(action.q.items()))
                if isinstance(action, JointTarget)
                else None,
                audited_gripper_opening_m=action.opening
                if isinstance(action, GripperCmd)
                else None,
                predicted_tool_base_m=tuple(
                    item.robot.ee_pose.position for item in trajectory.states
                ),
            )
        )
        failed = [name for name, result in assessment.checks if result is not True]
        if not failed and record_accepted_forecast is not None:
            record_accepted_forecast(trajectory)
        return ApplyResult(not failed, ", ".join(failed) if failed else None)

    return check
