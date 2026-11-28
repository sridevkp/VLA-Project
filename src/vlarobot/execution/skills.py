"""Single-action CAN manipulation skills for the oracle-assisted V4 experiment."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from vlarobot.backends.webots.config import ARM_JOINTS
from vlarobot.backends.worldmind_bindings import runtime_safety_limits
from vlarobot.core.interfaces import RobotBackend
from vlarobot.core.types import (
    Action,
    ApplyResult,
    GripperCmd,
    GroundTruth,
    JointTarget,
    Observation,
    Vec3,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_can import ScriptedCanPolicy

HOME_VERIFICATION_TOLERANCE_RAD = 0.01


class Skill(StrEnum):
    APPROACH = "PREGRASP"
    DESCEND = "DESCEND"
    LIFT = "LIFT"
    TRANSFER = "TRANSIT"
    LOWER = "PLACE_DESCEND"
    RETREAT = "RETREAT"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    RELEASE = "RELEASE"
    HOME = "HOME"


EMPIRICAL_SKILLS = tuple(skill for skill in Skill if skill is not Skill.HOME)


@dataclass(frozen=True)
class SkillAction:
    skill: Skill
    target_base: Vec3 | None = None
    home_joints: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.skill is Skill.HOME:
            if self.target_base is not None or set(dict(self.home_joints)) != set(ARM_JOINTS):
                raise ValueError("HOME requires the configured arm joint vector.")
            return
        if self.home_joints:
            raise ValueError("Only HOME accepts a joint vector.")
        gripper_only = self.skill in (Skill.OPEN, Skill.CLOSE, Skill.RELEASE)
        if gripper_only != (self.target_base is None):
            raise ValueError("Cartesian skills require a base-frame target.")


@dataclass(frozen=True)
class SkillOutcome:
    settled: bool
    observation: Observation
    can_rise_m: float
    gripper_before_m: float
    vetoes: tuple[str, ...]
    dispatched: tuple[Action, ...]
    failure_reason: str | None


CommandAudit = Callable[[Action, Observation, GroundTruth, bool], ApplyResult]


class _CheckedBackend:
    def __init__(
        self,
        backend: RobotBackend,
        monitor: SafetyMonitor,
        observation: Observation,
        audit: CommandAudit | None = None,
    ) -> None:
        self.backend = backend
        self.monitor = monitor
        self.observation = observation
        self.vetoes: list[str] = []
        self.dispatched: list[Action] = []
        self.audit = audit
        self.terminal = True

    def apply(self, action: Action) -> ApplyResult:
        verdict = self.monitor.validate(action, self.observation.robot)
        if not verdict.accepted:
            self.vetoes.append(verdict.reason or "safety monitor veto")
            return verdict
        if self.audit is not None:
            audited_q = action.q.copy() if isinstance(action, JointTarget) else None
            verdict = self.audit(
                action, self.observation, self.backend.get_ground_truth(), self.terminal
            )
            if isinstance(action, JointTarget) and action.q != audited_q:
                verdict = ApplyResult(False, "joint command changed during predictive audit")
            if not verdict.accepted:
                self.vetoes.append(verdict.reason or "predictive proxy veto")
                return verdict
        self.dispatched.append(action)
        return self.backend.apply(action)

    def step(self, n_steps: int = 1) -> Observation:
        self.observation = self.backend.step(n_steps)
        return self.observation

    def get_ground_truth(self) -> GroundTruth:
        return self.backend.get_ground_truth()


class SkillExecutor(ScriptedCanPolicy):
    """Reuse expert primitives with an explicit one-skill entry point and command guard."""

    def __init__(self, seed: int, initial: Observation) -> None:
        super().__init__()
        self.kinematics.set_seed(seed)
        self._reset_base_pose_world = initial.robot.base_pose_world
        self.monitor = SafetyMonitor(runtime_safety_limits())

    def _joint_waypoint_settled(self, measured, commanded, tolerance, *, terminal_home):
        if terminal_home:
            return all(
                abs(measured[name] - target) <= HOME_VERIFICATION_TOLERANCE_RAD
                for name, target in commanded.items()
            )
        return super()._joint_waypoint_settled(
            measured, commanded, tolerance, terminal_home=terminal_home
        )

    def execute(self, backend: RobotBackend, seed: int) -> None:
        raise TypeError("Use run_skill with an explicit SkillAction.")

    def run_skill(
        self,
        backend: RobotBackend,
        observation: Observation,
        action: SkillAction,
        *,
        audit: CommandAudit | None = None,
    ) -> SkillOutcome:
        checked = _CheckedBackend(backend, self.monitor, observation, audit)
        self._last_observation = observation
        self._last_failure_reason = None
        self._last_failure_is_safety = False
        self._last_commanded_joint_targets = {
            name: observation.robot.joint_pos[name] for name in ARM_JOINTS
        }
        can_before = self._can_pose(backend.get_ground_truth())[2]
        if action.skill in (Skill.OPEN, Skill.RELEASE):
            settled = self._open_gripper(checked, observation) is not None
        elif action.skill is Skill.CLOSE:
            settled = self._close_gripper(checked)
        elif action.skill is Skill.HOME:
            current = self._full_joint_vector(observation.robot.joint_pos)
            goal = self._full_joint_vector(observation.robot.joint_pos | dict(action.home_joints))
            settled = (
                self._move_to_joint_target(checked, current, goal, action.skill.value) is not None
            )
        else:
            assert action.target_base is not None
            target = action.target_base.as_array()
            if action.skill is Skill.APPROACH:
                q = self._full_joint_vector(observation.robot.joint_pos)
                settled = self._move_to(checked, q, target, action.skill.value) is not None
            else:
                settled = (
                    self._move_linear(checked, observation, target, action.skill.value) is not None
                )
        return SkillOutcome(
            settled=settled,
            observation=checked.observation,
            can_rise_m=float(self._can_pose(backend.get_ground_truth())[2] - can_before),
            gripper_before_m=observation.robot.gripper_opening,
            vetoes=tuple(checked.vetoes),
            dispatched=tuple(checked.dispatched),
            failure_reason=self._last_failure_reason,
        )

    def _command_and_wait(self, backend, *args, phase_terminal=None, **kwargs):
        backend.terminal = phase_terminal is not False
        return super()._command_and_wait(backend, *args, phase_terminal=phase_terminal, **kwargs)

    def _close_gripper(self, backend: _CheckedBackend) -> bool:
        response = backend.apply(GripperCmd(0.0, force=50.0))
        if not response.accepted:
            self._last_failure_reason = response.reason or "gripper command rejected"
            return False
        previous = backend.observation.robot.gripper_opening
        still_count = 0
        for index in range(300):
            observation = backend.step()
            self._track_observation(observation, state="CLOSE")
            if self._base_moved(observation, "CLOSE"):
                return False
            opening = observation.robot.gripper_opening
            change = abs(opening - previous)
            if index == 0 and opening <= 0.001 and change > 0.01:
                return True
            still_count = still_count + 1 if change < 2e-5 else 0
            if (index > 10 and still_count >= 15) or (opening <= 0.001 and still_count >= 3):
                return True
            previous = opening
        self._last_failure_reason = "gripper close did not settle"
        return False
