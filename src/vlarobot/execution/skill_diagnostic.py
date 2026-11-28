"""Bounded single-skill Webots diagnostic on the first approved TRAIN clear scene."""

from pathlib import Path
from sys import executable
from typing import Any

from vlarobot.core.types import GripperCmd, JointTarget, Vec3
from vlarobot.eval.runner import build_episode_plan
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import launch_webots_session
from vlarobot.execution.skills import Skill, SkillAction, SkillExecutor


def run_skill_diagnostic(out: Path, webots: str | None, skill: Skill) -> dict[str, Any]:
    if skill not in (Skill.OPEN, Skill.APPROACH):
        raise ValueError("The reset-scene diagnostic supports OPEN and PREGRASP.")
    entry = build_episode_plan("train", 0, n=1)[0]
    out.mkdir(parents=True, exist_ok=False)
    with launch_webots_session(
        webots=webots, trace_dir=out, headless=True, extra_env={"WEBOTS_PYTHON": executable}
    ) as session:
        backend = session.client
        initial = backend.reset(entry.scene, entry.seed)
        executor = SkillExecutor(entry.seed, initial)
        target = None
        if skill is Skill.APPROACH:
            target = Vec3(*map(float, executor._base_frame_targets(backend, initial)["pregrasp"]))
        action = SkillAction(skill, target)
        result = executor.run_skill(backend, initial, action)
        commands = result.dispatched
        if skill is Skill.OPEN:
            isolated = len(commands) == 1 and isinstance(commands[0], GripperCmd)
        else:
            isolated = bool(commands) and all(isinstance(cmd, JointTarget) for cmd in commands)
        summary = {
            "mode": "oracle-assisted",
            "split": "train",
            "scene": entry.scene.name,
            "seed": entry.seed,
            "skill": action.skill.value,
            "target_base_m": None if target is None else target.as_array().tolist(),
            "settled": result.settled,
            "isolated": isolated,
            "commands": [type(command).__name__ for command in commands],
            "can_rise_m": result.can_rise_m,
            "gripper_before_m": result.gripper_before_m,
            "gripper_after_m": result.observation.robot.gripper_opening,
            "vetoes": result.vetoes,
            "failure_reason": result.failure_reason,
        }
        write_json(out / "skill-result.json", summary)
    return summary
