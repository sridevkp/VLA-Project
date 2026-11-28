"""Bounded, explicit single-skill probes on the first TRAIN clear scene."""

from dataclasses import asdict
from pathlib import Path
from sys import executable
from typing import Any

from vlarobot.backends.webots.config import CAN_RADIUS_M, CLOSE_ALIGN_TOL, GRIPPER_OPEN, SETTINGS
from vlarobot.core.types import GripperCmd, JointTarget, Vec3
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import launch_webots_session
from vlarobot.execution.skill_proxy import command_proxy
from vlarobot.execution.skills import Skill, SkillAction, SkillExecutor

_CARTESIAN = {
    Skill.APPROACH: "pregrasp",
    Skill.DESCEND: "grasp",
    Skill.LIFT: "lift",
    Skill.TRANSFER: "transit",
    Skill.LOWER: "place_descend",
    Skill.RETREAT: "retreat",
}
_PROBE_ORDER = (
    Skill.OPEN,
    Skill.APPROACH,
    Skill.DESCEND,
    Skill.CLOSE,
    Skill.LIFT,
    Skill.TRANSFER,
    Skill.LOWER,
    Skill.RELEASE,
    Skill.RETREAT,
)
_OFF_EXPERT = (Skill.OPEN, Skill.CLOSE, Skill.OPEN)


def run_skill_diagnostic(
    out: Path,
    webots: str | None,
    skill: Skill,
    *,
    empty_grasp: bool = False,
    imperfect_approach: bool = False,
    scene_index: int = 0,
    proxy: bool = False,
) -> dict[str, Any]:
    """Issue each setup and test action explicitly, keeping one observed result per dispatch."""
    if imperfect_approach and (empty_grasp or skill is not Skill.APPROACH):
        raise ValueError("Imperfect approach requires an isolated PREGRASP probe.")
    if scene_index not in (0, 1, 2):
        raise ValueError("Use one of the three bounded TRAIN diagnostic scene indices.")
    entry = build_episode_plan("train", 0, n=3)[scene_index]
    out.mkdir(parents=True, exist_ok=False)
    probes = (
        _OFF_EXPERT
        if empty_grasp
        else (skill,)
        if skill is Skill.APPROACH
        else _PROBE_ORDER[: _PROBE_ORDER.index(skill) + 1]
    )
    rows: list[dict[str, Any]] = []
    with launch_webots_session(
        webots=webots, trace_dir=out, headless=True, extra_env={"WEBOTS_PYTHON": executable}
    ) as session:
        backend = session.client
        observation = backend.reset(entry.scene, entry.seed)
        executor = SkillExecutor(entry.seed, observation)
        initial_can = backend.get_ground_truth().object_poses["CAN"]
        for index, selected in enumerate(probes):
            before = backend.get_ground_truth().object_poses["CAN"].position
            target = None
            if selected in _CARTESIAN:
                targets = executor._base_frame_targets(backend, observation)
                x, y, z = map(float, targets[_CARTESIAN[selected]])
                target = Vec3(x, y + (CAN_RADIUS_M if imperfect_approach else 0.0), z)
            audits = []
            audit = command_proxy(executor.monitor, selected, audits.append) if proxy else None
            result = executor.run_skill(
                backend, observation, SkillAction(selected, target), audit=audit
            )
            commands = result.dispatched
            if target is None:
                commanded_opening = 0.0 if selected is Skill.CLOSE else GRIPPER_OPEN
                isolated = (
                    len(commands) == 1
                    and isinstance(commands[0], GripperCmd)
                    and commands[0].opening == commanded_opening
                )
            else:
                isolated = bool(commands) and all(isinstance(cmd, JointTarget) for cmd in commands)
            after = backend.get_ground_truth().object_poses["CAN"].position
            robot_before = asdict(observation.robot)
            robot_before.pop("diagnostics")
            tip_gap = executor._tip_to_can_xy_m(backend, result.observation)
            row = {
                "skill": selected.value,
                "role": "probe" if empty_grasp or index == len(probes) - 1 else "setup",
                "target_base_m": None if target is None else asdict(target),
                "settled": result.settled,
                "isolated": isolated,
                "predictive_audits": [asdict(item) for item in audits],
                "commands": [asdict(command) for command in commands],
                "measurement_source": "simulator-observed",
                "can_before_world_m": asdict(before),
                "can_after_world_m": asdict(after),
                "can_rise_m": result.can_rise_m,
                "robot_before": robot_before,
                "tip_can_xy_after_m": tip_gap,
                "empty_grasp_verified": (
                    selected is Skill.CLOSE and result.settled and tip_gap > CLOSE_ALIGN_TOL
                    if empty_grasp
                    else None
                ),
                "tool_before_base_m": asdict(observation.robot.ee_pose.position),
                "tool_after_base_m": asdict(result.observation.robot.ee_pose.position),
                "gripper_before_m": result.gripper_before_m,
                "gripper_after_m": result.observation.robot.gripper_opening,
                "gripper_holding_after": result.observation.robot.gripper_holding,
                "vetoes": result.vetoes,
                "failure_reason": result.failure_reason,
            }
            rows.append(row)
            write_json(out / f"skill-{index:02d}.json", row)
            if not result.settled or not isolated:
                break
            observation = result.observation
        final_can = backend.get_ground_truth().object_poses["CAN"]
        placement = final_can_outcome(
            initial_can,
            final_can,
            place_world_m=SETTINGS.place_world,
            distance_tolerance_m=SETTINGS.place_tolerance_m,
            height_tolerance_m=SETTINGS.can_place_z_tolerance_m,
            tilt_limit_deg=SETTINGS.can_tilt_limit_deg,
            home_reached=False,
            failure_state=None,
            failure_reason=None,
        )
        summary = {
            "mode": "oracle-assisted",
            "split": "train",
            "scene": entry.scene.name,
            "seed": entry.seed,
            "probe": "empty-grasp-recovery" if empty_grasp else skill.value,
            "imperfect_approach": imperfect_approach,
            "initial_can_world_m": asdict(initial_can.position),
            "independent_final_metrics": placement,
            "place_target_world_m": SETTINGS.place_world,
            "settled": len(rows) == len(probes) and all(row["settled"] for row in rows),
            "isolated": len(rows) == len(probes) and all(row["isolated"] for row in rows),
            "rows": rows,
        }
        write_json(out / "skill-result.json", summary)
    return summary
