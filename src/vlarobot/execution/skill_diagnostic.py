"""Bounded, explicit single-skill probes on the first TRAIN clear scene."""

from contextlib import nullcontext
from copy import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vlarobot.backends.webots.config import (
    CAN_RADIUS_M,
    CLOSE_ALIGN_TOL,
    GRIPPER_OPEN,
    LIFT_MIN,
    SETTINGS,
)
from vlarobot.backends.worldmind_bindings import corrected_kinematics
from vlarobot.core.types import GripperCmd, JointTarget, Vec3
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import WebotsSession, launch_webots_session
from vlarobot.execution.skill_proxy import command_proxy
from vlarobot.execution.skills import (
    V4_REGISTRATION_RECORD,
    Skill,
    SkillAction,
    SkillExecutor,
    require_v4_registration,
    v4_simulator_environment,
)

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
    seed: int = 0,
    session: WebotsSession | None = None,
    standoff: bool = False,
) -> dict[str, Any]:
    """Issue each setup and test action explicitly, keeping one observed result per dispatch."""
    if imperfect_approach and (empty_grasp or skill is not Skill.APPROACH):
        raise ValueError("Imperfect approach requires an isolated PREGRASP probe.")
    if not 0 <= scene_index < 8:
        raise ValueError("Use a TRAIN base scene index from 0 through 7.")
    entry = build_episode_plan("train", seed, n=8)[scene_index]
    out.mkdir(parents=True, exist_ok=False)
    probes = (
        _OFF_EXPERT
        if empty_grasp
        else (skill,)
        if skill is Skill.APPROACH
        else _PROBE_ORDER[: _PROBE_ORDER.index(skill) + 1]
    )
    from vlarobot.execution.skill_live import grasp_evidence, measured_attachment
    from vlarobot.execution.worldmind_pipeline import _inverse_pose
    from vlarobot.world_model.rollout import compose

    if standoff and skill is not Skill.APPROACH:
        raise ValueError("Standoff requires an isolated approach.")
    rows: list[dict[str, Any]] = []
    context = (
        nullcontext(session)
        if session is not None
        else launch_webots_session(
            webots=webots, trace_dir=out, headless=True, extra_env=v4_simulator_environment()
        )
    )
    with context as active:
        backend = active.client
        observation = backend.reset(entry.scene, entry.seed)
        require_v4_registration(observation)
        executor = SkillExecutor(entry.seed, observation, kinematics=copy(corrected_kinematics()))
        initial_can = backend.get_ground_truth().object_poses["CAN"]
        attachment_offset = None
        for index, selected in enumerate(probes):
            before_pose = backend.get_ground_truth().object_poses["CAN"]
            before = before_pose.position
            gap_before = executor._tip_to_can_xy_m(backend, observation)
            target = None
            if selected in _CARTESIAN:
                targets = executor._base_frame_targets(backend, observation)
                x, y, z = map(float, targets[_CARTESIAN[selected]])
                target = Vec3(
                    x,
                    y + (CAN_RADIUS_M if imperfect_approach else 0.0),
                    z + (CAN_RADIUS_M if standoff else 0.0),
                )
            audits = []
            audit = command_proxy(executor.monitor, selected, audits.append) if proxy else None
            if (
                proxy
                and selected in (Skill.OPEN, Skill.RELEASE)
                and before.z - initial_can.position.z >= LIFT_MIN
            ):
                break
            if proxy and selected in (Skill.LIFT, Skill.TRANSFER, Skill.LOWER, Skill.RELEASE):
                anchor = executor._base_frame_targets(backend, observation)["grasp"]
                base_pose = observation.robot.base_pose_world
                can_base = (
                    None if base_pose is None else compose(_inverse_pose(base_pose), before_pose)
                )
                if not grasp_evidence(
                    observation,
                    anchor,
                    measured_can_base=can_base,
                    attachment_offset=attachment_offset,
                ):
                    break
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
            final_pose = backend.get_ground_truth().object_poses["CAN"]
            after = final_pose.position
            if result.settled:
                observed = measured_attachment(
                    observation,
                    result.observation,
                    before,
                    after,
                    previous_offset=attachment_offset,
                )
                if observed is not None:
                    attachment_offset = observed
                if result.observation.robot.gripper_opening >= GRIPPER_OPEN - 0.002:
                    attachment_offset = None
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
                "observed_attachment_tool_m": None
                if attachment_offset is None
                else attachment_offset.tolist(),
                "robot_before": robot_before,
                "tip_can_xy_before_m": gap_before,
                "tip_can_xy_after_m": tip_gap,
                "can_orientation_after_xyzw": final_pose.orientation,
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
            **V4_REGISTRATION_RECORD,
            "scene": entry.scene.name,
            "seed": entry.seed,
            "probe": "empty-grasp-recovery" if empty_grasp else skill.value,
            "imperfect_approach": imperfect_approach,
            "standoff": standoff,
            "initial_can_world_m": asdict(initial_can.position),
            "independent_final_metrics": placement,
            "place_target_world_m": SETTINGS.place_world,
            "settled": len(rows) == len(probes) and all(row["settled"] for row in rows),
            "isolated": len(rows) == len(probes) and all(row["isolated"] for row in rows),
            "rows": rows,
        }
        write_json(out / "skill-result.json", summary)
    return summary
