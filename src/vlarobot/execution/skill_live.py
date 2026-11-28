"""Bounded oracle-assisted learned clear-scene TRAIN diagnostic."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from sys import executable

import numpy as np

from vlarobot.backends.webots.config import (
    CAN_RADIUS_M,
    CLOSE_ALIGN_TOL,
    GRIPPER_OPEN,
    LIFT_MIN,
    PLACE_TOLERANCE_M,
    PLACE_WORLD,
    SETTINGS,
    TIME_STEP,
)
from vlarobot.backends.worldmind_bindings import _corrected_forward
from vlarobot.core.types import (
    Action,
    ApplyResult,
    GripperCmd,
    GroundTruth,
    JointTarget,
    ObjectState,
    Observation,
    Pose3D,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import launch_webots_session
from vlarobot.execution.skill_empirical import FORECAST_PROFILE, forecast_candidate
from vlarobot.execution.skill_proxy import PROFILE_ID, command_proxy
from vlarobot.execution.skill_selection import UtilityModel, enumerate_skills
from vlarobot.execution.skills import Skill, SkillAction, SkillExecutor
from vlarobot.execution.worldmind_pipeline import _inverse_pose
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.rollout import compose

_TARGETS = {
    Skill.APPROACH: "pregrasp",
    Skill.DESCEND: "grasp",
    Skill.LIFT: "lift",
    Skill.TRANSFER: "transit",
    Skill.LOWER: "place_descend",
    Skill.RETREAT: "retreat",
}


class _PreviewBackend:
    """Nominal waypoint settle for offline preview; every transition is forecast separately."""

    def __init__(self, observation: Observation, truth: GroundTruth) -> None:
        self.observation = observation
        self.truth = truth

    def get_ground_truth(self) -> GroundTruth:
        return self.truth

    def apply(self, action: Action) -> ApplyResult:
        robot = self.observation.robot
        if isinstance(action, JointTarget):
            joints = robot.joint_pos | action.q
            robot = replace(robot, joint_pos=joints, ee_pose=_corrected_forward(joints))
        elif isinstance(action, GripperCmd):
            robot = replace(robot, gripper_opening=action.opening)
        else:
            return ApplyResult(False, "unsupported preview command")
        self.observation = replace(self.observation, robot=robot)
        return ApplyResult(True)

    def step(self, n_steps: int = 1) -> Observation:
        self.observation = replace(
            self.observation, t=self.observation.t + n_steps * TIME_STEP / 1000
        )
        return self.observation


def measured_attachment(
    before: Observation,
    after: Observation,
    can_before_world: Vec3,
    can_after_world: Vec3,
) -> np.ndarray | None:
    """Establish a tool-relative CAN offset only after observed upward co-motion."""
    old_base, new_base = before.robot.base_pose_world, after.robot.base_pose_world
    if old_base is None or new_base is None:
        return None
    if not 0.002 < after.robot.gripper_opening < GRIPPER_OPEN - 0.002:
        return None
    old_tool = compose(old_base, before.robot.ee_pose)
    new_tool = compose(new_base, after.robot.ee_pose)
    tool_delta = new_tool.position.as_array() - old_tool.position.as_array()
    can_delta = can_after_world.as_array() - can_before_world.as_array()
    if can_delta[2] < LIFT_MIN or np.linalg.norm(can_delta - tool_delta) > CAN_RADIUS_M:
        return None
    can_in_tool = compose(_inverse_pose(new_tool), Pose3D(can_after_world, (0.0, 0.0, 0.0, 1.0)))
    return can_in_tool.position.as_array()


def grasp_evidence(
    observation: Observation,
    grasp_base: np.ndarray,
    *,
    measured_can_base: Pose3D | None = None,
    attachment_offset: np.ndarray | None = None,
) -> bool:
    """Require current aperture plus alignment or a still-observed measured attachment."""
    robot = observation.robot
    if robot.gripper_holding is not None:
        return robot.gripper_holding
    if not 0.002 < robot.gripper_opening < GRIPPER_OPEN - 0.002:
        return False
    if attachment_offset is not None and measured_can_base is not None:
        observed_offset = compose(_inverse_pose(robot.ee_pose), measured_can_base).position
        return bool(
            np.linalg.norm(observed_offset.as_array() - attachment_offset) < CLOSE_ALIGN_TOL
        )
    return bool(np.linalg.norm(robot.ee_pose.position.as_array() - grasp_base) < CLOSE_ALIGN_TOL)


def generate_candidates(
    executor: SkillExecutor,
    backend,
    observation: Observation,
    state: WorldState,
    *,
    can_rise_m: float,
    measured_can_base: Pose3D | None = None,
    attachment_offset: np.ndarray | None = None,
) -> tuple[SkillAction, ...]:
    """Use current geometry and aperture, independent of the previous task phase."""
    base = observation.robot.base_pose_world
    if base is None:
        raise ValueError("missing base transform")
    can_world = compose(base, state.scene.objects[0].pose).position.as_array()
    anchors = executor._base_frame_targets(backend, observation, can_world=can_world)
    targets = {skill: Vec3(*map(float, anchors[key])) for skill, key in _TARGETS.items()}
    # Retreat clears a low tool locally. Traversing to the placement hover is a separate skill.
    tool_pose = state.robot.ee_pose.position
    retreat_z = max(tool_pose.z, targets[Skill.APPROACH].z, targets[Skill.RETREAT].z)
    targets[Skill.RETREAT] = replace(tool_pose, z=retreat_z)
    plausible_grasp = grasp_evidence(
        observation,
        anchors["grasp"],
        measured_can_base=measured_can_base,
        attachment_offset=attachment_offset,
    )
    grasp = targets[Skill.DESCEND].as_array()
    tool = tool_pose.as_array()
    if (
        state.robot.gripper_opening < GRIPPER_OPEN - 0.002
        or np.linalg.norm((tool - grasp)[:2]) > CAN_RADIUS_M + CLOSE_ALIGN_TOL
        or tool[2] <= grasp[2]
    ):
        targets.pop(Skill.DESCEND)
    lower = targets[Skill.LOWER].as_array()
    if np.linalg.norm((tool - lower)[:2]) > PLACE_TOLERANCE_M or tool[2] <= lower[2]:
        targets.pop(Skill.LOWER)
    if not plausible_grasp:
        for carry in (Skill.LIFT, Skill.TRANSFER, Skill.LOWER):
            targets.pop(carry, None)
    if can_rise_m < LIFT_MIN:
        targets.pop(Skill.TRANSFER, None)
        targets.pop(Skill.LOWER, None)
    else:
        targets.pop(Skill.LIFT, None)
    state = replace(state, robot=replace(state.robot, gripper_holding=plausible_grasp))
    candidates = enumerate_skills(
        state,
        targets,
        position_tolerance_m=0.02,
        open_tolerance_m=0.002,
        open_position_m=GRIPPER_OPEN,
    )
    approach = targets[Skill.APPROACH]
    offset = replace(approach, y=approach.y + CAN_RADIUS_M)
    tool = state.robot.ee_pose.position.as_array()
    alternate = (
        (SkillAction(Skill.APPROACH, offset),)
        if np.linalg.norm(offset.as_array() - tool) > 0.02
        else ()
    )
    return (
        *(
            candidate
            for candidate in candidates
            if candidate.skill is not Skill.RELEASE or can_rise_m < LIFT_MIN
        ),
        *alternate,
    )


def approach_progress_score(
    raw_score: float, candidate: SkillAction, can_base: Vec3, best_gap_m: float | None
) -> float:
    """Discount another approach that cannot improve the measured current approach streak."""
    if candidate.skill is not Skill.APPROACH or best_gap_m is None:
        return raw_score
    target = candidate.target_base
    assert target is not None
    target_gap = float(np.linalg.norm((target.as_array() - can_base.as_array())[:2]))
    remaining_gain = best_gap_m - target_gap
    if remaining_gain > CLOSE_ALIGN_TOL:
        return raw_score
    return min(raw_score, (remaining_gain - CLOSE_ALIGN_TOL) / CAN_RADIUS_M)


def release_progress_score(raw_score: float, candidate: SkillAction, can_world: Vec3) -> float:
    """Treat releasing away from the destination as recovery with no placement credit."""
    if candidate.skill is not Skill.RELEASE:
        return raw_score
    error = float(np.linalg.norm(can_world.as_array() - np.asarray(PLACE_WORLD)))
    return min(raw_score, 0.0) if error > PLACE_TOLERANCE_M else raw_score


def close_progress_score(
    raw_score: float,
    candidate: SkillAction,
    aperture_m: float,
    grasp_available: bool,
) -> float:
    """A settled closed or grasping hand earns no added progress for another close."""
    if candidate.skill is Skill.CLOSE and (aperture_m <= 0.002 or grasp_available):
        return min(raw_score, 0.0)
    return raw_score


def run_learned_smoke(
    out: Path,
    model_path: Path,
    webots: str | None,
    *,
    localizer_path: Path | None = None,
    context: str = "smoke",
    max_decisions: int = 6,
    scene_index: int = 0,
) -> dict:
    """Run bounded receding-horizon decisions with scoped predictive vetoes."""
    if not 1 <= max_decisions <= 16:
        raise ValueError("Bounded diagnostic requires 1–16 decisions")
    if scene_index not in (0, 1, 2):
        raise ValueError("Choose one of the three approved TRAIN diagnostic scenes")
    model_bytes = model_path.read_bytes()
    artifact = json.loads(model_bytes)
    localize = None
    if localizer_path is not None:
        from vlarobot.execution.skill_vision import DinoCanLocalizer

        localize = DinoCanLocalizer(localizer_path)
    if artifact["identity"] != "v4-initial-empirical-nominal-no-context":
        raise ValueError("expected the initial empirical no-context smoke model")
    if context not in {"smoke", "matched-none", "predicted", "shuffled", "controlled"}:
        raise ValueError("unknown utility context mode")
    uses_context = context in {"predicted", "shuffled", "controlled"}
    weights = artifact
    if context != "smoke":
        if artifact.get("forecast_profile") != FORECAST_PROFILE:
            raise ValueError(
                "Refit the matched utility model for the current LIFT forecast profile"
            )
        matched = artifact["measured_only_forecast_comparison"]
        weights = matched["context_model" if uses_context else "matched_no_context_model"]
    model = UtilityModel(
        uses_context,
        np.asarray(weights["mean"]),
        np.asarray(weights["scale"]),
        np.asarray(weights["coefficients"]),
        weights["intercept"],
    )
    gate = RiskGate(
        weights=(1.0, 0.0, 0.0, 0.0),
        maximize=True,
        required_constraints=("proxy",),
        inactive_components=(1, 2, 3),
    )
    entry = build_episode_plan("train", 0, n=3)[scene_index]
    out.mkdir(parents=True, exist_ok=False)
    with launch_webots_session(
        webots=webots, trace_dir=out, headless=True, extra_env={"WEBOTS_PYTHON": executable}
    ) as session:
        backend = session.client
        obs = backend.reset(entry.scene, entry.seed)
        home = obs.robot.joint_pos.copy()
        initial = backend.get_ground_truth().object_poses["CAN"]
        decisions = []
        stop_reason = "decision budget"
        best_approach_gap = None
        attachment_offset = None
        for index in range(max_decisions):
            truth = backend.get_ground_truth()
            base = obs.robot.base_pose_world
            if base is None:
                raise ValueError("missing oracle base transform")
            oracle_can = compose(_inverse_pose(base), truth.object_poses["CAN"])
            can = oracle_can
            if localize is not None:
                can = Pose3D(Vec3(*map(float, localize(obs.rgb))), (0.0, 0.0, 0.0, 1.0))
            state = WorldState(
                SceneState([ObjectState("CAN", "can", can, Vec3(0, 0, 0), 1.0)]), obs.robot
            )
            executor = SkillExecutor(entry.seed, obs)
            measured_rise = truth.object_poses["CAN"].position.z - initial.position.z
            candidates = generate_candidates(
                executor,
                backend,
                obs,
                state,
                can_rise_m=measured_rise,
                measured_can_base=oracle_can,
                attachment_offset=attachment_offset,
            )
            forecasts = (
                [forecast_candidate(state, candidate).trajectory for candidate in candidates]
                if context != "smoke"
                else []
            )
            futures = [item.states[-1] for item in forecasts]
            if context == "shuffled" and len(futures) > 1:
                shift = int(np.random.default_rng(entry.seed + index).integers(1, len(futures)))
                futures = futures[shift:] + futures[:shift]
            rows, assessments = [], []
            for candidate_index, candidate in enumerate(candidates):
                checks = []
                rehearsal = _PreviewBackend(obs, truth)
                preview = SkillExecutor(entry.seed, obs)
                predicted = preview.run_skill(
                    rehearsal,
                    obs,
                    candidate,
                    audit=command_proxy(preview.monitor, candidate.skill, checks.append),
                )
                safe = predicted.settled and bool(checks) and not predicted.vetoes
                forecast = forecasts[candidate_index] if forecasts else None
                if forecast is not None:
                    safe = (
                        safe
                        and len(forecast.states) > 1
                        and set(forecast.hard_reasons) <= {"unverified-arm-sweep"}
                    )
                future = None
                if uses_context:
                    future = state if context == "controlled" else futures[candidate_index]
                raw_score = model.score(state, candidate, future)
                score = approach_progress_score(
                    raw_score, candidate, can.position, best_approach_gap
                )
                score = release_progress_score(score, candidate, truth.object_poses["CAN"].position)
                score = close_progress_score(
                    score,
                    candidate,
                    obs.robot.gripper_opening,
                    any(
                        option.skill in (Skill.LIFT, Skill.TRANSFER, Skill.LOWER, Skill.RELEASE)
                        for option in candidates
                    ),
                )
                assessments.append(
                    CandidateAssessment((("proxy", safe),), (score, None, None, None))
                )
                rows.append(
                    {
                        "action": asdict(candidate),
                        "score": score,
                        "raw_model_score": raw_score,
                        "strict_forecast_reasons": None
                        if forecast is None
                        else forecast.hard_reasons,
                        "analytic_forecast_tool_base": None
                        if forecast is None
                        else asdict(forecast.states[-1].robot.ee_pose.position),
                        "utility_context_tool_base": None
                        if future is None
                        else asdict(future.robot.ee_pose.position),
                        "proxy_pass": safe,
                        "predicted_tool_base": asdict(predicted.observation.robot.ee_pose.position),
                        "predicted_gripper_opening_m": predicted.observation.robot.gripper_opening,
                        "planned_commands": [asdict(command) for command in predicted.dispatched],
                        "failure": predicted.failure_reason,
                        "audits": [asdict(check) for check in checks],
                    }
                )
            assessments.append(CandidateAssessment((("proxy", True),), (0.0, None, None, None)))
            rows.append({"action": "terminate", "score": 0.0, "proxy_pass": True})
            decision = gate.select(assessments)
            record = {
                "index": index,
                "causal_best_approach_gap_m": best_approach_gap,
                "observed_attachment_tool_m": (
                    None if attachment_offset is None else attachment_offset.tolist()
                ),
                "can_estimate_base_m": asdict(can.position),
                "localization_error_m": None
                if localize is None
                else float(
                    np.linalg.norm(can.position.as_array() - oracle_can.position.as_array())
                ),
                "carrying_candidates": [
                    c.skill.value
                    for c in candidates
                    if c.skill in (Skill.LIFT, Skill.TRANSFER, Skill.LOWER, Skill.RELEASE)
                ],
                "measured_gripper_opening_m": obs.robot.gripper_opening,
                "measured_holding": obs.robot.gripper_holding,
                "measured_can_rise_m": measured_rise,
                "candidates": rows,
                "decision": asdict(decision),
                "feasible_skill_count": len(
                    {
                        candidate.skill
                        for candidate, row in zip(candidates, rows[:-1], strict=True)
                        if row["proxy_pass"]
                    }
                ),
            }
            decisions.append(record)
            write_json(out / f"decision-{index:02d}.json", record)
            if decision.selected_index is None or decision.selected_index == len(candidates):
                stop_reason = "termination selected"
                break
            candidate = candidates[decision.selected_index]
            live_audits = []
            actual = executor.run_skill(
                backend,
                obs,
                candidate,
                audit=command_proxy(executor.monitor, candidate.skill, live_audits.append),
            )
            obs_before = obs
            tool_before = obs.robot.ee_pose.position
            obs = actual.observation
            pose_after = backend.get_ground_truth().object_poses["CAN"]
            if actual.settled:
                new_attachment = measured_attachment(
                    obs_before, obs, truth.object_poses["CAN"].position, pose_after.position
                )
                if new_attachment is not None:
                    attachment_offset = new_attachment
                if obs.robot.gripper_opening >= GRIPPER_OPEN - 0.002:
                    attachment_offset = None
            if candidate.skill is Skill.APPROACH and obs.robot.base_pose_world is not None:
                can_after_base = compose(_inverse_pose(obs.robot.base_pose_world), pose_after)
                gap = float(
                    np.linalg.norm(
                        (
                            obs.robot.ee_pose.position.as_array()
                            - can_after_base.position.as_array()
                        )[:2]
                    )
                )
                can_shift = np.linalg.norm(
                    pose_after.position.as_array() - truth.object_poses["CAN"].position.as_array()
                )
                if best_approach_gap is None or can_shift >= CLOSE_ALIGN_TOL:
                    best_approach_gap = gap
                else:
                    best_approach_gap = min(gap, best_approach_gap)
            else:
                best_approach_gap = None
            x, y, _, _ = pose_after.orientation
            tilt_deg = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
            record["actual"] = {
                "settled": actual.settled,
                "can_rise_m": actual.can_rise_m,
                "can_tilt_deg": tilt_deg,
                "tool_before_base_m": asdict(tool_before),
                "tool_after_base_m": asdict(obs.robot.ee_pose.position),
                "can_before_world_m": asdict(truth.object_poses["CAN"].position),
                "can_after_world_m": asdict(pose_after.position),
                "gripper_opening_m": obs.robot.gripper_opening,
                "commands": [asdict(command) for command in actual.dispatched],
                "failure": actual.failure_reason,
                "vetoes": actual.vetoes,
                "audits": [asdict(audit) for audit in live_audits],
            }
            write_json(out / f"decision-{index:02d}.json", record)
            if not actual.settled or tilt_deg >= SETTINGS.can_tilt_limit_deg:
                stop_reason = "skill failed" if not actual.settled else "CAN exceeded tilt limit"
                break
        final = backend.get_ground_truth().object_poses["CAN"]
        result = {
            "split": "train",
            "scene": entry.scene.name,
            "seed": entry.seed,
            "mode": "oracle-assisted" if localize is None else "visual-can-oracle-safety",
            "utility_context": context,
            "stop_reason": stop_reason,
            "decision_budget": max_decisions,
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "profile": PROFILE_ID,
            "privileged_inputs": [
                "CAN pose for safety and verification",
                "HAM/APP poses for proxy",
                "world base transform",
                *([] if localize is not None else ["CAN pose for skill generation and utility"]),
            ],
            "instruction": "Pick up the can and place it at the target location.",
            "language_dependence": "unmeasured; fixed configured destination",
            "decisions": len(decisions),
            "verification": final_can_outcome(
                initial,
                final,
                place_world_m=SETTINGS.place_world,
                distance_tolerance_m=SETTINGS.place_tolerance_m,
                height_tolerance_m=SETTINGS.can_place_z_tolerance_m,
                tilt_limit_deg=SETTINGS.can_tilt_limit_deg,
                home_reached=all(abs(obs.robot.joint_pos[k] - v) <= 0.01 for k, v in home.items()),
                failure_state=None,
                failure_reason=None,
            ),
        }
        write_json(out / "learned-summary.json", result)
    return result


def run_matched_clear(out: Path, model: Path, webots: str | None) -> dict:
    """Freeze model bytes once and run all four matched clear TRAIN conditions."""
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "frozen-model.json"
    frozen.write_bytes(model.read_bytes())
    trials = {}
    for context in ("matched-none", "predicted", "shuffled", "controlled"):
        try:
            trials[context] = run_learned_smoke(
                out / context, frozen, webots, context=context, max_decisions=10
            )
        except Exception as error:
            trials[context] = {"error": str(error)}
        write_json(out / "matched-progress.json", trials)
    return trials
