"""Bounded oracle-assisted learned clear-scene TRAIN diagnostic."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from sys import executable

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
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
from vlarobot.eval.metrics import wilson_interval
from vlarobot.eval.runner import EvalEpisode, build_episode_plan
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import launch_webots_session
from vlarobot.execution.skill_empirical import FORECAST_PROFILE, forecast_candidate
from vlarobot.execution.skill_proxy import PROFILE_ID, command_proxy, placement_clearance_offset_m
from vlarobot.execution.skill_selection import UtilityModel, enumerate_skills
from vlarobot.execution.skills import (
    HOME_VERIFICATION_TOLERANCE_RAD,
    Skill,
    SkillAction,
    SkillExecutor,
)
from vlarobot.execution.worldmind_pipeline import _inverse_pose
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.rollout import compose, rotate

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
    *,
    previous_offset: np.ndarray | None = None,
) -> np.ndarray | None:
    """Establish attachment by lifting, then refresh it on measured carrying motion."""
    old_base, new_base = before.robot.base_pose_world, after.robot.base_pose_world
    if old_base is None or new_base is None:
        return None
    if not 0.002 < after.robot.gripper_opening < GRIPPER_OPEN - 0.002:
        return None
    old_tool = compose(old_base, before.robot.ee_pose)
    new_tool = compose(new_base, after.robot.ee_pose)
    tool_delta = new_tool.position.as_array() - old_tool.position.as_array()
    can_delta = can_after_world.as_array() - can_before_world.as_array()
    can_in_tool = compose(_inverse_pose(new_tool), Pose3D(can_after_world, (0.0, 0.0, 0.0, 1.0)))
    offset = can_in_tool.position.as_array()
    separation_error = np.linalg.norm(can_delta - tool_delta)
    if previous_offset is None:
        valid = can_delta[2] >= LIFT_MIN and separation_error <= CAN_RADIUS_M
    else:
        diameter = 2 * CAN_RADIUS_M
        valid = (
            min(np.linalg.norm(can_delta), np.linalg.norm(tool_delta)) >= LIFT_MIN
            and separation_error <= diameter
            and np.linalg.norm(offset) <= diameter
        )
    return offset if valid else None


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
    lower_alternative = ()
    if Skill.LOWER in targets:
        lifted_target = targets[Skill.LOWER].as_array() + rotate(
            np.asarray(_inverse_pose(base).orientation),
            np.array((0.0, 0.0, placement_clearance_offset_m())),
        )
        lower_alternative = (SkillAction(Skill.LOWER, Vec3(*map(float, lifted_target))),)
    return (
        *(
            candidate
            for candidate in candidates
            if candidate.skill not in (Skill.OPEN, Skill.RELEASE) or can_rise_m < LIFT_MIN
        ),
        *alternate,
        *lower_alternative,
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


def _verify_pose(initial: Pose3D, current: Pose3D, home_reached: bool) -> dict:
    return final_can_outcome(
        initial,
        current,
        place_world_m=SETTINGS.place_world,
        distance_tolerance_m=SETTINGS.place_tolerance_m,
        height_tolerance_m=SETTINGS.can_place_z_tolerance_m,
        tilt_limit_deg=SETTINGS.can_tilt_limit_deg,
        home_reached=home_reached,
        failure_state=None,
        failure_reason=None,
    )


def _at_home(observation: Observation, home: dict[str, float]) -> bool:
    return all(
        abs(observation.robot.joint_pos[name] - target) <= HOME_VERIFICATION_TOLERANCE_RAD
        for name, target in home.items()
    )


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
    if scene_index not in (0, 1, 2):
        raise ValueError("Choose one of the three approved TRAIN diagnostic scenes")
    return _run_train_episode(
        out,
        model_path,
        webots,
        build_episode_plan("train", 0, n=3)[scene_index],
        localizer_path=localizer_path,
        context=context,
        max_decisions=max_decisions,
    )


def _run_train_episode(
    out: Path,
    model_path: Path,
    webots: str | None,
    entry: EvalEpisode,
    *,
    localizer_path: Path | None = None,
    context: str,
    max_decisions: int,
) -> dict:
    if not 1 <= max_decisions <= 16:
        raise ValueError("Bounded diagnostic requires 1–16 decisions")
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
        observed_release = False
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
            home_reached = _at_home(obs, home)
            placement = _verify_pose(initial, truth.object_poses["CAN"], home_reached)
            placed_and_released = (
                observed_release
                and obs.robot.gripper_opening >= GRIPPER_OPEN - 0.002
                and all(
                    placement[key] for key in ("place_distance_ok", "can_height_ok", "can_tilt_ok")
                )
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
            # HOME has exact joint forecasts and no empirical labels in this fit.
            if placed_and_released and not home_reached:
                candidates += (
                    SkillAction(
                        Skill.HOME, home_joints=tuple((name, home[name]) for name in ARM_JOINTS)
                    ),
                )
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
                forecast = forecasts[candidate_index] if candidate_index < len(forecasts) else None
                if forecast is not None:
                    safe = (
                        safe
                        and len(forecast.states) > 1
                        and set(forecast.hard_reasons) <= {"unverified-arm-sweep"}
                    )
                future = None
                if uses_context and forecast is not None:
                    future = state if context == "controlled" else futures[candidate_index]
                raw_score = (
                    None if candidate.skill is Skill.HOME else model.score(state, candidate, future)
                )
                score = (
                    0.0
                    if raw_score is None
                    else approach_progress_score(
                        raw_score, candidate, can.position, best_approach_gap
                    )
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
                score_source = "empirical-utility"
                if placed_and_released:
                    score_source = "measured-completion-objective"
                    if candidate.skill is Skill.HOME:
                        score = float(_at_home(predicted.observation, home))
                    elif candidate.skill is Skill.RETREAT and not home_reached:
                        score = min(0.5, max(0.0, score))
                    else:
                        score = min(-1.0, score)
                assessments.append(
                    CandidateAssessment((("proxy", safe),), (score, None, None, None))
                )
                rows.append(
                    {
                        "action": asdict(candidate),
                        "score": score,
                        "raw_model_score": raw_score,
                        "score_source": score_source,
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
            termination_score = 1.0 if placed_and_released and home_reached else 0.0
            assessments.append(
                CandidateAssessment((("proxy", True),), (termination_score, None, None, None))
            )
            rows.append({"action": "terminate", "score": termination_score, "proxy_pass": True})
            decision = gate.select(assessments)
            record = {
                "index": index,
                "placed_and_released": placed_and_released,
                "home_reached": home_reached,
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
            if actual.settled and candidate.skill in (Skill.OPEN, Skill.RELEASE):
                observed_release = True
            elif candidate.skill is Skill.CLOSE:
                observed_release = False
            if actual.settled:
                new_attachment = measured_attachment(
                    obs_before,
                    obs,
                    truth.object_poses["CAN"].position,
                    pose_after.position,
                    previous_offset=attachment_offset,
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
            if candidate.skill is Skill.HOME:
                record["actual"]["home_reached"] = _at_home(obs, home)
                record["actual"]["joint_after_rad"] = obs.robot.joint_pos.copy()
            write_json(out / f"decision-{index:02d}.json", record)
            if not actual.settled or tilt_deg >= SETTINGS.can_tilt_limit_deg:
                stop_reason = "skill failed" if not actual.settled else "CAN exceeded tilt limit"
                break
        final = backend.get_ground_truth().object_poses["CAN"]
        verification = _verify_pose(initial, final, _at_home(obs, home))
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
                "CAN pose for grasp and release safety, completion scoring and verification",
                "HAM/APP poses for proxy",
                "world base transform",
                *([] if localize is not None else ["CAN pose for skill generation and utility"]),
            ],
            "instruction": "Pick up the can and place it at the target location.",
            "language_dependence": "unmeasured; fixed configured destination",
            "decisions": len(decisions),
            "home_reference_rad": home,
            "home_tolerance_rad": HOME_VERIFICATION_TOLERANCE_RAD,
            "object_placement_passed": all(
                verification[key] for key in ("place_distance_ok", "can_height_ok", "can_tilt_ok")
            ),
            "verified_termination": stop_reason == "termination selected"
            and verification["success"],
            "verification": verification,
        }
        write_json(out / "learned-summary.json", result)
    return result


def run_matched_clear(
    out: Path, model: Path, webots: str | None, *, max_decisions: int = 16
) -> dict:
    """Freeze model bytes once and run all four matched clear TRAIN conditions."""
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "frozen-model.json"
    frozen.write_bytes(model.read_bytes())
    trials = {}
    for context in ("matched-none", "predicted", "shuffled", "controlled"):
        try:
            trials[context] = run_learned_smoke(
                out / context, frozen, webots, context=context, max_decisions=max_decisions
            )
        except Exception as error:
            trials[context] = {"error": str(error)}
        write_json(out / "matched-progress.json", trials)
    return trials


def run_train_admission(
    out: Path, model: Path, webots: str | None, *, verified_clear: Path
) -> dict:
    """Run the frozen 24 TRAIN trials only after verified predicted-context termination."""
    prerequisite = json.loads(verified_clear.read_bytes())
    model_bytes = model.read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    plan = build_episode_plan("train", 0)
    expected = {
        "split": "train",
        "scene": plan[0].scene.name,
        "seed": plan[0].seed,
        "mode": "oracle-assisted",
        "utility_context": "predicted",
        "profile": PROFILE_ID,
        "model_sha256": model_sha,
        "home_tolerance_rad": HOME_VERIFICATION_TOLERANCE_RAD,
        "stop_reason": "termination selected",
    }
    if any(prerequisite.get(key) != value for key, value in expected.items()) or not all(
        prerequisite.get("verification", {}).get(key) is True
        for key in ("success", "home_reached", "place_distance_ok", "can_height_ok", "can_tilt_ok")
    ):
        raise ValueError(
            "Admission requires independently verified completion by the frozen policy."
        )
    budget = prerequisite["decision_budget"]
    if not 1 <= budget <= 16 or len(plan) != 24:
        raise ValueError("Admission requires the unchanged diagnostic budget and 24 TRAIN trials.")

    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "frozen-model.json"
    frozen.write_bytes(model_bytes)
    write_json(out / "verified-clear.json", prerequisite)
    source = Path(__file__).resolve().parent
    runtime_modules = ("skill_live.py", "skill_proxy.py", "skill_selection.py", "skills.py")
    runtime_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in runtime_modules
    }
    write_json(
        out / "admission-plan.json",
        {
            **expected,
            "decision_budget": budget,
            "minimum_successes": 10,
            "runtime_module_sha256": runtime_hashes,
            "plan": [
                {"index": item.index, "scene": item.scene.name, "seed": item.seed} for item in plan
            ],
        },
    )

    def run(entry: EvalEpisode) -> dict:
        folder = out / f"episode-{entry.index:02d}"
        row = {"index": entry.index, "scene": entry.scene.name, "seed": entry.seed}
        try:
            return row | _run_train_episode(
                folder, frozen, webots, entry, context="predicted", max_decisions=budget
            )
        except Exception as error:
            return row | {"execution_error": str(error)}

    report = {"split": "train", "threshold": 10, "total": 24, "episodes": []}
    with ThreadPoolExecutor(max_workers=3) as workers:
        for row in workers.map(run, plan):
            report["episodes"].append(row)
            write_json(out / "admission-progress.json", report)
    measured = [row for row in report["episodes"] if "verification" in row]
    successes = sum(row["verification"]["success"] for row in measured)
    complete = len(measured) == 24
    report.update(
        measured=len(measured),
        successes=successes,
        wilson_95=wilson_interval(successes, 24) if complete else None,
        passed=successes >= 10 if complete else None,
    )
    write_json(out / "admission-summary.json", report)
    return report
