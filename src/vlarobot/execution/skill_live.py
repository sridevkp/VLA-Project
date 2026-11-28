"""Bounded oracle-assisted learned clear-scene TRAIN diagnostic."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from sys import executable

import numpy as np

from vlarobot.backends.webots.config import CAN_RADIUS_M, GRIPPER_OPEN, SETTINGS, TIME_STEP
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
from vlarobot.execution.skill_empirical import forecast_candidate
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


def generate_candidates(
    executor: SkillExecutor, backend, observation: Observation, state: WorldState
) -> tuple[SkillAction, ...]:
    """Use current geometry and aperture, independent of the previous task phase."""
    base = observation.robot.base_pose_world
    if base is None:
        raise ValueError("missing base transform")
    can_world = compose(base, state.scene.objects[0].pose).position.as_array()
    anchors = executor._base_frame_targets(backend, observation, can_world=can_world)
    targets = {skill: Vec3(*map(float, anchors[key])) for skill, key in _TARGETS.items()}
    candidates = enumerate_skills(
        state,
        targets,
        position_tolerance_m=0.02,
        open_tolerance_m=0.002,
        open_position_m=GRIPPER_OPEN,
    )
    approach = targets[Skill.APPROACH]
    return (
        *candidates,
        SkillAction(Skill.APPROACH, replace(approach, y=approach.y + CAN_RADIUS_M)),
    )


def run_learned_smoke(
    out: Path,
    model_path: Path,
    webots: str | None,
    *,
    localizer_path: Path | None = None,
    context: str = "smoke",
) -> dict:
    """Run six receding-horizon decisions with a learned utility and scoped predictive vetoes."""
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
    entry = build_episode_plan("train", 0, n=1)[0]
    out.mkdir(parents=True, exist_ok=False)
    with launch_webots_session(
        webots=webots, trace_dir=out, headless=True, extra_env={"WEBOTS_PYTHON": executable}
    ) as session:
        backend = session.client
        obs = backend.reset(entry.scene, entry.seed)
        home = obs.robot.joint_pos.copy()
        initial = backend.get_ground_truth().object_poses["CAN"]
        decisions = []
        for index in range(6):
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
            candidates = generate_candidates(executor, backend, obs, state)
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
                score = model.score(state, candidate, future)
                assessments.append(
                    CandidateAssessment((("proxy", safe),), (score, None, None, None))
                )
                rows.append(
                    {
                        "action": asdict(candidate),
                        "score": score,
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
                "can_estimate_base_m": asdict(can.position),
                "localization_error_m": None
                if localize is None
                else float(
                    np.linalg.norm(can.position.as_array() - oracle_can.position.as_array())
                ),
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
                break
            candidate = candidates[decision.selected_index]
            live_audits = []
            actual = executor.run_skill(
                backend,
                obs,
                candidate,
                audit=command_proxy(executor.monitor, candidate.skill, live_audits.append),
            )
            obs = actual.observation
            record["actual"] = {
                "settled": actual.settled,
                "can_rise_m": actual.can_rise_m,
                "gripper_opening_m": obs.robot.gripper_opening,
                "commands": [asdict(command) for command in actual.dispatched],
                "failure": actual.failure_reason,
                "vetoes": actual.vetoes,
                "audits": [asdict(audit) for audit in live_audits],
            }
            write_json(out / f"decision-{index:02d}.json", record)
            if not actual.settled:
                break
        final = backend.get_ground_truth().object_poses["CAN"]
        result = {
            "split": "train",
            "scene": entry.scene.name,
            "seed": entry.seed,
            "mode": "oracle-assisted-no-context"
            if localize is None
            else "visual-can-oracle-safety-no-context",
            "utility_context": context,
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
