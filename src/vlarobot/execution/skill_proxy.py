"""Owner-scoped V4 tool-envelope audit over a lockstep nominal skill trajectory."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vlarobot.backends.webots.config import PLACE_TOLERANCE_M, PLACE_WORLD, TIME_STEP
from vlarobot.core.types import ActionChunk, EEPoseTarget, GripperCmd, GroundTruth, WorldState
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_can import _ik_tolerance_for_state
from vlarobot.execution.skills import Skill
from vlarobot.execution.worldmind_proxy_binding import (
    _MODEL_PINS,
    _matched_proxy_evidence,
    proxy_scene,
)
from vlarobot.world_model.rollout import AnalyticSurrogate, Attachment

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


@dataclass(frozen=True)
class SkillProxyAssessment:
    profile: str
    profile_sha256: str
    clearance_sha256: str
    full_arm_sweep_verified: bool
    strict_reasons: tuple[str, ...]
    checks: tuple[tuple[str, bool], ...]


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
    settings, profile_sha256 = _clearance_configuration()
    trajectory = surrogate.predict_trajectory(state, chunk, attachment=attachment)
    command_type = GripperCmd if skill in (Skill.OPEN, Skill.CLOSE, Skill.RELEASE) else EEPoseTarget
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
        evidence = _matched_proxy_evidence(
            scene,
            chunk,
            trajectory,
            monitor,
            settings,
            active_phase=skill.value,
            phase_terminal=True,
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
                        command_type is GripperCmd
                        or np.linalg.norm(
                            trajectory.states[-1].robot.ee_pose.position.as_array()
                            - chunk.actions[-1].pose.position.as_array()
                        )
                        <= _ik_tolerance_for_state(skill.value)
                    ),
                ),
                ("table", evidence.geometric_hard_pass[0] is True),
                ("modeled_obstacles", evidence.geometric_hard_pass[1] is True),
            )
        )
        if skill is Skill.LOWER:
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
