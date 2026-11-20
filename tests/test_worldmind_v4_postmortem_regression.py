"""Offline regressions for the Stage 1 V4 postmortem (non-scored diagnostic evidence).

V1-V4 are read-only evidence. Nothing here starts Webots, resumes a campaign, writes into a
preserved run directory, scores a pipeline, selects a calibration, or reads TRAIN/TEST. Values
labelled "runtime" are taken from preserved sidecars/NPZ files; waypoint values labelled
"reconstructed" are derived from the pinned source and the recorded initial oracle state.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_worldmind_production import SpyBackend  # noqa: PLC2701 - existing shared spy
from test_worldmind_stage1_runner import _one_episode_plan, _setup  # noqa: PLC2701
from test_worldmind_v2_postmortem_regression import (  # noqa: PLC2701 - existing shared helpers
    _IntermediateVetoMonitor,
    _NoVerification,
    _production_waypoints,
    _tree_hash,
)

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import MAX_JOINT_STEP_RAD
from vlarobot.backends.worldmind_bindings import runtime_safety_limits
from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState, Vec3
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.worldmind_pipeline import _home_reached
from vlarobot.execution.worldmind_stage1_pipeline import run_stage1_evidence

REPO_ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = REPO_ROOT / "runs/worldmind/stage1-validation-v4-2026-11-20"
APPROVED_CONFIG = REPO_ROOT / "configs/worldmind_diagnostic.approved.yaml"
AGGREGATES = {
    "v1": "08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712",
    "v2": "b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb",
    "v3": "d4204a6fae79c16988e68f0b7cb61b27c90ba614f55fbfa507d1f666cf3d4d29",
    "v4": "48ce9f020802dac08c3fa065f4b53ef7d0f547160a891f488b081966f4eeba47",
}
VETO_CASES = {
    "validation-clear-i03-s0": {
        "observed": 0.44138256595535447,
        "previous_command": -1.2686945979476707,
        "rejected_target": -1.4186945979476706,
        "observed_implied_velocity": 1.8600771639030251,
    },
    "validation-reach-limited-i05-s1": {
        "observed": -0.048049874615676044,
        "previous_command": 1.6620230047232916,
        "rejected_target": 1.8120230047232915,
        "observed_implied_velocity": 1.8600728793389676,
    },
}
ARM7 = "arm_7_joint"
VETO_REASON = "joint target for arm_7_joint violates velocity limits"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _meta(episode_id: str) -> dict[str, Any]:
    for path in sorted(V4_ROOT.glob("episode-*.meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["scenario"]["name"] == episode_id:
            return meta
    raise AssertionError(f"episode metadata not found: {episode_id}")


def _arrays(episode_id: str) -> dict[str, np.ndarray]:
    index = int(_meta(episode_id)["index"])
    with np.load(V4_ROOT / f"episode-{index:02d}.steps.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _observed_at(times: np.ndarray, values: np.ndarray, timestamp: float) -> float:
    return float(values[int(np.argmin(np.abs(times - timestamp)))])


def _state(joint_pos: dict[str, float]) -> RobotState:
    return RobotState(
        joint_pos=dict(joint_pos),
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.0,
    )


def _veto_trace(episode_id: str) -> tuple[dict[str, Any], dict[str, float], float]:
    arrays = _arrays(episode_id)
    decisions = _jsonl(V4_ROOT / "worldmind-sidecars/decisions.jsonl")
    record = next(item for item in decisions if item["decision_id"] == f"{episode_id}-d01")
    actions = record["execution"]["chunk"]["actions"]
    action = actions[15]
    names = [str(name) for name in arrays["joint_names"]]
    observed = _observed_at(
        arrays["t"],
        arrays["joint_pos"][:, names.index(ARM7)],
        float(action["observation_t_s"]),
    )
    target = float(action["action"]["q"][ARM7])
    previous = float(actions[14]["action"]["q"][ARM7])
    return action, {"observed": observed, "previous": previous, "target": target}, target - previous


def test_v1_v2_v3_v4_campaign_aggregates_remain_unchanged() -> None:
    for tag, expected in AGGREGATES.items():
        root = REPO_ROOT / f"runs/worldmind/stage1-validation-{tag}-2026-11-20"
        assert _tree_hash(root, REPO_ROOT) == expected


def test_v4_arm7_veto_is_reproduced_from_preserved_trace() -> None:
    limits = runtime_safety_limits()
    monitor = SafetyMonitor(limits)
    assert limits.joint_velocity[ARM7] == 1.76
    assert limits.command_horizon == 1.0
    decisions = _jsonl(V4_ROOT / "worldmind-sidecars/decisions.jsonl")

    for episode_id, expected in VETO_CASES.items():
        action, trace, delta = _veto_trace(episode_id)
        observed = trace["observed"]
        previous = trace["previous"]
        target = trace["target"]
        assert action["action_index"] == 15
        assert action["pre_apply_accepted"] is False
        assert action["pre_apply_reason"] == VETO_REASON

        assert observed == pytest.approx(expected["observed"], abs=1e-12)
        assert previous == pytest.approx(expected["previous_command"], abs=1e-12)
        assert target == pytest.approx(expected["rejected_target"], abs=1e-12)
        assert delta == pytest.approx(target - previous, abs=1e-12)
        assert abs(delta) <= MAX_JOINT_STEP_RAD + 1e-12

        planned_velocity = abs(delta) / limits.command_horizon
        observed_velocity = abs(target - observed) / limits.command_horizon
        assert planned_velocity == pytest.approx(0.15, abs=1e-12)
        assert observed_velocity == pytest.approx(expected["observed_implied_velocity"], abs=1e-12)
        assert monitor.validate(JointTarget({ARM7: target}), _state({ARM7: previous})).accepted
        observed_verdict = monitor.validate(JointTarget({ARM7: target}), _state({ARM7: observed}))
        assert observed_verdict.accepted is False
        assert observed_verdict.reason == VETO_REASON

        record = next(item for item in decisions if item["decision_id"] == f"{episode_id}-d01")
        chunk = record["execution"]["chunk"]
        assert chunk["termination_reason"] == "safety_rejected"
        assert chunk["attempted_action_count"] == 16
        assert chunk["applied_action_count"] == 15
        assert chunk["backend_step_count"] == 15
        assert _meta(episode_id)["outcome"]["worldmind_control_steps"] == 50


def test_v4_veto_correction_keeps_reviewed_velocity_limit() -> None:
    limits = runtime_safety_limits()
    monitor = SafetyMonitor(limits)
    _action, trace, _delta = _veto_trace("validation-clear-i03-s0")
    observed = trace["observed"]
    lower, upper = limits.joint_position[ARM7]
    max_delta = limits.joint_velocity[ARM7] * limits.command_horizon

    at_limit = observed - (max_delta - 1e-9)
    beyond_limit = observed - (max_delta + 1e-9)
    assert lower <= beyond_limit < at_limit <= upper
    assert monitor.validate(JointTarget({ARM7: at_limit}), _state({ARM7: observed})).accepted
    beyond = monitor.validate(JointTarget({ARM7: beyond_limit}), _state({ARM7: observed}))
    assert beyond.accepted is False
    assert beyond.reason == VETO_REASON


def test_v4_home_tracking_and_terminal_verification_use_observed_state() -> None:
    config = load_worldmind_diagnostic_config(APPROVED_CONFIG)
    episode_id = "validation-clear-i00-s0"
    arrays = _arrays(episode_id)
    names = [str(name) for name in arrays["joint_names"]]
    observed_joints = {
        name: float(arrays["joint_pos"][-1, index]) for index, name in enumerate(names)
    }
    observed_state = _state(observed_joints)
    commanded_home = _state(dict(config.verification.home_joints))

    assert _home_reached(config, commanded_home) is True
    assert _home_reached(config, observed_state) is False
    max_home_error = max(
        abs(observed_joints[name] - target)
        for name, target in config.verification.home_joints.items()
    )
    assert max_home_error == pytest.approx(1.1536445247351717, abs=1e-12)
    assert max_home_error > config.verification.home_tolerance_rad

    gt_names = [str(name) for name in arrays["gt_names"]]
    can_index = gt_names.index("CAN")
    initial_pose = Pose3D(
        Vec3(*map(float, arrays["gt_xyz"][0, can_index])),
        tuple(map(float, arrays["gt_quat"][0, can_index])),
    )
    final_pose = Pose3D(
        Vec3(*map(float, arrays["gt_xyz"][-1, can_index])),
        tuple(map(float, arrays["gt_quat"][-1, can_index])),
    )
    outcome = final_can_outcome(
        initial_pose,
        final_pose,
        place_world_m=config.verification.destination_world,
        distance_tolerance_m=config.verification.distance_tolerance_m,
        height_tolerance_m=config.verification.height_tolerance_m,
        tilt_limit_deg=math.degrees(config.verification.tilt_limit_rad),
        home_reached=_home_reached(config, observed_state),
        failure_state=None,
        failure_reason=None,
    )
    assert outcome["final_target_distance"] == pytest.approx(0.2531504291422367, abs=1e-12)
    assert outcome["place_distance_ok"] is False
    assert outcome["home_reached"] is False
    assert outcome["success"] is False


def test_v4_unachievable_trajectory_fails_closed() -> None:
    config = load_worldmind_diagnostic_config(APPROVED_CONFIG)
    episode_id = "validation-clear-i00-s0"
    meta = _meta(episode_id)
    arrays = _arrays(episode_id)
    names = [str(name) for name in arrays["joint_names"]]
    observed_joints = {
        name: float(arrays["joint_pos"][-1, index]) for index, name in enumerate(names)
    }
    home_reached = _home_reached(config, _state(observed_joints))
    gt_names = [str(name) for name in arrays["gt_names"]]
    can_index = gt_names.index("CAN")
    initial_pose = Pose3D(
        Vec3(*map(float, arrays["gt_xyz"][0, can_index])),
        tuple(map(float, arrays["gt_quat"][0, can_index])),
    )
    final_pose = Pose3D(
        Vec3(*map(float, arrays["gt_xyz"][-1, can_index])),
        tuple(map(float, arrays["gt_quat"][-1, can_index])),
    )
    outcome = final_can_outcome(
        initial_pose,
        final_pose,
        place_world_m=config.verification.destination_world,
        distance_tolerance_m=config.verification.distance_tolerance_m,
        height_tolerance_m=config.verification.height_tolerance_m,
        tilt_limit_deg=math.degrees(config.verification.tilt_limit_rad),
        home_reached=home_reached,
        failure_state=meta["outcome"]["failure_state"],
        failure_reason=meta["outcome"]["failure_reason"],
    )
    assert outcome["success"] is False
    assert outcome["failure_state"] == "VERIFICATION_ABORT"
    assert meta["outcome"]["success"] is False

    gripper = arrays["gripper"]
    assert float(gripper.min()) < config.verification.home_tolerance_rad
    assert float(gripper.max()) < 0.045
    can_trace = arrays["gt_xyz"][:, can_index]
    assert float(np.linalg.norm(np.diff(can_trace, axis=0), axis=1).sum()) < 1e-4


def test_v4_preserved_bookkeeping_gap_is_reconstructed() -> None:
    recorded_backend_steps = 0
    recorded_control_steps = 0
    for path in sorted(V4_ROOT.glob("episode-*.meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        index = int(meta["index"])
        with np.load(V4_ROOT / f"episode-{index:02d}.steps.npz", allow_pickle=False) as archive:
            recorded_backend_steps += len(archive["t"]) - 1
        recorded_control_steps += int(meta["outcome"]["worldmind_control_steps"])
    decisions = _jsonl(V4_ROOT / "worldmind-sidecars/decisions.jsonl")
    aborted_prefixes = [
        item["execution"]["chunk"]
        for item in decisions
        if item["execution"]["chunk"]["termination_reason"] == "safety_rejected"
    ]
    assert recorded_backend_steps == 1730
    assert recorded_control_steps == 1700
    assert recorded_backend_steps - recorded_control_steps == 30
    assert len(aborted_prefixes) == 2
    assert sum(item["backend_step_count"] for item in aborted_prefixes) == 30


def test_complete_open_loop_chunk_counts_steps_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=50))
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = SpyBackend(FakeBackend(), fixture.shared)
    out = tmp_path / "v4-complete-steps"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    decision = _jsonl(sidecars["decisions"])[0]
    chunk = decision["execution"]["chunk"]
    meta = json.loads((out / "episode-00.meta.json").read_text(encoding="utf-8"))
    assert backend.events.count("step") == 50
    assert chunk["termination_reason"] == "chunk_complete"
    assert chunk["backend_step_count"] == 50
    assert meta["outcome"]["worldmind_control_steps"] == 50
    assert meta["step_count"] == 51
    pairing = _jsonl(sidecars["prediction_pairing"])[0]
    assert pairing["status"] == "paired"
    assert pairing["valid_paired_steps"] == 50
    assert pairing["missing_observations"] == []


def test_safety_aborted_chunk_counts_accepted_steps_and_keeps_pairing_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, config, runtime = _setup(tmp_path, monkeypatch)
    config = replace(config, bounds=replace(config.bounds, max_decisions=1, max_steps=50))
    plan = _one_episode_plan()
    _production_waypoints(fixture, config, plan)
    backend = SpyBackend(FakeBackend(), fixture.shared)
    base_safety = fixture.bindings.safety["safety_factory"]

    def safety_factory(loaded: Any, inputs: Any) -> Any:
        return _IntermediateVetoMonitor(base_safety(loaded, inputs), backend, 7)

    fixture.bindings.safety["safety_factory"] = safety_factory
    fixture.bindings.verification["placement_verifier"] = lambda *args: _NoVerification()
    out = tmp_path / "v4-safety-abort-steps"
    _, sidecars = run_stage1_evidence(
        plan,
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode="diagnostic_open_loop_chunk",
        runtime_values=runtime,
    )

    decision = _jsonl(sidecars["decisions"])[0]
    chunk = decision["execution"]["chunk"]
    meta = json.loads((out / "episode-00.meta.json").read_text(encoding="utf-8"))
    assert backend.events == ["reset"] + ["apply", "step"] * 7
    assert chunk["termination_reason"] == "safety_rejected"
    assert chunk["attempted_action_count"] == 8
    assert chunk["applied_action_count"] == 7
    assert chunk["backend_step_count"] == 7
    assert chunk["observation_count"] == 8
    assert meta["outcome"]["worldmind_control_steps"] == 7
    assert meta["step_count"] == 8
    with np.load(out / "episode-00.steps.npz", allow_pickle=False) as archive:
        assert len(archive["t"]) == 8

    pairing = _jsonl(sidecars["prediction_pairing"])[0]
    assert pairing["status"] == "abort_boundary"
    assert pairing["prediction_horizon_steps"] == 50
    assert pairing["valid_paired_steps"] == chunk["applied_action_count"]
    assert pairing["missing_observations"] == list(range(8, 51))
