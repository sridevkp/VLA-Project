"""Offline regressions for the Stage 1 V2 postmortem (non-scored diagnostic evidence).

The V2 campaign is read-only evidence: nothing here reads Webots, resumes a campaign, or writes
into either preserved run directory. The arm-joint case is explicitly reconstructed from the
preserved first observation plus the pinned corrected solver/limits; it is not presented as a
runtime-recorded target.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worldmind_stage1_runner import _setup

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import MAX_JOINT_STEP_RAD
from vlarobot.core.types import (
    ApplyResult,
    JointTarget,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.runner import CampaignFailStop
from vlarobot.eval.worldmind_config import plan_worldmind_diagnostic
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import (
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.execution.worldmind_stage1_pipeline import (
    Stage1WorldMindEpisodePolicy,
    run_stage1_evidence,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
JOINT = "arm_6_joint"
REVIEWED_LOWER = -1.39
REVIEWED_UPPER = 1.39
# Reconstructed from the preserved V2 episode-00 state and draw seed 8787345, not runtime-recorded.
RECONSTRUCTED_CURRENT = -1.2534098586703584
RECONSTRUCTED_RAW_ENDPOINT = -1.3899999999987871
RECONSTRUCTED_JITTERED_ENDPOINT = -1.3912135100874889
RECONSTRUCTED_DELTA = -0.13780365141713058
ORIGINAL_EXCEPTION = (
    "Safety validation rejected action: joint target for arm_6_joint violates position limits"
)


def _limits() -> SafetyLimits:
    return SafetyLimits(
        joint_position={JOINT: (REVIEWED_LOWER, REVIEWED_UPPER)},
        joint_velocity={JOINT: 1.76},
        command_horizon=1.0,
    )


def _state(joint_value: float) -> WorldState:
    robot = RobotState(
        joint_pos={JOINT: joint_value},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
        gripper_opening=0.045,
        base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), IDENTITY),
    )
    return WorldState(SceneState([]), robot)


def _stub_solver(endpoint: dict[str, float], residual_m: float = 0.0) -> Any:
    return SimpleNamespace(
        solve=lambda _pose, _state, *, seed=0: SimpleNamespace(  # noqa: ARG005
            joints=dict(endpoint), position_error_m=residual_m, orientation_error_rad=0.0
        )
    )


class _SpyMonitor:
    def __init__(self, inner: SafetyMonitor) -> None:
        self.inner = inner
        self.limits = inner.limits
        self.calls: list[ApplyResult] = []

    def validate(self, action: Any, robot: Any) -> ApplyResult:
        verdict = self.inner.validate(action, robot)
        self.calls.append(verdict)
        return verdict


def _adapter(
    *,
    endpoint: dict[str, float],
    monitor: Any,
    hold_steps: int,
    residual_m: float = 0.0,
    jitter: JointJitter | None = None,
    position_tolerance_m: float = 0.005,
) -> ScriptedChunkAdapter:
    return ScriptedChunkAdapter(
        (ScriptedWaypoint("TRANSIT", hold_steps, pose=Pose3D(Vec3(0.0, 0.0, 0.8), IDENTITY)),),
        solver=_stub_solver(endpoint, residual_m),
        monitor=monitor,
        horizon_steps=hold_steps,
        control_dt_s=0.032,
        jitter=jitter or JointJitter(0.0, 0.0, 1),
        max_joint_step_rad=MAX_JOINT_STEP_RAD,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=0.01,
    )


def test_reconstructed_v2_rejection_is_produced_by_the_original_delta_only_rule() -> None:
    """Legacy formula, not endpoint IK, produced the rejected absolute target."""
    bounded_delta = max(
        -MAX_JOINT_STEP_RAD,
        min(MAX_JOINT_STEP_RAD, RECONSTRUCTED_JITTERED_ENDPOINT - RECONSTRUCTED_CURRENT),
    )
    legacy_target = RECONSTRUCTED_CURRENT + bounded_delta

    assert bounded_delta == pytest.approx(RECONSTRUCTED_DELTA, abs=1e-15)
    assert abs(bounded_delta) <= MAX_JOINT_STEP_RAD
    assert legacy_target == pytest.approx(RECONSTRUCTED_JITTERED_ENDPOINT, abs=1e-15)
    assert legacy_target < REVIEWED_LOWER

    verdict = SafetyMonitor(_limits()).validate(
        JointTarget({JOINT: legacy_target}), _state(RECONSTRUCTED_CURRENT).robot
    )
    assert not verdict.accepted
    assert verdict.reason == "joint target for arm_6_joint violates position limits"


def test_fixed_candidate_path_eliminates_the_reconstructed_v2_rejection() -> None:
    """The near-limit endpoint is redrawn by jitter; SafetyMonitor thresholds stay unchanged."""
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    chunk = _adapter(
        endpoint={JOINT: RECONSTRUCTED_RAW_ENDPOINT},
        monitor=monitor,
        hold_steps=1,
        jitter=JointJitter(0.001, 0.005, 10),
    ).draw()(_task(), _state(RECONSTRUCTED_CURRENT), None, 147)
    target = float(chunk.actions[0].q[JOINT])  # type: ignore[attr-defined]

    assert REVIEWED_LOWER <= target <= REVIEWED_UPPER
    assert abs(target - RECONSTRUCTED_CURRENT) <= MAX_JOINT_STEP_RAD
    assert target != pytest.approx(RECONSTRUCTED_JITTERED_ENDPOINT, abs=1e-12)
    assert len(monitor.calls) == len(chunk.actions)
    assert all(verdict.accepted for verdict in monitor.calls)


def test_generated_targets_respect_absolute_and_per_tick_limits_without_bypassing_safety() -> None:
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    chunk = _adapter(
        endpoint={JOINT: RECONSTRUCTED_RAW_ENDPOINT},
        monitor=monitor,
        hold_steps=16,
        jitter=JointJitter(0.001, 0.005, 10),
    ).draw()(_task(), _state(1.0), None, 147)

    current = 1.0
    for action in chunk.actions:
        target = float(action.q[JOINT])  # type: ignore[attr-defined]
        assert REVIEWED_LOWER <= target <= REVIEWED_UPPER
        assert abs(target - current) <= MAX_JOINT_STEP_RAD + 1e-12
        current = target
    assert len(monitor.calls) == len(chunk.actions)
    assert all(verdict.accepted for verdict in monitor.calls)


def test_endpoint_residual_outside_reviewed_tolerance_fails_closed() -> None:
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    with pytest.raises(ValueError, match="residual outside reviewed tolerance"):
        _adapter(
            endpoint={JOINT: 0.0},
            monitor=monitor,
            hold_steps=1,
            residual_m=0.01,
            position_tolerance_m=0.005,
        ).draw()(_task(), _state(0.0), None, 3)
    assert monitor.calls == []


def test_raw_endpoint_outside_reviewed_absolute_limits_fails_closed() -> None:
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    with pytest.raises(ValueError, match="violates reviewed absolute position limits"):
        _adapter(
            endpoint={JOINT: -1.5},
            monitor=monitor,
            hold_steps=1,
        ).draw()(_task(), _state(0.0), None, 3)
    assert monitor.calls == []


def test_stage1_partial_18_episode_fail_stop_finalizes_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, source, config, runtime = _setup(tmp_path, monkeypatch)
    _, plan = plan_worldmind_diagnostic(
        source,
        tmp_path / "frozen-plan",
        split="validation",
        backend="fake",
        state_source="oracle",
        bindings=fixture.bindings,
    )
    assert len(plan) == 18

    calls: list[int] = []

    def explode(_policy: Stage1WorldMindEpisodePolicy, _backend: Any, seed: int) -> Any:
        calls.append(seed)
        raise RuntimeError(ORIGINAL_EXCEPTION)

    monkeypatch.setattr(Stage1WorldMindEpisodePolicy, "execute", explode)
    out = tmp_path / "stage1-v2-reconstructed-partial"

    with pytest.raises(CampaignFailStop, match="arm_6_joint violates position limits"):
        run_stage1_evidence(
            plan,
            FakeBackend(),
            config,
            out,
            fixture.bindings,
            arm_mode="scripted_proposer",
            runtime_values=runtime,
        )

    assert calls == [0]
    assert not (out / "episode-01.meta.json").exists()
    for name in ("plan.json", "episodes.csv", "summary.json", "timings.json"):
        assert (out / name).is_file()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign_status"] == "incomplete_fail_stop"
    assert summary["planned_episodes"] == 18
    assert summary["n_episodes"] == 1
    assert summary["seeds"] == [0, 1, 2]
    assert len(summary["per_scene"]) == 6
    assert all(scene["seeds"] == [0, 1, 2] for scene in summary["per_scene"])
    assert ORIGINAL_EXCEPTION in summary["fail_stop"]["reason"]

    plan_payload = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert len(plan_payload["episodes"]) == 18
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["failure_state"] == "EXCEPTION"
    assert ORIGINAL_EXCEPTION in rows[0]["failure_reason"]
    timings = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    assert len(timings["episodes"]) == 1

    sidecars = out / "worldmind-sidecars"
    for name in (
        "provenance.json",
        "decisions.jsonl",
        "prediction_pairing.jsonl",
        "errors.jsonl",
        "risk_rejection.json",
        "phase_latency.json",
        "verification_recovery.json",
        "instrumentation.json",
    ):
        assert (sidecars / name).is_file()


def _task() -> Any:
    return SimpleNamespace(instruction="offline V2 postmortem", target="CAN", destination="target")
