"""Offline A-E regressions for the V5b postmortem (non-scored diagnostic evidence).

V5b remains read-only evidence. Nothing here starts Webots, resumes a campaign, writes into a
preserved run directory, scores a pipeline, selects a calibration or reads TRAIN/TEST. Every
fixture distinguishes injected IK residuals from preserved V5b measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from test_worldmind_observed_state_feedback import (  # noqa: PLC2701 - existing shared helpers
    FEEDBACK_MODE,
    _IncrementalBackend,
    _jsonl,
    _setup_feedback,
)
from test_worldmind_production import (  # noqa: PLC2701 - existing shared solver stub
    StaticPoseSolver,
    _one_episode_plan,
)
from test_worldmind_v4_postmortem_regression import REPO_ROOT, _tree_hash  # noqa: PLC2701

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import MAX_JOINT_STEP_RAD as REVIEWED_STEP_RAD
from vlarobot.core.types import ApplyResult, GripperCmd, GroundTruth, JointTarget, Pose3D, Vec3
from vlarobot.eval.decision_adapters import BackendApplyProbe, SafetyMonitorProbe
from vlarobot.eval.observed_state_feedback import (
    DIAGNOSTIC_EVENT_SIDECAR_NAME,
    FEEDBACK_SIDECAR_NAME,
    build_collision_proxy_attribution,
    read_feedback_evidence,
    validate_feedback_evidence_row,
)
from vlarobot.eval.runner import CampaignFailStop
from vlarobot.eval.settings import load_eval_settings
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.execution.worldmind_pipeline import _world_state
from vlarobot.execution.worldmind_stage1_pipeline import (
    Stage1WorldMindEpisodePolicy,
    _execute_observed_state_feedback_chunk,
    _load_feedback_contract,
    run_stage1_evidence,
)
from vlarobot.risk.comparison import Selection

ARM_JOINTS = tuple(f"arm_{index}_joint" for index in range(1, 8))
ARM5 = "arm_5_joint"
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)

INJECTED_RESIDUAL_POSITION_M = 2.770961553780716
INJECTED_RESIDUAL_ORIENTATION_RAD = 0.3235694262270646
V5B_HOME_JOINT_RAD = 1.0899390970577643
V5B_CONFIGURED_HOME_RAD = 4.885743745552387e-06
V5B_SELECTED_TARGET_ERROR_RAD = 0.018605150431497686
V5B_NON_CONVERGED_ERROR_RAD = 0.1409631703015909


def _with_approved_verification(config: Any) -> Any:
    """Use the frozen approved HOME vector and tolerance, never a test-only substitute."""
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/worldmind_diagnostic.approved.yaml").read_text(encoding="utf-8")
    )
    approved = raw["verification"]
    return replace(
        config,
        verification=replace(
            config.verification,
            home_joints={
                str(name): float(value) for name, value in approved["home_joints"].items()
            },
            home_tolerance_rad=float(approved["home_tolerance_rad"]),
        ),
    )


class _SeedSensitiveSolver(StaticPoseSolver):
    seed_calls: list[int]
    failed_seed: int
    failure: str

    def solve(self, pose: Pose3D, state: Any, *, seed: int = 0) -> Any:
        self.seed_calls.append(seed)
        if seed == self.failed_seed:
            if self.failure == "solver":
                raise RuntimeError("solver residual exploded")
            return SimpleNamespace(
                joints=self.joints,
                position_error_m=(
                    math.nan if self.failure == "nonfinite" else INJECTED_RESIDUAL_POSITION_M
                ),
                orientation_error_rad=INJECTED_RESIDUAL_ORIENTATION_RAD,
            )
        return super().solve(pose, state, seed=seed)


def _install_solver(
    fixture: Any, config: Any, failed_seed: int, failure: str
) -> _SeedSensitiveSolver:
    solver = _SeedSensitiveSolver(dict(fixture.solver.joints))
    solver.seed_calls = []
    solver.failed_seed = failed_seed
    solver.failure = failure
    fixture.bindings.pose_solvers[config.adapter.pose_solver_binding] = solver
    return solver


def _pose_waypoints(fixture: Any, config: Any) -> list[ScriptedWaypoint]:
    return [
        waypoint
        for waypoint in fixture.bindings.waypoint_sets[config.adapter.waypoints_binding]
        if waypoint.pose is not None
    ]


def _approved_draw_seeds() -> tuple[int, list[int]]:
    decision_seed = int(np.random.SeedSequence([0, 0]).generate_state(1)[0])
    assert decision_seed == 2968811710
    return decision_seed, np.random.default_rng(decision_seed).integers(0, 2**31, size=32).tolist()


class _RejectingMonitor:
    def validate(self, action: Any, robot: Any) -> ApplyResult:
        del action, robot
        return ApplyResult(False, "synthetic safety rejection")


class _ApplyRejectingBackend(FakeBackend):
    def apply(self, action: Any) -> ApplyResult:
        del action
        return ApplyResult(False, "synthetic backend rejection")


class _ExplodingStepBackend(FakeBackend):
    def step(self, n_steps: int = 1) -> Any:
        del n_steps
        raise RuntimeError("synthetic backend step exception")


def _waypoints(*labels: str) -> tuple[ScriptedWaypoint, ...]:
    return tuple(
        ScriptedWaypoint(label, 1, joints={name: 0.0 for name in ARM_JOINTS}) for label in labels
    )


def _direct(
    *,
    fixture: Any,
    config: Any,
    backend: Any,
    actions: tuple[Any, ...],
    waypoints: tuple[ScriptedWaypoint, ...],
    monitor: Any | None = None,
    observation: Any | None = None,
    episode_steps_before: int = 0,
) -> Any:
    if observation is None:
        observation = backend.reset(_one_episode_plan()[0].scene, 0)
    ground_truth = backend.get_ground_truth()
    state = _world_state(observation, ground_truth, config, target=config.task.target)
    if monitor is None:
        monitor = fixture.bindings.safety[config.safety.binding](config, {})
    selection = Selection(
        SimpleNamespace(actions=actions, source="scripted"),
        0,
        "offline V5b postmortem regression",
    )

    return _execute_observed_state_feedback_chunk(
        decision_id="postmortem-d00",
        episode_id="validation-clear-i00-s0",
        selection=selection,
        config=config,
        contract=_load_feedback_contract(config, fixture.solver),
        task=SimpleNamespace(
            instruction=config.task.instruction,
            target=config.task.target,
            destination=config.task.destination,
        ),
        observation=observation,
        state=state,
        ground_truth=ground_truth,
        backend_probe=BackendApplyProbe(backend),
        monitor_probe=SafetyMonitorProbe(monitor),
        waypoints=waypoints,
        episode_steps_before=episode_steps_before,
    )


def test_a_seed_zero_cannot_veto_and_successful_pools_match_later_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch, max_decisions=3)
    solver = _install_solver(fixture, config, 0, "finite")
    # Force re-observation to exercise fresh generator construction without moving the robot.
    captured: list[Any] = []
    original_select = Stage1WorldMindEpisodePolicy._select

    def defer_selection(policy: Any, candidates: Any, predictions: Any, assessments: Any) -> Any:
        selection = original_select(policy, candidates, predictions, assessments)
        assert selection.index == 0
        captured.append([asdict(chunk) for chunk in candidates])
        return replace(selection, chunk=None, index=None)

    monkeypatch.setattr(Stage1WorldMindEpisodePolicy, "_select", defer_selection)
    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        FakeBackend(),
        config,
        tmp_path / "seed-parity",
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode=FEEDBACK_MODE,
        runtime_values=runtime,
    )
    decision_seed, draw_seeds = _approved_draw_seeds()
    decisions = _jsonl(sidecars["decisions"])
    assert [row["decision_index"] for row in decisions][:2] == [0, 1]
    assert all(row["draw_seed"] == decision_seed for row in decisions)
    assert all(row["draw_seeds"] == draw_seeds for row in decisions)
    assert all(
        [candidate["candidate_id"] for candidate in row["candidates"]]
        == [f"candidate-{slot:02d}" for slot in range(32)]
        for row in decisions
    )
    pose_count = len(_pose_waypoints(fixture, config))
    assert solver.seed_calls == [seed for seed in draw_seeds for _ in range(pose_count)] * len(
        decisions
    )
    assert captured[0] == captured[1]
    assert all(len(chunk["actions"]) == 50 for chunk in captured[0])
    events = _jsonl(sidecars["diagnostic_events"])
    assert all(row["event_type"] == "diagnostic_reobserve" for row in events)


@pytest.mark.parametrize(
    ("slot", "failure", "classification"),
    [
        (0, "finite", "candidate_ik_residual_outside_tolerance"),
        (5, "finite", "candidate_ik_residual_outside_tolerance"),
        (0, "nonfinite", "candidate_ik_nonfinite_residual"),
        (0, "solver", "candidate_ik_solver_exception"),
    ],
)
def test_a_failed_draw_stops_pool_and_reports_actual_ik_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slot: int, failure: str, classification: str
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch)
    _, draw_seeds = _approved_draw_seeds()
    solver = _install_solver(fixture, config, draw_seeds[slot], failure)
    backend = _IncrementalBackend(per_step_rad=REVIEWED_STEP_RAD)
    out = tmp_path / "failed-draw"
    reason = "solver residual exploded" if failure == "solver" else "outside reviewed tolerance"
    with pytest.raises(CampaignFailStop, match=reason):
        run_stage1_evidence(
            _one_episode_plan(),
            backend,
            config,
            out,
            fixture.bindings,
            arm_mode="scripted_proposer",
            execution_mode=FEEDBACK_MODE,
            runtime_values=runtime,
        )

    pose_waypoints = _pose_waypoints(fixture, config)
    assert solver.seed_calls == [seed for seed in draw_seeds[:slot] for _ in pose_waypoints] + [
        draw_seeds[slot]
    ]
    assert backend.applied == []
    folder = out / "worldmind-sidecars"
    assert _jsonl(folder / "decisions.jsonl") == []
    assert _jsonl(folder / "prediction_pairing.jsonl") == []
    assert _jsonl(folder / FEEDBACK_SIDECAR_NAME) == []
    errors = _jsonl(folder / "errors.jsonl")
    assert errors[0]["error_type"] == ("RuntimeError" if failure == "solver" else "ValueError")
    (event,) = _jsonl(folder / DIAGNOSTIC_EVENT_SIDECAR_NAME)
    waypoint = pose_waypoints[0]
    assert event["schema_version"] == 1
    assert event["event_type"] == classification
    assert event["episode_id"] == _one_episode_plan()[0].scene.name
    assert event["decision_id"] == f"{event['episode_id']}-d00"
    assert event["draw_slot"] == slot
    assert event["draw_seed"] == draw_seeds[slot]
    assert event["waypoint_label"] == waypoint.label
    assert event["target_pose_base"] == json.loads(json.dumps(asdict(waypoint.pose)))
    assert event["position_tolerance_m"] == waypoint.position_tolerance_m
    assert event["orientation_tolerance_rad"] == waypoint.orientation_tolerance_rad
    assert event["backend_actions_applied"] == 0
    assert event["position_error_m"] == (
        INJECTED_RESIDUAL_POSITION_M if failure == "finite" else None
    )
    assert event["orientation_error_rad"] == (
        None if failure == "solver" else INJECTED_RESIDUAL_ORIENTATION_RAD
    )
    assert event["solver_error"] == (reason if failure == "solver" else None)


def test_a_preserved_v5b_tree_and_historical_reader() -> None:
    root = REPO_ROOT / "runs/worldmind/stage1-validation-v5b-2026-11-21"
    assert _tree_hash(root, REPO_ROOT) == (
        "1946ff3ab5ce461ad92f3a354df4d292ecf469af2d6e7a77bd1322c530eae828"
    )
    rows = read_feedback_evidence(root / "worldmind-sidecars" / FEEDBACK_SIDECAR_NAME)
    assert rows and all(row["schema_version"] == 1 for row in rows)
    errors = _jsonl(root / "worldmind-sidecars" / "errors.jsonl")
    assert errors == [
        {
            "episode_id": "validation-clear-i03-s0",
            "error": "scripted pose residual outside reviewed tolerance",
            "error_type": "ValueError",
        }
    ]


def test_b_diagnostic_home_uses_configured_vector_and_is_separately_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    config = _with_approved_verification(config)
    result = _direct(
        fixture=fixture,
        config=config,
        backend=_IncrementalBackend(per_step_rad=REVIEWED_STEP_RAD),
        actions=(JointTarget({ARM5: 0.1}),),
        waypoints=_waypoints("PREGRASP"),
    )

    home = result.diagnostic.home_completion
    assert home is not None
    assert home.phase == "diagnostic_home_completion"
    assert home.source.endswith("verification.home_joints")
    assert dict(home.target_joint_pos_rad) == dict(config.verification.home_joints)
    assert home.tolerance_rad == config.verification.home_tolerance_rad == 0.02
    assert home.terminal_reason == "home_converged"
    assert home.final_max_joint_error_rad <= 0.02
    assert result.diagnostic.actions[0].selected_action == {"q": {ARM5: 0.1}}
    assert all(
        result.diagnostic.attempts[index].phase == "diagnostic_home_completion"
        for index in home.attempt_indices
    )
    assert result.diagnostic.attempts[0].phase == "selected_chunk"
    assert result.diagnostic.prediction_provenance.pairing_records_emitted == 0


def test_b_v5b_selected_home_error_is_preserved_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    config = _with_approved_verification(config)
    selected_target = V5B_HOME_JOINT_RAD + V5B_SELECTED_TARGET_ERROR_RAD
    backend = _IncrementalBackend(
        per_step_rad=REVIEWED_STEP_RAD, initial_joint_pos={ARM5: V5B_HOME_JOINT_RAD}
    )
    result = _direct(
        fixture=fixture,
        config=config,
        backend=backend,
        actions=(JointTarget({ARM5: selected_target}),),
        waypoints=_waypoints("HOME"),
    )

    selected = result.diagnostic.actions[0]
    assert selected.home_action is True
    assert selected.selected_action["q"][ARM5] == selected_target
    assert selected.terminal_reason == "target_converged"
    assert selected.final_max_joint_error_rad == pytest.approx(V5B_SELECTED_TARGET_ERROR_RAD)
    assert selected.backend_step_count == 0
    home = result.diagnostic.home_completion
    assert home is not None
    assert home.initial_max_joint_error_rad == pytest.approx(
        V5B_HOME_JOINT_RAD - V5B_CONFIGURED_HOME_RAD
    )
    assert home.final_max_joint_error_rad <= 0.02
    assert home.backend_step_count > 0


def test_c_global_step_ceiling_excludes_non_converged_error_from_tolerance_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    second_target = REVIEWED_STEP_RAD + V5B_NON_CONVERGED_ERROR_RAD
    result = _direct(
        fixture=fixture,
        config=config,
        backend=_IncrementalBackend(per_step_rad=REVIEWED_STEP_RAD),
        actions=(JointTarget({ARM5: 0.0}), JointTarget({ARM5: second_target})),
        waypoints=_waypoints("PREGRASP", "PREGRASP"),
        episode_steps_before=1058,
    )

    summary = result.diagnostic.convergence_summary
    assert result.diagnostic.termination_reason == "global_step_ceiling"
    assert result.diagnostic.actions[1].terminal_reason == "global_step_ceiling"
    assert result.diagnostic.actions[1].final_max_joint_error_rad == pytest.approx(
        V5B_NON_CONVERGED_ERROR_RAD
    )
    assert summary.converged_action_indices == (0,)
    assert summary.max_final_error_converged_rad == 0.0
    assert summary.non_converged_action_indices == (1,)
    assert summary.non_converged_reasons == {"global_step_ceiling": 1}
    assert summary.non_converged_final_errors == (
        (1, "global_step_ceiling", pytest.approx(V5B_NON_CONVERGED_ERROR_RAD)),
    )
    assert result.backend_step_count == 1
    assert result.diagnostic.episode_steps_after == 1059


def test_c_safety_rejection_and_backend_rejection_and_exception_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    action = JointTarget({ARM5: 0.1})

    rejected = _direct(
        fixture=fixture,
        config=config,
        backend=FakeBackend(),
        actions=(action,),
        waypoints=_waypoints("PREGRASP"),
        monitor=_RejectingMonitor(),
    )
    backend_rejected = _direct(
        fixture=fixture,
        config=config,
        backend=_ApplyRejectingBackend(),
        actions=(action,),
        waypoints=_waypoints("PREGRASP"),
    )
    exception = _direct(
        fixture=fixture,
        config=config,
        backend=_ExplodingStepBackend(),
        actions=(action,),
        waypoints=_waypoints("PREGRASP"),
    )

    assert rejected.diagnostic.termination_reason == "safety_rejected"
    assert rejected.diagnostic.safety_rejected_count == 1
    assert rejected.diagnostic.apply_accepted_count == 0
    assert backend_rejected.diagnostic.termination_reason == "backend_apply_rejected"
    assert backend_rejected.diagnostic.apply_rejected_count == 1
    assert backend_rejected.backend_step_count == 0
    assert exception.diagnostic.termination_reason == "execution_exception"
    assert "step exception" in (exception.diagnostic.error or "")
    assert {
        rejected.diagnostic.actions[0].terminal_reason,
        backend_rejected.diagnostic.actions[0].terminal_reason,
        exception.diagnostic.actions[0].terminal_reason,
    } == {"safety_rejected", "backend_apply_rejected", "execution_exception"}


def test_d_gripper_completion_uses_reviewed_rate_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    backend = FakeBackend(gripper_rate_limited=True, finger_speed_m_s=0.03)
    result = _direct(
        fixture=fixture,
        config=config,
        backend=backend,
        actions=(GripperCmd(opening=0.0),),
        waypoints=_waypoints("HOME"),
    )

    record = result.diagnostic.actions[0]
    assert record.terminal_reason == "gripper_command_complete"
    assert record.completion_target == 0.0
    assert 0.0 < record.completion_error <= result.diagnostic.budgets.gripper_arrival_tolerance_m
    assert record.completion_unit == "m"
    assert record.backend_step_count == 46
    assert result.diagnostic.budgets.finger_speed_m_s == 0.03
    assert result.diagnostic.budgets.gripper_arrival_tolerance_m == pytest.approx(0.00096)
    assert result.diagnostic.budgets.gripper_timeout_steps == 125
    assert all(item.safety_accepted is True for item in result.diagnostic.attempts)
    assert all(item.apply_accepted is True for item in result.diagnostic.attempts)
    assert "grasp" not in record.selected_action_type.lower()
    assert "release" not in record.selected_action_type.lower()


def test_d_collision_proxy_attribution_is_per_object_and_not_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del tmp_path, monkeypatch
    settings = load_eval_settings()
    first = GroundTruth(
        object_poses={
            "CAN": Pose3D(Vec3(0.0, 0.0, 0.0), IDENTITY_QUAT),
            "BOX": Pose3D(Vec3(1.0, 0.0, 0.0), IDENTITY_QUAT),
        }
    )
    second = GroundTruth(
        object_poses={
            "CAN": Pose3D(Vec3(0.01, 0.0, 0.0), IDENTITY_QUAT),
            "BOX": Pose3D(Vec3(1.001, 0.0, 0.0), IDENTITY_QUAT),
        }
    )
    observations = (
        SimpleNamespace(robot=SimpleNamespace(diagnostics={})),
        SimpleNamespace(robot=SimpleNamespace(diagnostics={"contact_nodes": ["finger"]})),
    )

    evidence = build_collision_proxy_attribution(
        episode_id="validation-clear-i00-s0",
        ground_truths=(first, second),
        observations=observations,
        threshold_source="configs/eval.yaml:collision_proxy",
        max_displacement_m=settings.collision_proxy_max_displacement_m,
        max_tilt_deg=settings.collision_proxy_max_tilt_deg,
    )

    assert settings.collision_proxy_max_displacement_m == 0.005
    assert settings.collision_proxy_max_tilt_deg == 5.0
    assert evidence.schema_version == 1
    assert [item.name for item in evidence.objects] == ["BOX", "CAN"]
    can = next(item for item in evidence.objects if item.name == "CAN")
    box = next(item for item in evidence.objects if item.name == "BOX")
    assert can.displacement_hit is True
    assert box.displacement_hit is False
    assert evidence.proxy_hit is True
    assert evidence.backend_contact_steps_observed == 1
    assert "never promoted" in evidence.contact_channel
    assert "no grasp, release" in evidence.note


def test_e_full_observed_base_pose_quaternion_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    backend = FakeBackend()
    observation = backend.reset(_one_episode_plan()[0].scene, 0)
    qx, qy, qz = 0.01, -0.02, 0.03
    qw = math.sqrt(1.0 - qx * qx - qy * qy - qz * qz)
    quaternion = (qx, qy, qz, qw)
    backend._robot_state.base_pose_world = Pose3D(Vec3(1.0, 2.0, 3.0), quaternion)
    observation = backend._observation()

    result = _direct(
        fixture=fixture,
        config=config,
        backend=backend,
        actions=(GripperCmd(opening=0.045),),
        waypoints=_waypoints("HOME"),
        observation=observation,
    )

    for attempt in result.diagnostic.attempts:
        for snapshot in (attempt.observed_before, attempt.observed_after):
            if snapshot is None:
                continue
            assert snapshot["base_pose_world"]["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
            assert snapshot["base_pose_world"]["orientation"] == quaternion


def test_e_schema_1_rows_remain_readable_alongside_schema_2(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    schema_1 = {
        "schema_version": 1,
        "execution_mode": FEEDBACK_MODE,
        "termination_reason": "selected_chunk_converged",
    }
    schema_2 = {
        "schema_version": 2,
        "execution_mode": FEEDBACK_MODE,
        "termination_reason": "global_step_ceiling",
        "convergence_summary": {"non_converged_reasons": {"global_step_ceiling": 1}},
        "home_completion": {"phase": "diagnostic_home_completion"},
    }
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (schema_1, schema_2)) + "\n",
        encoding="utf-8",
    )

    rows = read_feedback_evidence(path)
    assert [row["schema_version"] for row in rows] == [1, 2]
    assert rows[0] == schema_1
    assert validate_feedback_evidence_row(rows[1]) == schema_2


def test_e_wiring_writes_all_versioned_diagnostic_sidecars_without_pairings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch)
    out = tmp_path / "sidecars"
    _, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        _IncrementalBackend(per_step_rad=REVIEWED_STEP_RAD),
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode=FEEDBACK_MODE,
        runtime_values=runtime,
    )

    assert sidecars["observed_state_feedback"].name == FEEDBACK_SIDECAR_NAME
    assert sidecars["diagnostic_events"].name == "diagnostic-events.jsonl"
    assert sidecars["collision_proxy_attribution"].name == "collision-proxy-attribution.jsonl"
    assert _jsonl(sidecars["prediction_pairing"]) == []
    assert _jsonl(sidecars["observed_state_feedback"])[0]["schema_version"] == 3
    assert any(
        item["event_type"] == "diagnostic_home_completion"
        for item in _jsonl(sidecars["diagnostic_events"])
    )
    proxy = _jsonl(sidecars["collision_proxy_attribution"])[0]
    assert proxy["execution_mode"] == FEEDBACK_MODE
    assert proxy["objects"]


@pytest.mark.parametrize(
    ("readings", "defect"),
    [
        ({0: 0.0, 1: 0.0}, None),
        ({}, "missing"),
        ({0: 0.0}, "missing"),
        ({0: 0.0, 1: float("nan")}, "invalid_or_nonfinite"),
        ({0: 0.0, 1: -0.02}, "outside_reviewed_limits"),
    ],
)
def test_feedback_gripper_requires_two_valid_raw_sensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readings: dict[int, float], defect: str | None
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    config = replace(config, backend=replace(config.backend, kind="webots"))
    contract = _load_feedback_contract(config, fixture.solver)
    raw = {contract.finger_names[index]: value for index, value in readings.items()}

    class SensorBackend(FakeBackend):
        def _observation(self) -> Any:
            obs = super()._observation()
            return replace(
                obs,
                robot=replace(
                    obs.robot, gripper_opening=0.0, diagnostics={"finger_joint_sensor_rad": raw}
                ),
            )

    result = _direct(
        fixture=fixture,
        config=config,
        backend=SensorBackend(),
        actions=(GripperCmd(0.0),),
        waypoints=_waypoints("CLOSE"),
    )
    if defect is None:
        assert result.diagnostic.actions[0].terminal_reason == "gripper_command_complete"
        assert result.diagnostic.apply_accepted_count == result.backend_step_count == 1
    else:
        assert result.diagnostic.actions[0].terminal_reason == "execution_exception"
        assert result.diagnostic.apply_accepted_count == result.backend_step_count == 0
        (event,) = result.diagnostic_events
        assert event["event_type"] == "invalid_gripper_telemetry"
        assert defect in event["sensor_defects"].values()


@pytest.mark.parametrize(
    ("fault", "expected_apply", "expected_step", "expected_observation", "steps"),
    [
        ("none", "accepted", "confirmed", "available", 1),
        ("reject", "rejected", "not_attempted", "not_available", 0),
        ("apply_unknown", "unknown", "not_attempted", "not_available", 0),
        ("unexecuted", "accepted", "not_executed", "not_available", 0),
        ("step_lost", "accepted", "unknown", "not_available", 0),
        ("observation", "accepted", "confirmed", "failed", 1),
        ("record_apply", "accepted", "not_attempted", "not_available", 0),
        ("record_step", "accepted", "confirmed", "available", 1),
    ],
)
def test_diagnostic_outcomes_keep_backend_facts_when_later_operations_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    expected_apply: str,
    expected_step: str,
    expected_observation: str,
    steps: int,
) -> None:
    from vlarobot.eval.recorder import RecordingBackend

    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    calls: list[str] = []

    class InjectedBackend(FakeBackend):
        def apply(self, action: Any) -> ApplyResult:
            calls.append("apply")
            if fault == "apply_unknown":
                raise TimeoutError("lost apply reply")
            if fault == "reject":
                return ApplyResult(False, "rejected")
            return super().apply(action)

        def diagnostic_step(self) -> dict[str, Any]:
            calls.append("step")
            if fault == "unexecuted":
                return {"confirmed_ticks": 0, "error": "stopped"}
            observation = self.step(1)
            if fault == "step_lost":
                raise TimeoutError("lost step reply after possible advancement")
            if fault == "observation":
                return {"confirmed_ticks": 1, "error": "observation construction failed"}
            return {"confirmed_ticks": 1, "observation": observation}

    def record_action(*args: Any) -> None:
        if fault == "record_apply":
            raise OSError("apply recorder failed")

    def record_step(observation: Any, ground_truth: Any) -> None:
        if fault == "record_step" and observation.t > 0:
            raise OSError("step recorder failed")

    recorder = SimpleNamespace(
        on_reset=lambda *args: None, record_action=record_action, record_step=record_step
    )
    result = _direct(
        fixture=fixture,
        config=config,
        backend=RecordingBackend(InjectedBackend(), recorder),
        actions=(GripperCmd(0.0),),
        waypoints=_waypoints("CLOSE"),
        episode_steps_before=1058,
    )
    evidence = result.diagnostic
    (attempt,) = evidence.attempts
    assert (attempt.apply_outcome, attempt.step_outcome, attempt.observation_outcome) == (
        expected_apply,
        expected_step,
        expected_observation,
    )
    assert calls.count("apply") == 1
    assert calls.count("step") == int(expected_step != "not_attempted")
    assert evidence.apply_accepted_count == int(expected_apply == "accepted")
    assert evidence.apply_rejected_count == int(expected_apply == "rejected")
    assert evidence.apply_unknown_count == int(expected_apply == "unknown")
    assert evidence.step_unknown_count == int(expected_step == "unknown")
    assert evidence.observation_failure_count == int(expected_observation == "failed")
    assert evidence.recording_failure_count == int(fault.startswith("record_"))
    assert evidence.backend_step_count == steps
    assert evidence.episode_steps_after == 1058 + steps
    assert evidence.observation_count == 1 + int(expected_observation == "available")
    assert (attempt.observed_after is not None) == (expected_observation == "available")


def test_lost_step_reply_ends_stage1_feedback_without_another_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch)
    calls: list[str] = []

    class LostResponseBackend(_IncrementalBackend):
        def apply(self, action: Any) -> ApplyResult:
            calls.append("apply")
            return super().apply(action)

        def diagnostic_step(self) -> dict[str, Any]:
            calls.append("step")
            self.step(1)
            raise TimeoutError("lost diagnostic step response")

    backend = LostResponseBackend(per_step_rad=REVIEWED_STEP_RAD)
    out = tmp_path / "lost-reply"
    with pytest.raises(CampaignFailStop, match="lost diagnostic step response"):
        run_stage1_evidence(
            _one_episode_plan(),
            backend,
            config,
            out,
            fixture.bindings,
            arm_mode="scripted_proposer",
            execution_mode=FEEDBACK_MODE,
            runtime_values=runtime,
        )
    (row,) = _jsonl(out / "worldmind-sidecars" / FEEDBACK_SIDECAR_NAME)
    assert calls == ["apply", "step"]
    assert row["apply_accepted_count"] == 1
    assert row["backend_step_count"] == 0
    assert row["step_unknown_count"] == 1
    assert row["observation_count"] == 1
    assert _jsonl(out / "worldmind-sidecars" / "prediction_pairing.jsonl") == []
