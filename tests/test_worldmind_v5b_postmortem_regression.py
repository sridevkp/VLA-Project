"""Offline A-E regressions for the V5b postmortem (non-scored diagnostic evidence).

V5b remains read-only evidence. Nothing here starts Webots, resumes a campaign, writes into a
preserved run directory, scores a pipeline, selects a calibration or reads TRAIN/TEST. Every
numerical input below is either a frozen V5b measurement or a value already present in the
reviewed config/bindings.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from test_worldmind_v4_postmortem_regression import REPO_ROOT  # noqa: PLC2701

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
    _execute_observed_state_feedback_chunk,
    _load_feedback_contract,
    _precheck_observed_waypoint_reachability,
    run_stage1_evidence,
)
from vlarobot.risk.comparison import Selection

ARM_JOINTS = tuple(f"arm_{index}_joint" for index in range(1, 8))
ARM5 = "arm_5_joint"
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)

V5B_RESIDUAL_POSITION_M = 2.770961553780716
V5B_RESIDUAL_ORIENTATION_RAD = 0.3235694262270646
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


class _RecordingSolver:
    """Zero-residual solver that records the observed state used by each solve."""

    def __init__(self, joints: dict[str, float]) -> None:
        self.joints = dict(joints)
        self.calls: list[tuple[Pose3D, dict[str, float], int]] = []
        self.position_error_m = 0.0
        self.orientation_error_rad = 0.0
        self.execution_max_joint_step_rad = REVIEWED_STEP_RAD

    def solve(self, pose: Pose3D, state: Any, *, seed: int = 0) -> _RecordingSolver:
        self.calls.append((pose, dict(state.robot.joint_pos), seed))
        return self


class _ExplodingSolver:
    """Solver whose residual exception must remain a true campaign exception."""

    execution_max_joint_step_rad = REVIEWED_STEP_RAD

    def solve(self, pose: Pose3D, robot: Any, *, seed: int = 0) -> None:
        del pose, robot, seed
        raise RuntimeError("solver residual exploded")


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


def test_a_v5b_residual_becomes_structured_unreachable_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    solver = StaticPoseSolver({name: 0.0 for name in ARM_JOINTS})
    solver.position_error_m = V5B_RESIDUAL_POSITION_M
    solver.orientation_error_rad = V5B_RESIDUAL_ORIENTATION_RAD
    backend = FakeBackend()
    observation = backend.reset(_one_episode_plan()[0].scene, 0)
    ground_truth = backend.get_ground_truth()
    state = _world_state(observation, ground_truth, config, target=config.task.target)
    waypoints = (
        ScriptedWaypoint(
            "PREGRASP",
            1,
            pose=Pose3D(Vec3(0.5, 0.0, 0.5), IDENTITY_QUAT),
        ),
    )

    event = _precheck_observed_waypoint_reachability(
        config=config,
        solver=solver,
        waypoints=waypoints,
        state=state,
        episode_id="validation-clear-i00-s0",
        decision_id="validation-clear-i00-s0-d00",
        decision_index=0,
    )

    assert event is not None
    assert event["event_type"] == "unreachable_observed_target"
    assert event["seed"] == 0
    assert event["waypoint_label"] == "PREGRASP"
    assert event["position_error_m"] == V5B_RESIDUAL_POSITION_M
    assert event["orientation_error_rad"] == V5B_RESIDUAL_ORIENTATION_RAD
    assert event["position_tolerance_m"] == config.adapter.ik_position_tolerance_m
    assert event["orientation_tolerance_rad"] == config.adapter.ik_orientation_tolerance_rad
    assert event["candidate_generation_performed"] is False
    assert event["backend_actions_applied"] == 0
    assert "no clamp, substitution or actuation" in event["note"]


def test_a_precheck_uses_reviewed_tolerances_and_chain_without_actuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, _ = _setup_feedback(tmp_path, monkeypatch)
    solver = _RecordingSolver({name: 0.0 for name in ARM_JOINTS})
    backend = FakeBackend()
    observation = backend.reset(_one_episode_plan()[0].scene, 0)
    ground_truth = backend.get_ground_truth()
    state = _world_state(observation, ground_truth, config, target=config.task.target)
    endpoint = 0.1
    waypoints = (
        ScriptedWaypoint("JOINT", 1, joints={ARM5: endpoint}),
        ScriptedWaypoint("PREGRASP", 1, pose=Pose3D(Vec3(0.5, 0.0, 0.5), IDENTITY_QUAT)),
    )

    event = _precheck_observed_waypoint_reachability(
        config=config,
        solver=solver,
        waypoints=waypoints,
        state=state,
        episode_id="validation-clear-i00-s0",
        decision_id="validation-clear-i00-s0-d00",
        decision_index=0,
    )

    assert event is None
    assert len(solver.calls) == 1
    assert solver.calls[0][2] == 0
    assert solver.calls[0][1][ARM5] == pytest.approx(min(endpoint, REVIEWED_STEP_RAD))
    assert backend._seq == 0


def test_a_policy_records_unreachable_event_before_any_candidate_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch)
    fixture.solver.position_error_m = V5B_RESIDUAL_POSITION_M
    fixture.solver.orientation_error_rad = V5B_RESIDUAL_ORIENTATION_RAD
    backend = _IncrementalBackend(per_step_rad=REVIEWED_STEP_RAD)
    out = tmp_path / "precheck-abort"

    report, sidecars = run_stage1_evidence(
        _one_episode_plan(),
        backend,
        config,
        out,
        fixture.bindings,
        arm_mode="scripted_proposer",
        execution_mode=FEEDBACK_MODE,
        runtime_values=runtime,
    )

    assert report.summary["campaign_status"] == "complete"
    assert backend.applied == []
    assert _jsonl(sidecars["decisions"]) == []
    sidecar_folder = out / "worldmind-sidecars"
    assert _jsonl(sidecar_folder / FEEDBACK_SIDECAR_NAME) == []
    events = _jsonl(sidecar_folder / DIAGNOSTIC_EVENT_SIDECAR_NAME)
    assert len(events) == 1
    assert events[0]["event_type"] == "unreachable_observed_target"
    assert events[0]["position_error_m"] == V5B_RESIDUAL_POSITION_M
    expected_waypoint = next(
        item
        for item in fixture.bindings.waypoint_sets[config.adapter.waypoints_binding]
        if item.label == events[0]["waypoint_label"]
    )
    assert events[0]["position_tolerance_m"] == expected_waypoint.position_tolerance_m
    assert events[0]["orientation_tolerance_rad"] == expected_waypoint.orientation_tolerance_rad


def test_a_solver_exception_still_fails_the_campaign_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, config, runtime = _setup_feedback(tmp_path, monkeypatch)
    solver_key = config.adapter.pose_solver_binding
    fixture.bindings.pose_solvers[solver_key] = _ExplodingSolver()
    out = tmp_path / "precheck-exception"

    with pytest.raises(CampaignFailStop, match="solver residual exploded"):
        run_stage1_evidence(
            _one_episode_plan(),
            FakeBackend(),
            config,
            out,
            fixture.bindings,
            arm_mode="scripted_proposer",
            execution_mode=FEEDBACK_MODE,
            runtime_values=runtime,
        )

    errors = _jsonl(out / "worldmind-sidecars" / "errors.jsonl")
    assert errors[0]["error_type"] == "RuntimeError"
    assert "solver residual exploded" in errors[0]["error"]


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
    assert _jsonl(sidecars["observed_state_feedback"])[0]["schema_version"] == 2
    assert any(
        item["event_type"] == "diagnostic_home_completion"
        for item in _jsonl(sidecars["diagnostic_events"])
    )
    proxy = _jsonl(sidecars["collision_proxy_attribution"])[0]
    assert proxy["execution_mode"] == FEEDBACK_MODE
    assert proxy["objects"]
