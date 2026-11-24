from dataclasses import replace

import pytest

from vlarobot.core.types import (
    ActionChunk,
    GripperCmd,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_objectives import MissingObjectiveInput
from vlarobot.eval.worldmind_proxy_policy import (
    GRID,
    HARD_CHECKS,
    assess_proxy,
    observe_proxy_clearance,
    select_policy,
    selections_by_policy,
)
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.rollout import AnalyticSurrogate


def box(name, x):
    return ObjectState(name, name, Pose3D(Vec3(x, 0, 0), (0, 0, 0, 1)), Vec3(0.1, 0.1, 0.1), 1.0)


def test_proxy_offline_positive_and_negative_controls():
    tool = Pose3D(Vec3(0, 0, 0), (0, 0, 0, 1))
    robot = RobotState({}, {}, Pose2D(0, 0, 0), tool, 0)
    chunk = ActionChunk([GripperCmd(0)] * 50, 0.032, "scripted")
    predictor = AnalyticSurrogate(
        tracking=None,
        inverse=lambda *_: None,
        gripper_speed_m_s=0.03,
        gravity_m_s2=9.81,
        support_z_m={"CAN": -0.05},
        validate=lambda *_: True,
    )
    assessments = []
    for x in (0, 1):
        state = WorldState(SceneState([box("CAN", x)]), robot)
        prediction = predictor.predict_trajectory(state, chunk)
        assert "unverified-arm-sweep" in prediction.hard_reasons
        arguments = dict(
            table_boxes=(box("TABLE", 2),),
            expected_object_ids=frozenset({"CAN"}),
            joint_limits_pass=True,
            ik_pass=True,
        )
        evidence = observe_proxy_clearance(state, chunk, prediction, **arguments)
        assessments.append(
            assess_proxy(
                state,
                chunk,
                prediction,
                evidence,
                placement_state_index=43,
                placement_tool_goal_base=tool.position,
                placement_tolerance_m=0.10,
            )
        )
        with pytest.raises(MissingObjectiveInput, match="identities"):
            observe_proxy_clearance(
                replace(state, scene=SceneState([])), chunk, prediction, **arguments
            )
    gate = RiskGate(weights=GRID["equal"], maximize=False, required_constraints=HARD_CHECKS)
    assert gate.select(assessments[:1]).selected_index is None
    assert gate.select(assessments).selected_index == 1
    assert assessments[0].components[2] == 1.0


def test_policy_difference_and_complete_outcomes():
    checks = tuple((name, True) for name in HARD_CHECKS)
    pool = (
        CandidateAssessment(checks, (0.1, 0.25, 0, 0)),
        CandidateAssessment(checks, (0.2, 0.1, 0, 0)),
    )
    assert selections_by_policy((pool,)) == {"equal": (1,), "goal_x2": (0,)}
    from vlarobot.eval.runner import build_episode_plan

    ids = tuple(entry.scene.name for entry in build_episode_plan("validation", 0))
    outcomes = {policy: dict.fromkeys(ids, False) for policy in GRID}
    assert select_policy(ids, outcomes) == ("equal", {"equal": 0.0, "goal_x2": 0.0})
    outcomes["equal"].pop(ids[0])
    with pytest.raises(ValueError, match="incomplete"):
        select_policy(ids, outcomes)


def test_same_seed_retry_and_reliable_veto(tmp_path):
    from vlarobot.execution.worldmind_nov24_campaign import run_trial

    calls = []

    def attempt(path, episode_id):
        calls.append((path.name, episode_id))
        return "tick_unknown" if path.name == "attempt-1" else "NO_SAFE_SET"

    result = run_trial(tmp_path, "validation-example-s2", 2, attempt)
    assert calls == [("attempt-1", "validation-example-s2"), ("attempt-2", "validation-example-s2")]
    assert result["seed"] == 2 and result["eligible"] and result["success"] is False
    assert not result["attempts"][0]["eligible"]


def test_finger_excursion_annotation_preserves_strict_qualification():
    from vlarobot.execution.worldmind_heuristic_qualification import (
        _raw,
        finger_log_status,
    )

    assert finger_log_status(-7.142594888992681e-7) == (
        "outside_motor_range_within_logging_tolerance"
    )
    assert not _raw(-7.142594888992681e-7)
    assert finger_log_status(-0.0002) == "outside_logging_tolerance"
    assert finger_log_status(float("nan")) == "invalid"


@pytest.mark.parametrize("profiles", [(False, False), (True, True), (False, True)])
def test_stateful_freeze_compares_two_eligible_full_system_campaigns(
    tmp_path, monkeypatch, profiles
):
    from datetime import date

    from vlarobot.execution import worldmind_nov24_campaign as campaign
    from vlarobot.execution.worldmind_idle_fingers import record

    class ValidationDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 25)

    monkeypatch.setattr(campaign, "date", ValidationDate)
    checked = []
    monkeypatch.setattr(campaign, "verify_policy_contrast", lambda *args: checked.append(args))
    plan = campaign.build_episode_plan("validation", 0)
    roots = []
    for (policy, capacity), home in zip((("equal", 11), ("goal_x2", 12)), profiles, strict=True):
        run = tmp_path / policy
        roots.append(run)
        run.mkdir()
        record(
            run / "launch.json",
            {
                "arm": "full_system",
                "split": "validation",
                "proxy_policy": policy,
                "proxy_config_sha256": campaign.MATCHED_PROXY_SHA,
                "executor_config_sha256": campaign.executor_profile_sha(
                    campaign.STATEFUL_PROXY_PROFILES[policy][1], home
                ),
                "home_profile_sha256": campaign.VERIFIED_HOME_PROFILE_SHA if home else None,
                "plan_sha256": campaign._PLAN,
                "config_sha256": campaign._CONFIG,
            },
        )
        results = []
        for index, entry in enumerate(plan):
            row = {"episode_id": entry.scene.name, "eligible": True, "success": index < capacity}
            folder = run / entry.scene.name
            folder.mkdir()
            record(folder / "result.json", row)
            results.append(row)
        record(run / "terminal.json", {"eligible": 18, "results": results})

    with pytest.raises(ValueError, match="both approved"):
        campaign.freeze_validation(tmp_path / "partial.json", roots[:1])
    out = tmp_path / "freeze.json"
    if profiles[0] != profiles[1]:
        with pytest.raises(ValueError, match="incompatible"):
            campaign.freeze_validation(out, roots)
        assert not out.exists()
        return
    assert campaign.freeze_validation(out, roots) == "goal_x2"
    assert checked == [(roots[0], "goal_x2", campaign.MATCHED_PROXY_SHA)]
    payload = campaign._read_event(out)
    assert payload["schema_version"] == 2
    assert payload["executor_config_sha256"] == campaign.executor_profile_sha(
        campaign.STATEFUL_PROXY_PROFILES["goal_x2"][1], profiles[1]
    )
    assert payload["scores"] == {"equal": 11 / 18, "goal_x2": 12 / 18}
