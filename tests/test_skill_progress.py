import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vlarobot.backends.fake.backend import FakeBackend
from vlarobot.backends.webots.config import GRIPPER_OPEN
from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.skill_empirical import measured_utility
from vlarobot.execution.skill_live import grasp_evidence


def test_current_grasp_evidence_requires_measured_aperture_and_alignment() -> None:
    obs = FakeBackend().reset(ScenarioSpec(), 0)
    aligned = obs.robot.ee_pose.position.as_array()
    open_hand = replace(
        obs, robot=replace(obs.robot, gripper_holding=None, gripper_opening=GRIPPER_OPEN)
    )
    partially_closed = replace(open_hand, robot=replace(open_hand.robot, gripper_opening=0.03))
    assert not grasp_evidence(open_hand, aligned)
    assert grasp_evidence(partially_closed, aligned)
    assert not grasp_evidence(partially_closed, aligned + np.array([0.3, 0, 0]))
    assert not grasp_evidence(
        replace(partially_closed, robot=replace(partially_closed.robot, gripper_holding=False)),
        aligned,
    )


def test_progress_labels_penalize_repeated_approach_and_empty_lift() -> None:
    row = {"settled": True, "isolated": True, "tip_can_xy_before_m": 0.03}
    assert (
        measured_utility(
            {
                **row,
                "skill": "PREGRASP",
                "tip_can_xy_after_m": 0.03,
            }
        )
        == 0.0
    )
    assert (
        measured_utility(
            {
                **row,
                "skill": "PREGRASP",
                "tip_can_xy_after_m": 0.01,
            }
        )
        > 0
    )
    assert measured_utility({**row, "skill": "LIFT", "can_rise_m": 0.0}) == 0.0


def test_placement_requires_grasp_and_measured_elevation() -> None:
    from unittest.mock import Mock

    from vlarobot.core.types import ObjectState, SceneState, Vec3, WorldState
    from vlarobot.execution.skill_live import generate_candidates
    from vlarobot.execution.skills import Skill, SkillExecutor

    backend = FakeBackend()
    obs = backend.reset(ScenarioSpec(), 0)
    executor = Mock(spec=SkillExecutor)
    tool = obs.robot.ee_pose.position.as_array()
    executor._base_frame_targets.return_value = {
        "pregrasp": tool + [0, 0, 0.12],
        "grasp": tool + [0, 0, -0.05],
        "lift": tool + [0, 0, 0.12],
        "transit": tool + [0.2, 0, 0.12],
        "place_descend": tool + [0.05, 0, -0.05],
        "retreat": tool + [0, 0, 0.12],
    }
    scene = SceneState([ObjectState("CAN", "can", obs.robot.ee_pose, Vec3(0.06, 0.06, 0.1), 1)])
    for holding, rise, expected in (
        (False, 0.1, {Skill.APPROACH, Skill.RETREAT}),
        (True, 0.0, {Skill.LIFT}),
        (True, 0.1, {Skill.TRANSFER, Skill.LOWER}),
    ):
        current = replace(obs, robot=replace(obs.robot, gripper_holding=holding))
        candidates = generate_candidates(
            executor, backend, current, WorldState(scene, current.robot), can_rise_m=rise
        )
        retreat = next(action for action in candidates if action.skill is Skill.RETREAT)
        assert retreat.target_base is not None
        np.testing.assert_allclose(retreat.target_base.as_array()[:2], tool[:2])
        observed = {action.skill for action in candidates}
        assert expected <= observed
        assert (Skill.LOWER in observed) == (holding and rise > 0)
        if rise > 0:
            assert observed.isdisjoint({Skill.OPEN, Skill.RELEASE})

        closed = replace(current, robot=replace(current.robot, gripper_opening=0.0))
        closed_actions = generate_candidates(
            executor, backend, closed, WorldState(scene, closed.robot), can_rise_m=rise
        )
        assert Skill.DESCEND not in {action.skill for action in closed_actions}

    open_obs = replace(obs, robot=replace(obs.robot, gripper_opening=GRIPPER_OPEN))
    open_state = WorldState(scene, open_obs.robot)
    assert Skill.DESCEND in {
        action.skill
        for action in generate_candidates(executor, backend, open_obs, open_state, can_rise_m=0.0)
    }
    executor._base_frame_targets.return_value["grasp"] = tool + [0.2, 0, -0.05]
    assert Skill.DESCEND not in {
        action.skill
        for action in generate_candidates(executor, backend, open_obs, open_state, can_rise_m=0.0)
    }


def test_descent_measures_progress_and_object_disturbance() -> None:
    def xyz(z: float) -> dict[str, float]:
        return {"x": 0.0, "y": 0.0, "z": z}

    row = {
        "skill": "DESCEND",
        "settled": True,
        "isolated": True,
        "tip_can_xy_before_m": 0.0,
        "target_base_m": xyz(0.8),
        "tool_before_base_m": xyz(0.9),
        "tool_after_base_m": xyz(0.8),
        "can_before_world_m": xyz(0.8),
        "can_after_world_m": xyz(0.8),
        "can_orientation_after_xyzw": (0.0, 0.0, 0.0, 1.0),
    }
    assert measured_utility(row) == 1.0
    assert measured_utility({**row, "can_after_world_m": xyz(0.65)}) == 0.0
    assert measured_utility({**row, "tool_after_base_m": xyz(0.9)}) == 0.0
    assert measured_utility({**row, "can_orientation_after_xyzw": (1, 0, 0, 0)}) == 0.0


def test_approach_history_prefers_a_real_improvement_over_a_repeated_hover() -> None:
    from vlarobot.core.types import Vec3
    from vlarobot.execution.skill_live import approach_progress_score
    from vlarobot.execution.skills import Skill, SkillAction

    can = Vec3(0.7, -0.2, 0.75)
    centered = SkillAction(Skill.APPROACH, Vec3(0.7, -0.2, 0.9))
    offset = SkillAction(Skill.APPROACH, Vec3(0.7, -0.17, 0.9))
    assert approach_progress_score(0.5, centered, can, None) == 0.5
    assert approach_progress_score(0.5, centered, can, 0.03) == 0.5
    assert approach_progress_score(0.5, centered, can, 0.002) < 0
    assert approach_progress_score(0.5, offset, can, 0.002) < 0
    assert approach_progress_score(0.4, SkillAction(Skill.OPEN), can, 0.002) == 0.4


def test_transport_uses_live_attachment_evidence_after_observed_lift() -> None:
    from vlarobot.core.types import Vec3
    from vlarobot.execution.skill_live import measured_attachment

    obs = FakeBackend().reset(ScenarioSpec(), 0)
    base = obs.robot.base_pose_world
    assert base is not None
    old_pose = obs.robot.ee_pose
    new_pose = replace(old_pose, position=replace(old_pose.position, z=old_pose.position.z + 0.11))
    before = replace(obs, robot=replace(obs.robot, gripper_holding=None, gripper_opening=0.03))
    after = replace(before, robot=replace(before.robot, ee_pose=new_pose))
    from vlarobot.world_model.rollout import compose

    old = compose(base, old_pose).position
    new = compose(base, new_pose).position
    can_before = Vec3(old.x + 0.025, old.y, old.z)
    can_after = Vec3(new.x + 0.025, new.y, new.z)
    measured = measured_attachment(before, after, can_before, can_after)
    assert measured is not None
    assert measured_attachment(before, after, can_before, can_before) is None
    can_base = replace(new_pose, position=replace(new_pose.position, x=new_pose.position.x + 0.025))
    far = new_pose.position.as_array() + np.array([0.025, 0, 0])
    assert grasp_evidence(after, far, measured_can_base=can_base, attachment_offset=measured)
    moved = replace(can_base, position=replace(can_base.position, x=can_base.position.x + 0.02))
    assert not grasp_evidence(after, far, measured_can_base=moved, attachment_offset=measured)


def test_release_away_from_destination_remains_a_recovery_option() -> None:
    from vlarobot.backends.webots.config import PLACE_WORLD
    from vlarobot.core.types import Vec3
    from vlarobot.execution.skill_live import release_progress_score
    from vlarobot.execution.skills import Skill, SkillAction

    release = SkillAction(Skill.RELEASE)
    target = Vec3(*PLACE_WORLD)
    assert release_progress_score(0.8, release, target) == 0.8
    assert release_progress_score(0.8, release, replace(target, x=target.x + 1.0)) == 0.0


def test_lift_forecast_uses_multiple_cartesian_waypoints() -> None:
    from unittest.mock import patch

    from vlarobot.core.types import ObjectState, SceneState, Vec3, WorldState
    from vlarobot.execution.skill_empirical import forecast_candidate
    from vlarobot.execution.skills import Skill, SkillAction
    from vlarobot.world_model.rollout import Trajectory

    observation = FakeBackend().reset(ScenarioSpec(), 0)
    pose = observation.robot.ee_pose
    state = WorldState(
        SceneState([ObjectState("CAN", "can", pose, Vec3(0.06, 0.06, 0.1), 1.0)]),
        observation.robot,
    )
    target = replace(pose.position, z=pose.position.z + 0.12)
    with patch("vlarobot.execution.skill_empirical.AnalyticSurrogate") as surrogate:
        predictor = surrogate.return_value
        predictor.predict_trajectory.return_value = Trajectory((state,), 1.0, (), (), ())
        forecast_candidate(state, SkillAction(Skill.LIFT, target))
    _, chunk = predictor.predict_trajectory.call_args.args
    assert len(chunk.actions) >= 3
    assert chunk.actions[-1].pose.position == target
    assert pose.position.z < chunk.actions[0].pose.position.z < target.z


def test_close_progress_does_not_reward_recommanding_a_settled_grasp() -> None:
    from vlarobot.execution.skill_live import close_progress_score
    from vlarobot.execution.skills import Skill, SkillAction

    close = SkillAction(Skill.CLOSE)
    assert close_progress_score(0.8, close, GRIPPER_OPEN, False) == 0.8
    assert close_progress_score(0.8, close, 0.0308, True) == 0.0
    assert close_progress_score(0.8, close, 0.001, False) == 0.0


def test_existing_attachment_can_refresh_from_horizontal_co_motion() -> None:
    from vlarobot.core.types import Vec3
    from vlarobot.execution.skill_live import measured_attachment
    from vlarobot.world_model.rollout import compose

    obs = FakeBackend().reset(ScenarioSpec(), 0)
    base = obs.robot.base_pose_world
    assert base is not None
    old = obs.robot.ee_pose
    moved = replace(old, position=replace(old.position, x=old.position.x + 0.23))
    before = replace(obs, robot=replace(obs.robot, gripper_opening=0.0305))
    after = replace(before, robot=replace(before.robot, ee_pose=moved))
    start = compose(base, old).position
    end = compose(base, moved).position
    can_start = Vec3(start.x - 0.015, start.y, start.z)
    can_end = Vec3(end.x + 0.025, end.y, end.z)
    previous = np.array([-0.015, 0, 0])
    assert measured_attachment(before, after, can_start, can_end) is None
    assert (
        measured_attachment(before, after, can_start, can_end, previous_offset=previous) is not None
    )
    assert (
        measured_attachment(before, after, can_start, can_start, previous_offset=previous) is None
    )


def test_admission_requires_verified_home_and_preserves_all_24_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlarobot.execution import skill_live

    model = tmp_path / "model.json"
    model.write_bytes(b"frozen model")
    prerequisite = tmp_path / "clear.json"
    out = tmp_path / "admission"
    clear = skill_live.build_episode_plan("train", 0, n=1)[0]
    criteria = {
        "success": True,
        "home_reached": False,
        "place_distance_ok": True,
        "can_height_ok": True,
        "can_tilt_ok": True,
    }
    payload = {
        "split": "train",
        "mode": "oracle-assisted",
        "scene": clear.scene.name,
        "seed": clear.seed,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "profile": skill_live.PROFILE_ID,
        "urdf_registration_sha256": skill_live.CORRECTED_REGISTRATION_SHA256,
        "waypoint_forecast_policy": skill_live.WAYPOINT_FORECAST_POLICY,
        "clearance_policy": skill_live.CLEARANCE_POLICY,
        "urdf_registration": skill_live.CORRECTED_REGISTRATION_RELATIVE.as_posix(),
        "home_tolerance_rad": skill_live.HOME_VERIFICATION_TOLERANCE_RAD,
        "utility_context": "predicted",
        "stop_reason": "termination selected",
        "decision_budget": 16,
        "verification": criteria,
    }
    prerequisite.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independently verified"):
        skill_live.run_train_admission(out, model, None, verified_clear=prerequisite)
    assert not out.exists()

    criteria["home_reached"] = True
    prerequisite.write_text(json.dumps(payload), encoding="utf-8")

    def run(folder, frozen, webots, episode, *, context, max_decisions):
        assert frozen.read_bytes() == b"frozen model"
        assert context == "predicted"
        assert max_decisions == 16
        return {"verification": {"success": episode.index < 10}}

    monkeypatch.setattr(skill_live, "_run_train_episode", run)
    report = skill_live.run_train_admission(out, model, None, verified_clear=prerequisite)
    assert report["measured"] == 24
    assert report["successes"] == 10
    assert report["passed"] is True
    assert [row["index"] for row in report["episodes"]] == list(range(24))


@pytest.mark.parametrize(
    ("nominal_reasons", "path_reasons", "passed", "replace_forecast"),
    [
        (("ik-joint-safety-veto",), ("unverified-arm-sweep",), True, True),
        (("ik-joint-safety-veto",), ("unverified-arm-sweep",), False, False),
        (("ik-joint-safety-veto",), ("safety-monitor-veto",), True, False),
        (("unverified-arm-sweep",), ("unverified-arm-sweep",), True, False),
    ],
)
def test_waypoint_forecast_requires_a_complete_passing_preflight(
    nominal_reasons, path_reasons, passed, replace_forecast
) -> None:
    from vlarobot.core.types import SceneState, WorldState
    from vlarobot.execution.skill_live import _waypoint_forecast
    from vlarobot.world_model.rollout import Trajectory

    state = WorldState(SceneState([]), FakeBackend().reset(ScenarioSpec(), 0).robot)
    nominal = Trajectory((state,), 1.0, nominal_reasons, (), ())
    audited = Trajectory((state, state), 0.032, path_reasons, (), ())
    chosen, recorded_veto = _waypoint_forecast(nominal, [audited], preflight_passed=passed)
    assert chosen == (audited if replace_forecast else nominal)
    assert recorded_veto == (nominal_reasons if replace_forecast else None)


def test_clearance_recovery_is_bounded_and_requires_transit_height() -> None:
    from vlarobot.execution.skill_proxy import clearance_path_passes

    origin = np.array([0.7, 0.0, 0.947])
    floor = 0.949
    end = origin + [0.0, 0.001, 0.008]
    assert clearance_path_passes(np.array([origin, end]), origin, floor, True)
    for bad in (origin + [0.0, 0.0, 0.001], end + [0.01, 0, 0], end + [0, 0, 0.02]):
        assert not clearance_path_passes(np.array([origin, bad]), origin, floor, True)
    dipped = origin + [0.0, 0.0, -0.002]
    assert not clearance_path_passes(np.array([origin, dipped, end]), origin, floor, True)


def test_v4_rejects_missing_or_different_simulator_registration() -> None:
    from vlarobot.execution.skills import V4_REGISTRATION_RECORD, require_v4_registration

    initial = FakeBackend().reset(ScenarioSpec(), 0)
    for diagnostics in ({}, {"urdf_registration_sha256": "unregistered"}):
        observation = replace(initial, robot=replace(initial.robot, diagnostics=diagnostics))
        with pytest.raises(ValueError, match="required V4 URDF registration"):
            require_v4_registration(observation)
    require_v4_registration(
        replace(initial, robot=replace(initial.robot, diagnostics=V4_REGISTRATION_RECORD))
    )
