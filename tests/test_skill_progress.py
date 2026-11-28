from dataclasses import replace

import numpy as np

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
