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
        "grasp": tool,
        "lift": tool + [0, 0, 0.12],
        "transit": tool + [0.2, 0, 0.12],
        "place_descend": tool + [0.2, 0, 0],
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
