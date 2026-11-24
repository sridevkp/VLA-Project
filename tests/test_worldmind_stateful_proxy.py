import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import (
    ActionChunk,
    GripperCmd,
    JointTarget,
    ObjectState,
    Pose3D,
    ScenarioSpec,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.execution.worldmind_stateful_proxy import Waypoint, rank_phase_pool
from vlarobot.world_model.rollout import Trajectory

Q = (0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def fixture():
    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/worldmind_arm5_matched_clearance.json"
        ).read_text()
    )
    robot = FakeBackend().reset(ScenarioSpec(), 0).robot
    robot = replace(
        robot,
        base_pose_world=Pose3D(Vec3(0, 0, 0), Q),
        ee_pose=Pose3D(Vec3(0, 0, 1), Q),
        joint_pos={"joint": 0.0},
    )
    boxes = SceneState(
        [
            ObjectState(name, name, Pose3D(Vec3(i, 1, 1), Q), Vec3(0.1, 0.1, 0.1), 1)
            for i, name in enumerate(("CAN", "HAM", "APP"), 1)
        ]
    )
    return (
        config,
        WorldState(boxes, robot),
        SimpleNamespace(limits=SimpleNamespace(joint_position={"joint": (-1.0, 1.0)})),
    )


def rank(fixture, waypoint, action, after, goal):
    config, initial, monitor = fixture
    chunk = ActionChunk([action] * 50, 0.032, "scripted")
    return rank_phase_pool(
        initial,
        [chunk],
        waypoint,
        lambda state, _chunk: Trajectory((state, *([after] * 50)), 0.032, (), (), ()),
        monitor,
        config,
        goal,
        0.1,
        lambda _cmd: True,
    )


def test_init_admits_without_fabricated_placement(fixture):
    _, initial, _ = fixture
    decision, (row,) = rank(
        fixture, Waypoint("INIT", False, None, None), GripperCmd(0.045), initial, (3, 0, 1)
    )
    assert decision.selected_index == 0
    assert row["placement"]["status"] == "not_applicable"
    assert row["placement"]["accepted"] is None and row["components"][0] is None
    assert row["active_prefix_ticks"] == 50


def test_required_missing_and_out_of_tolerance_placement_veto(fixture):
    _, initial, _ = fixture
    action = JointTarget({"joint": 0.1})
    point = (0.0, 0.0, 0.85089)
    waypoint = Waypoint("PLACE_DESCEND", True, 0.01, point)
    target = replace(initial, robot=replace(initial.robot, ee_pose=Pose3D(Vec3(*point), Q)))
    decision, (row,) = rank(fixture, waypoint, action, target, point)
    assert decision.selected_index is None and row["placement"]["status"] == "unavailable"
    settled = replace(target, robot=replace(target.robot, joint_pos=action.q))
    decision, (row,) = rank(fixture, waypoint, action, settled, (0.2, 0.0, point[2]))
    assert decision.selected_index is None and row["placement"]["accepted"] is False
    decision, (row,) = rank(fixture, waypoint, action, settled, point)
    assert decision.selected_index == 0
    assert row["active_prefix_ticks"] == row["placement"]["state_index"] == 1


def test_clearance_applies_to_early_chunk_with_no_placement(fixture):
    _, initial, _ = fixture
    below = replace(initial, robot=replace(initial.robot, ee_pose=Pose3D(Vec3(0, 0, 0.74), Q)))
    decision, (row,) = rank(
        fixture, Waypoint("INIT", False, None, None), GripperCmd(0.045), below, (0, 0, 1)
    )
    assert decision.selected_index is None and row["checks"]["table_clearance"] is False
    through_ham = replace(initial, robot=replace(initial.robot, ee_pose=Pose3D(Vec3(2, 1, 1), Q)))
    decision, (row,) = rank(
        fixture, Waypoint("INIT", False, None, None), GripperCmd(0.045), through_ham, (0, 0, 1)
    )
    assert decision.selected_index is None and row["checks"]["object_clearance"] is False


def test_contact_endpoint_is_not_inferred_from_an_intermediate_waypoint(fixture):
    _, initial, _ = fixture
    waypoint = Waypoint("DESCEND", False, 0.01, (0.0, 0.0, 0.95))
    target = replace(
        initial,
        robot=replace(initial.robot, ee_pose=Pose3D(Vec3(0, 0, 0.95), Q), joint_pos={"joint": 0.1}),
    )
    decision, (row,) = rank(fixture, waypoint, JointTarget({"joint": 0.1}), target, (0, 0, 1))
    assert decision.selected_index == 0 and row["active_prefix_ticks"] == 1
    assert row["checks"]["table_clearance"]


def test_goal_heavy_changes_only_applicable_goal_ranking():
    from vlarobot.execution.worldmind_stateful_proxy import select_phase_assessments
    from vlarobot.risk.gate import CandidateAssessment

    names = (
        "joint_limit_margin",
        "ik_feasibility_and_residual",
        "table_clearance",
        "object_clearance",
    )
    checks = tuple((name, True) for name in names)
    measured = [
        CandidateAssessment((*checks, ("placement_tolerance", True)), (0.02, 0.00, 0.0, 0.0)),
        CandidateAssessment((*checks, ("placement_tolerance", True)), (0.00, 0.03, 0.0, 0.0)),
    ]
    assert select_phase_assessments("equal", True, measured).selected_index == 0
    assert select_phase_assessments("goal_x2", True, measured).selected_index == 1
    early = [CandidateAssessment(checks, (None, row.components[1], 0.0, 0.0)) for row in measured]
    assert select_phase_assessments("equal", False, early).selected_index == 0
    assert select_phase_assessments("goal_x2", False, early).selected_index == 0


def test_arm4_calls_prediction_and_ranks_without_riskgate_or_placement_veto(fixture, monkeypatch):
    from vlarobot.risk.gate import RiskGate

    def unexpected(*_args):
        pytest.fail("no-veto arm reached a predictive hard gate")

    monkeypatch.setattr(RiskGate, "select", unexpected)
    geometry, initial, monitor = fixture
    endpoint = (0.2, 0.0, 0.74)
    waypoint = Waypoint("PLACE_DESCEND", True, 0.01, endpoint)
    commands = [JointTarget({"joint": 0.1}), JointTarget({"joint": 0.2})]
    chunks = [ActionChunk([command] * 50, 0.032, "scripted") for command in commands]
    calls = []

    def predict(state, chunk):
        calls.append(chunk.actions[0])
        position = endpoint if len(calls) == 1 else (0.4, 0.0, 0.74)
        end = replace(
            state,
            robot=replace(
                state.robot, joint_pos=chunk.actions[0].q, ee_pose=Pose3D(Vec3(*position), Q)
            ),
        )
        return Trajectory((state, *([end] * 50)), 0.032, (), (), ())

    decision, records = rank_phase_pool(
        initial,
        chunks,
        waypoint,
        predict,
        monitor,
        geometry,
        (0.0, 0.0, 0.74),
        0.1,
        unexpected,
        veto=False,
    )
    assert calls == commands and decision.selected_index == 0
    assert all(r["admitted"] and r["placement"]["accepted"] is None for r in records)
    assert all("placement_tolerance" not in r["checks"] for r in records)
    assert records[0]["components"][0] == pytest.approx(0.2)
