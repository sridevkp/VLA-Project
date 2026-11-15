"""Optimization retains quaternion boundaries, snapshots, FK contract and five-arm behavior."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vlarobot.core.types import (
    ActionChunk,
    GripperCmd,
    JointTarget,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.risk.comparison import ComparisonPlanner
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.prepared_fk import PreparedForward
from vlarobot.world_model.rollout import Trajectory, quaternion_product
from vlarobot.world_model.state_copy import copy_world


def test_frozen_values_share_only_immutable_state_and_diagnostics_are_copied():
    robot = RobotState(
        {"j": 0.0},
        {"j": 0.1},
        Pose2D(0, 0, 0),
        Pose3D(Vec3(0, 0, 0), (0, 0, 0, 1)),
        0.02,
        diagnostics={"nested": [{"value": 1}]},
    )
    state = WorldState(SceneState(), robot)
    snapshot = copy_world(state)
    snapshot.robot.joint_pos["j"] = 1.0
    snapshot.robot.diagnostics["nested"][0]["value"] = 2
    assert state.robot.joint_pos["j"] == 0 and state.robot.diagnostics["nested"][0]["value"] == 1
    assert snapshot.robot.ee_pose is state.robot.ee_pose


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_quaternion_optimization_matches_reference_and_norm_acceptance(dtype):
    rng = np.random.default_rng(190)
    for _ in range(100):
        a = rng.normal(size=4).astype(dtype)
        a /= np.linalg.norm(a)
        b = rng.normal(size=4).astype(dtype)
        b /= np.linalg.norm(b)
        expected = np.r_[
            a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]
        ]
        np.testing.assert_array_equal(quaternion_product(a, b), expected)
    for squared_norm in (
        1 - 1.001e-5 - 1e-12,
        1 - 1.001e-5 + 1e-12,
        1 + 1.001e-5 - 1e-12,
        1 + 1.001e-5 + 1e-12,
    ):
        a = np.array([np.sqrt(squared_norm), 0, 0, 0], dtype=dtype)
        if np.isclose(a @ a, 1):
            quaternion_product(a, np.array([0, 0, 0, 1], dtype=dtype))
        else:
            with pytest.raises(ValueError):
                quaternion_product(a, np.array([0, 0, 0, 1], dtype=dtype))


def test_prepared_fk_rejects_fixed_joint_changes_and_retains_exact_dot_order():
    def translation(x):
        m = np.eye(4)
        m[0, 3] = x
        return m

    links = [("fixed", translation), ("joint", translation), ("tool", lambda q: translation(0.1))]

    def pose(m):
        return Pose3D(Vec3(*map(float, m[:3, 3])), (0, 0, 0, 1))

    fk = PreparedForward(
        links,
        active_joints=frozenset(("joint",)),
        reference_joints={"fixed": 0.2, "joint": 0.0},
        to_pose=pose,
        geometry_id="synthetic-test",
    )
    expected = np.eye(4)
    for (_, f), q in zip(links, (0.2, 0.3, 0), strict=True):
        expected = np.dot(expected, f(q))
    assert fk({"fixed": 0.2, "joint": 0.3}) == pose(expected)
    with pytest.raises(ValueError, match="fixed joint changed"):
        fk({"fixed": 0.21, "joint": 0.3})


def test_all_five_arm_behaviors_are_distinct_with_same_live_safety_contract():
    robot = RobotState({"x": 0.0}, {}, Pose2D(0, 0, 0), Pose3D(Vec3(0, 0, 0.5), (0, 0, 0, 1)), 0.02)
    state = WorldState(SceneState(), robot)
    chunks = [ActionChunk([JointTarget({"x": float(i + 1)})], 0.032, "vla") for i in range(4)]

    def index(chunk):
        return int(chunk.actions[0].q["x"]) - 1

    def predict(state, chunk):
        i = index(chunk)
        return Trajectory(
            (copy.deepcopy(state), copy.deepcopy(state)),
            0.032,
            ("predicted-collision",) if i < 3 else (),
            (),
            (),
        )

    def assess(state, chunk, trajectory):
        i = index(chunk)
        score = (4 - i) if trajectory is None else (1, 3, 4, 2)[i]
        return CandidateAssessment((("heuristic_safe", i != 0),), (float(score), 0.0, 0.0, 0.0))

    gate = RiskGate(weights=(1, 0, 0, 0), maximize=True, required_constraints=("heuristic_safe",))
    observed = []
    for arm in ("vla_only", "vla_risk_veto", "vla_world_model", "full_system"):
        selection = ComparisonPlanner(arm, gate, predict=predict, assess=assess).select(
            state, chunks
        )
        observed.append(selection.index)
    assert observed == [0, 1, 2, 3]
    baseline = ComparisonPlanner("scripted_baseline", gate, predict=predict, assess=assess)
    script = ActionChunk([GripperCmd(0.03)], 0.032, "scripted")
    assert baseline.select(state, [script]).chunk is script
