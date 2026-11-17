"""VLA action semantics: physical units, base-frame rotations, vetoes and lockstep chunks."""

import numpy as np
import pytest

from vlarobot.core.types import (
    ApplyResult,
    CameraInfo,
    Observation,
    Pose2D,
    Pose3D,
    RobotState,
    Vec3,
)
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.vla.action_adapter import ActionAdapter, PoseSolution, pose_delta
from vlarobot.vla.schema import GripperMapping

CAMERA = CameraInfo(
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    1,
    1,
)


@pytest.fixture
def state():
    return RobotState(
        {"arm_1_joint": 0.0}, {}, Pose2D(0, 0, 0), Pose3D(Vec3(0.5, 0, 0.5), (0, 0, 0, 1)), 0.03
    )


class Solver:
    def __init__(self):
        self.solution = PoseSolution({"arm_1_joint": 0.01}, 0.001, 0.001)
        self.calls = 0

    def solve(self, pose, state):
        self.calls += 1
        return self.solution


class Backend:
    def __init__(self, state):
        self.state, self.commands, self.steps = state, [], []
        self.accept = True

    def apply(self, action):
        self.commands.append(action)
        return ApplyResult(self.accept, None if self.accept else "backend veto")

    def step(self, n_steps=1):
        self.steps.append(n_steps)
        return Observation(0, np.zeros((1, 1, 3), dtype=np.uint8), None, CAMERA, self.state)


@pytest.fixture
def adapter():
    return ActionAdapter(
        Solver(),
        SafetyMonitor(
            SafetyLimits(joint_position={"arm_1_joint": (-1, 1)}, joint_velocity={"arm_1_joint": 1})
        ),
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.01,
    )


def test_gripper_commands_are_per_finger_metres_and_feedback_is_not_clipped():
    mapping = GripperMapping(0, 0.045)
    assert mapping.target_position(0.0225) == 0.0225
    assert mapping.measured_half_aperture((0.061, 0.001)) == pytest.approx(0.031)
    assert mapping.target_position(0) == 0  # close command differs from held-object feedback
    with pytest.raises(ValueError):
        mapping.target_position(0.09)  # full gap is not a valid per-finger command


def test_pose_delta_round_trip_preserves_base_rotation_convention(adapter, state):
    values = np.array([0.01, -0.02, 0.03, 0.01, 0.02, -0.03, 0.0225])
    proposal = adapter.decode(values, state)
    np.testing.assert_allclose(
        pose_delta(state.ee_pose, proposal.target_pose), values[:6], atol=1e-12
    )
    assert proposal.gripper.opening == 0.0225


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(6),
        np.full(7, np.nan),
        np.zeros((1, 7)),
        np.array([0, 0, 0, 0, 0, 0, 0.09]),
        np.zeros(7, dtype=bool),
    ],
)
def test_invalid_model_outputs_never_reach_solver_or_backend(adapter, state, bad):
    backend = Backend(state)
    result = adapter.apply(backend, bad, state)
    assert not result.accepted
    assert not backend.commands
    assert adapter.solver.calls == 0


def test_solver_pose_residual_and_joint_safety_are_independent_vetoes(adapter, state):
    backend = Backend(state)
    adapter.solver.solution = PoseSolution({"arm_1_joint": 0.01}, 0.02, 0.001)
    assert not adapter.apply(backend, np.zeros(7), state).accepted
    adapter.solver.solution = PoseSolution({"arm_1_joint": 2}, 0.001, 0.001)
    assert not adapter.apply(backend, np.zeros(7), state).accepted
    assert not backend.commands


def test_backend_veto_stops_command_sequence_and_later_chunk_actions(adapter, state):
    backend = Backend(state)
    backend.accept = False
    observation = Observation(0, np.zeros((1, 1, 3), dtype=np.uint8), None, CAMERA, state)
    result, count = adapter.execute_chunk(
        backend, np.zeros((2, 7)), observation, steps_per_action=3, max_actions=50
    )
    assert not result.accepted and count == 0
    assert len(backend.commands) == 1
    assert not backend.steps


def test_chunk_cadence_is_explicit_and_late_nan_rejects_entire_chunk(adapter, state):
    backend = Backend(state)
    observation = Observation(0, np.zeros((1, 1, 3), dtype=np.uint8), None, CAMERA, state)
    bad = np.zeros((2, 7))
    bad[1, 0] = np.nan
    assert not adapter.execute_chunk(backend, bad, observation, steps_per_action=4, max_actions=50)[
        0
    ].accepted
    assert not backend.commands
    result, count = adapter.execute_chunk(
        backend, np.zeros((2, 7)), observation, steps_per_action=4, max_actions=50
    )
    assert result.accepted and count == 2
    assert backend.steps == [4, 4]


def test_float32_command_endpoint_is_canonicalized_but_next_outside_value_is_vetoed(adapter, state):
    values = np.zeros(7, dtype=np.float32)
    values[6] = 0.045
    assert adapter.decode(values, state).gripper.opening == 0.045
    values[6] = np.nextafter(np.float32(0.045), np.float32(np.inf))
    with pytest.raises(ValueError, match="outside"):
        adapter.decode(values, state)
