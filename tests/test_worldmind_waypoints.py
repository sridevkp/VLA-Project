"""Inert tests for the reviewed scripted waypoint cadence and adapter station types."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlarobot.backends.webots.config import (
    APPROACH_H,
    GRASP_Z,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    MAX_JOINT_STEP_RAD,
    PLACE_CLEAR,
    SETTINGS,
)
from vlarobot.backends.worldmind_bindings import corrected_scripted_waypoints
from vlarobot.core.types import (
    ApplyResult,
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    TaskSpec,
    Vec3,
    WorldState,
)
from vlarobot.execution.scripted_chunk_adapter import ScriptedChunkAdapter, ScriptedWaypoint

IDENTITY = (0.0, 0.0, 0.0, 1.0)
EXPECTED_LABELS = (
    "INIT",
    "PREGRASP",
    "DESCEND",
    "CLOSE",
    "LIFT",
    "TRANSIT",
    "PLACE_DESCEND",
    "RELEASE",
    "RETREAT",
    "HOME",
)


def _state() -> WorldState:
    robot = RobotState(
        joint_pos={"arm_1_joint": 0.0},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
        gripper_opening=0.045,
        base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), IDENTITY),
    )
    scene = SceneState(
        [
            ObjectState(
                "CAN",
                "can",
                Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
                Vec3(0.08, 0.08, 0.12),
                1.0,
            )
        ]
    )
    return WorldState(scene, robot)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(target="CAN"),
        pipeline=SimpleNamespace(horizon_steps=50),
        adapter=SimpleNamespace(ik_orientation_tolerance_rad=0.01),
        verification=SimpleNamespace(
            destination_world=(-1.0, -4.3, 0.821),
            home_joints={"arm_1_joint": 0.0},
        ),
    )


class _StubSolver:
    def __init__(self, *, residual_m: float = 0.0) -> None:
        self.residual_m = residual_m
        self.calls = 0

    def solve(self, pose: Pose3D, state: WorldState, *, seed: int = 0) -> SimpleNamespace:
        del state, seed, pose
        self.calls += 1
        return SimpleNamespace(
            joints={"arm_1_joint": 0.0},
            position_error_m=self.residual_m,
            orientation_error_rad=0.0,
        )


class _AcceptAllMonitor:
    def validate(self, action: object, robot: object) -> ApplyResult:
        del action, robot
        return ApplyResult(True, "synthetic accept")


def _adapter(waypoints: tuple[ScriptedWaypoint, ...], solver: _StubSolver) -> ScriptedChunkAdapter:
    return ScriptedChunkAdapter(
        waypoints,
        solver=solver,
        monitor=_AcceptAllMonitor(),  # type: ignore[arg-type]
        horizon_steps=sum(item.hold_steps for item in waypoints),
        control_dt_s=0.032,
        jitter=SimpleNamespace(std_rad=0.0, limit_rad=0.0, max_attempts=1),  # type: ignore[arg-type]
        max_joint_step_rad=MAX_JOINT_STEP_RAD,
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.01,
    )


def test_corrected_waypoints_mirror_frozen_baseline_station_order_and_gripper_timing():
    waypoints = corrected_scripted_waypoints(_config(), _state(), TaskSpec("t", "CAN", "dest"))
    assert tuple(item.label for item in waypoints) == EXPECTED_LABELS
    assert sum(item.hold_steps for item in waypoints) == 50
    assert waypoints[0].gripper_opening == pytest.approx(SETTINGS.gripper_opening_limits[1])
    assert waypoints[3].gripper_opening == pytest.approx(SETTINGS.gripper_opening_limits[0])
    assert waypoints[7].gripper_opening == pytest.approx(SETTINGS.gripper_opening_limits[1])
    assert waypoints[9].joints == {"arm_1_joint": 0.0}
    non_pose = (waypoints[0], waypoints[3], waypoints[7], waypoints[9])
    assert all(item.pose is None for item in non_pose)


def test_corrected_waypoints_use_reviewed_state_specific_tolerances():
    waypoints = corrected_scripted_waypoints(_config(), _state(), TaskSpec("t", "CAN", "dest"))
    tolerances = {item.label: item.position_tolerance_m for item in waypoints}
    assert tolerances["PREGRASP"] == pytest.approx(IK_TOL)
    assert tolerances["DESCEND"] == pytest.approx(IK_TOL_GRASP)
    assert tolerances["LIFT"] == pytest.approx(IK_TOL_TRANSIT)
    assert tolerances["TRANSIT"] == pytest.approx(IK_TOL_TRANSIT)
    assert tolerances["PLACE_DESCEND"] == pytest.approx(IK_TOL_GRASP)
    assert tolerances["RETREAT"] == pytest.approx(IK_TOL_TRANSIT)
    descend = waypoints[2].pose
    assert descend is not None
    assert descend.position.z == pytest.approx(0.8 + GRASP_Z)
    pregrasp = waypoints[1].pose
    assert pregrasp is not None
    assert pregrasp.position.z == pytest.approx(0.8 + GRASP_Z + APPROACH_H)
    place = waypoints[6].pose
    assert place is not None
    assert place.position.z == pytest.approx(0.821 + GRASP_Z + PLACE_CLEAR)


def test_adapter_supports_reviewed_joint_and_per_station_tolerance_stations():
    solver = _StubSolver(residual_m=0.02)
    waypoints = (
        ScriptedWaypoint(
            "DESCEND",
            1,
            pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
            position_tolerance_m=0.03,
        ),
        ScriptedWaypoint("HOME", 1, joints={"arm_1_joint": -0.1}),
    )
    chunk = _adapter(waypoints, solver).draw()(TaskSpec("t", "CAN", "dest"), _state(), object(), 0)
    assert len(chunk.actions) == 2
    assert isinstance(chunk.actions[0], JointTarget)
    assert isinstance(chunk.actions[1], JointTarget)
    assert chunk.actions[1].q == {"arm_1_joint": -0.1}
    assert chunk.dt == 0.032

    strict = (
        ScriptedWaypoint(
            "DESCEND",
            1,
            pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
            position_tolerance_m=0.005,
        ),
    )
    with pytest.raises(ValueError, match="outside reviewed tolerance"):
        _adapter(strict, _StubSolver(residual_m=0.02)).draw()(
            TaskSpec("t", "CAN", "dest"), _state(), object(), 0
        )


def test_waypoint_validation_rejects_ambiguous_or_partial_stations():
    with pytest.raises(ValueError, match="exactly one"):
        ScriptedWaypoint("bad", 1)
    with pytest.raises(ValueError, match="exactly one"):
        ScriptedWaypoint("bad", 1, gripper_opening=0.0, joints={"arm_1_joint": 0.0})
    with pytest.raises(ValueError, match="only valid for a pose station"):
        ScriptedWaypoint("bad", 1, joints={"arm_1_joint": 0.0}, position_tolerance_m=0.01)
