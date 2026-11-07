import json

import numpy as np
import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import (
    APPROACH_H,
    ARM_JOINTS,
    GRASP_Z,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
    PLACE_CLEAR,
    PLACE_WORLD,
    TORSO_HEIGHT,
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import IKTrustRegionError, Kinematics
from vlarobot.core.types import GroundTruth, Pose3D, Vec3
from vlarobot.execution.scripted_can import (
    ScriptedCanPolicy,
    _ik_tolerance_for_state,
    _world_to_base,
)


class FakeCanBackend(FakeBackend):
    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        self._robot_state.base_pose_world = Pose3D(
            position=Vec3(-0.53683, -3.70906, 0.0885438),
            orientation=(
                0.01121522834116144,
                -0.007522733160055142,
                -0.717650835270671,
                0.6962721492163582,
            ),
        )
        return self._observation()

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            object_poses={
                "CAN": Pose3D(
                    Vec3(-0.799879, -4.38998, 0.810949),
                    (0.0, 0.0, 0.0, 1.0),
                )
            }
        )


def test_scripted_can_policy_allows_transit_ik_with_specific_tolerance(tmp_path) -> None:
    backend = FakeCanBackend()
    trace_path = tmp_path / "trace.jsonl"
    policy = ScriptedCanPolicy(trace_path=trace_path)

    result = policy.execute(backend, seed=0)

    assert result.seed == 0
    assert result.rejections == 0
    assert result.final_state == "home"
    assert result.success is False
    assert result.failure_state is None
    expected_distance = np.linalg.norm(
        np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
        - np.asarray(PLACE_WORLD, dtype=float)
    )
    assert result.final_target_distance == pytest.approx(expected_distance, abs=1e-12)
    assert result.failure_reason is None
    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert not any(
        record.get("event") == "ik_rejection" and record.get("state") == "TRANSIT"
        for record in records
    )


def test_full_ik_vector_uses_configured_torso_height() -> None:
    policy = ScriptedCanPolicy()

    vector = policy._full_joint_vector({})

    assert vector[policy.kinematics.idx[TORSO_JOINT]] == TORSO_HEIGHT


def test_kinematics_seed_reproduces_random_restart_sequence() -> None:
    kin = Kinematics()
    kin.set_seed(17)
    first = kin.rng.uniform(-1.0, 1.0, size=8)
    kin.set_seed(17)
    second = kin.rng.uniform(-1.0, 1.0, size=8)

    np.testing.assert_array_equal(first, second)


def test_kinematics_prefers_nearest_feasible_joint_solution() -> None:
    kin = Kinematics()
    seed = np.zeros(len(kin.chain.links), dtype=float)
    near = seed.copy()
    near[kin.idx["arm_1_joint"]] = 0.1
    far = seed.copy()
    far[kin.idx["arm_1_joint"]] = 0.4

    selected, error, _ = kin._select_solution(
        [(far, 0.0001, 0.0), (near, 0.004, 0.0)],
        seed,
        position_tolerance=IK_TOL_GRASP,
    )

    np.testing.assert_array_equal(selected, near)
    assert error == pytest.approx(0.004)


def test_joint_interpolation_respects_configured_maximum_step() -> None:
    kin = Kinematics()
    start = np.zeros(len(kin.chain.links), dtype=float)
    target = start.copy()
    target[kin.idx["arm_7_joint"]] = 2.0

    waypoints = kin.interpolate_joint_path(start, target, MAX_JOINT_STEP_RAD)

    assert len(waypoints) == 14
    previous = start
    for waypoint in waypoints:
        assert max(
            abs(waypoint[kin.idx[name]] - previous[kin.idx[name]]) for name in ARM_JOINTS
        ) <= (MAX_JOINT_STEP_RAD + 1e-12)
        previous = waypoint
    np.testing.assert_array_equal(waypoints[-1], target)


def test_ik_tolerance_is_relaxed_only_for_non_grasp_states() -> None:
    assert _ik_tolerance_for_state("PREGRASP") == IK_TOL
    assert _ik_tolerance_for_state("TRANSIT") == IK_TOL_TRANSIT == 0.015
    assert _ik_tolerance_for_state("RETREAT") == IK_TOL_TRANSIT
    assert _ik_tolerance_for_state("HOME") == IK_TOL_TRANSIT
    assert _ik_tolerance_for_state("DESCEND") == IK_TOL_GRASP == 0.005
    assert _ik_tolerance_for_state("PLACE_DESCEND") == IK_TOL_GRASP
    assert _ik_tolerance_for_state("LIFT") == IK_TOL_GRASP


def test_world_to_base_matches_frames_transform() -> None:
    half_angle = np.sqrt(0.5)
    base_pose_world = Pose3D(
        position=Vec3(1.0, 2.0, 0.5),
        orientation=(0.0, 0.0, half_angle, half_angle),
    )

    point_base = _world_to_base(np.asarray((1.0, 3.0, 0.5)), base_pose_world)

    np.testing.assert_allclose(point_base, (1.0, 0.0, 0.0), atol=1e-12)


def test_approved_world_target_and_linear_path_ik_diagnostics() -> None:
    kin = Kinematics()
    base_pose_world = Pose3D(
        position=Vec3(-0.53683, -3.70906, 0.0885438),
        orientation=(
            0.01121522834116144,
            -0.007522733160055142,
            -0.717650835270671,
            0.6962721492163582,
        ),
    )
    can_world = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
    can_base = _world_to_base(can_world, base_pose_world)
    place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)

    q = np.zeros(len(kin.chain.links), dtype=float)
    q[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    pregrasp = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    grasp = can_base + np.asarray((0.0, 0.0, GRASP_Z))
    place_descend = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR))
    place_above = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR + APPROACH_H))

    for point in (pregrasp, grasp, place_descend):
        _, error, tilt = kin.solve(point, q, restarts=False)
        assert error <= IK_TOL_GRASP
        assert tilt <= 15.0

    _, transit_error, transit_tilt = kin.solve(place_above, q, restarts=False)
    assert transit_error == pytest.approx(0.010474806, abs=1e-6)
    assert transit_error > IK_TOL_GRASP
    assert transit_error <= IK_TOL
    assert transit_error <= IK_TOL_TRANSIT
    assert transit_tilt <= MAX_TILT_DEG

    q, _, _ = kin.solve(pregrasp, q, restarts=True)
    start = kin.forward(q)[:3, 3]
    count = max(2, int(np.linalg.norm(place_above - start) / 0.03))
    first_rejected_waypoint = None
    for index, alpha in enumerate(np.linspace(0.0, 1.0, count + 1)[1:], 1):
        point = (1.0 - alpha) * start + alpha * place_above
        q, error, tilt = kin.solve(point, q)
        if error > IK_TOL_GRASP or tilt > MAX_TILT_DEG:
            first_rejected_waypoint = (index, count, error, tilt)
            break
    assert first_rejected_waypoint is not None
    assert first_rejected_waypoint[:2] == (7, 7)
    assert first_rejected_waypoint[2] == pytest.approx(0.010474, abs=1e-5)


def test_trust_region_keeps_scripted_cartesian_waypoints_continuous() -> None:
    kin = Kinematics()
    kin.set_seed(0)
    base_pose_world = Pose3D(
        position=Vec3(-0.53683, -3.70906, 0.0885438),
        orientation=(
            0.01121522834116144,
            -0.007522733160055142,
            -0.717650835270671,
            0.6962721492163582,
        ),
    )
    can_base = _world_to_base(
        np.asarray((-0.799879, -4.38998, 0.810949), dtype=float),
        base_pose_world,
    )
    place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)
    pregrasp = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    grasp = can_base + np.asarray((0.0, 0.0, GRASP_Z))
    place_tip = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR))
    place_above = place_tip + np.asarray((0.0, 0.0, APPROACH_H))

    q = np.zeros(len(kin.chain.links), dtype=float)
    q[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    q, _, _ = kin.solve(pregrasp, q, restarts=True, position_tolerance=IK_TOL)

    for target, tolerance in (
        (grasp, IK_TOL_GRASP),
        (pregrasp, IK_TOL_GRASP),
        (place_above, IK_TOL_TRANSIT),
        (place_tip, IK_TOL_GRASP),
        (place_above, IK_TOL_TRANSIT),
    ):
        start = kin.forward(q)[:3, 3].copy()
        count = max(2, int(np.linalg.norm(target - start) / 0.03))
        for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
            point = (1.0 - alpha) * start + alpha * target
            previous = q.copy()
            q, error, tilt = kin.solve(
                point,
                q,
                restarts=True,
                position_tolerance=tolerance,
                max_joint_step_rad=MAX_JOINT_STEP_RAD,
            )
            step = max(abs(q[kin.idx[name]] - previous[kin.idx[name]]) for name in ARM_JOINTS)
            assert step <= MAX_JOINT_STEP_RAD + 1e-9
            assert error <= tolerance
            assert tilt <= MAX_TILT_DEG


def test_trust_region_rejects_reachable_target_that_requires_branch_jump() -> None:
    kin = Kinematics()
    kin.set_seed(0)
    base_pose_world = Pose3D(
        position=Vec3(-0.53683, -3.70906, 0.0885438),
        orientation=(
            0.01121522834116144,
            -0.007522733160055142,
            -0.717650835270671,
            0.6962721492163582,
        ),
    )
    can_base = _world_to_base(
        np.asarray((-0.799879, -4.38998, 0.810949), dtype=float),
        base_pose_world,
    )
    target = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    seed = np.zeros(len(kin.chain.links), dtype=float)
    seed[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    goal, _, _ = kin.solve(target, seed, restarts=True, position_tolerance=IK_TOL_GRASP)
    branch_seed = goal.copy()
    branch_seed[kin.idx["arm_1_joint"]] -= 0.5

    unbounded, error, _ = kin.solve(
        target,
        branch_seed,
        restarts=True,
        position_tolerance=IK_TOL_GRASP,
    )
    assert error <= IK_TOL_GRASP
    assert (
        max(abs(unbounded[kin.idx[name]] - branch_seed[kin.idx[name]]) for name in ARM_JOINTS)
        > MAX_JOINT_STEP_RAD
    )

    with pytest.raises(IKTrustRegionError, match="no solution within 0.150000 rad trust region"):
        kin.solve(
            target,
            branch_seed,
            restarts=True,
            position_tolerance=IK_TOL_GRASP,
            max_joint_step_rad=MAX_JOINT_STEP_RAD,
        )
