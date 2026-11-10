"""Validation-sweep tests on synthetic telemetry (Webots-free, ADR-0020 redesign)."""

from __future__ import annotations

import pytest

from types import SimpleNamespace
from typing import cast


from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    TABLE_FOOTPRINT_CENTRE_XY_M,
    TABLE_TOP_HEIGHT_M,
    TORSO_JOINT,
)
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.cli import _drive_pose
from vlarobot.core.types import ApplyResult
from vlarobot.execution.kinematics_validation import (
    PoseSpec,
    estimate_step_budget,
    feasible_pose_plan,
    plan_submoves,
    pose_limits,
    sample_is_safe,
    submove_bound,
)

# --- helpers / pose design ---


def test_feasible_pose_plan_is_deterministic_bounded_and_grouped():
    limits = {name: (-2.0, 2.0) for name in ARM_JOINTS}
    p1 = feasible_pose_plan(seed=0, limits=limits)
    p2 = feasible_pose_plan(seed=0, limits=limits)
    p3 = feasible_pose_plan(seed=1, limits=limits)
    assert len(p1) == 12
    assert p1 == p2
    assert p1 != p3
    assert [p.group for p in p1[:6]] == ["spread"] * 6
    assert [p.group for p in p1[6:]] == ["near_task"] * 6
    for p in p1:
        assert set(p.target) == set(ARM_JOINTS)
        for name, v in p.target.items():
            lo, hi = limits[name]
            assert lo + 0.1 - 1e-9 <= v <= hi - 0.1 + 1e-9


def test_submove_bound_and_plan_and_budget():
    bound = submove_bound(1.0, 1.0)
    assert bound == pytest.approx(0.4)
    start = {name: 0.0 for name in ARM_JOINTS}
    target = dict(start)
    target[ARM_JOINTS[0]] = 0.9
    subs = plan_submoves(start, target, bound)
    assert len(subs) == 3
    assert subs[-1][ARM_JOINTS[0]] == pytest.approx(0.9)
    plan = [PoseSpec(i, "spread" if i < 1 else "near_task", target) for i in range(2)]
    budget = estimate_step_budget(plan, start, bound, 447.0)
    assert budget["poses"] == 2
    assert budget["steps_per_submove_cap"] == 60
    assert budget["total_submoves"] == 4
    assert budget["max_total_steps"] == 240
    assert budget["estimated_s"] == pytest.approx(240 * 0.447)


def test_pose_limits_falls_back_without_snapshot():
    fallback = pose_limits(None)
    assert set(fallback) == set(ARM_JOINTS)
    assert fallback[ARM_JOINTS[0]] == (-2.0, 2.0)


# --- sample safety ---


def _base_world_pose():
    return {
        "world_xyz_m": [0.0, 0.0, 0.0],
        "world_orientation_matrix_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    }


def test_sample_is_safe_requires_complete_arm_poses_and_checks_table():
    centre_x, centre_y = TABLE_FOOTPRINT_CENTRE_XY_M
    telemetry = {
        "base_frame": _base_world_pose(),
        "joint_endpoint_solids": {
            ARM_JOINTS[0]: {
                "world_xyz_m": [centre_x, centre_y, 1.0],
                "world_orientation_matrix_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            },
        },
        "finger_endpoint_solids": {},
    }
    ok, reason = sample_is_safe(telemetry)
    assert ok is True, reason

    bad = {
        "base_frame": _base_world_pose(),
        "joint_endpoint_solids": {
            ARM_JOINTS[0]: {
                "world_xyz_m": [centre_x, centre_y, TABLE_TOP_HEIGHT_M + 0.01],
                "world_orientation_matrix_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            },
        },
        "finger_endpoint_solids": {},
    }
    ok, reason = sample_is_safe(bad)
    assert ok is False
    assert "table footprint" in reason

    missing = {"base_frame": _base_world_pose(), "joint_endpoint_solids": {}}
    ok, reason = sample_is_safe(missing)
    assert ok is False
    assert "no tracked" in reason


def test_sample_is_safe_allows_unreadable_fingers_but_not_unreadable_arm():
    telemetry = {
        "base_frame": _base_world_pose(),
        "joint_endpoint_solids": {
            ARM_JOINTS[0]: {
                "world_xyz_m": [0.0, 0.0, 1.0],
                "world_orientation_matrix_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            },
        },
        "finger_endpoint_solids": {
            "left": {"world_xyz_m": None, "world_orientation_matrix_row_major": None}
        },
    }
    ok, reason = sample_is_safe(telemetry)
    assert ok is True, reason

    bad_arm = {
        "base_frame": _base_world_pose(),
        "joint_endpoint_solids": {
            ARM_JOINTS[0]: {
                "world_xyz_m": None,
                "world_orientation_matrix_row_major": None,
            }
        },
    }
    ok, reason = sample_is_safe(bad_arm)
    assert ok is False
    assert ARM_JOINTS[0] in reason


TORSO_REF = 0.35

# --- legacy CLI driver (kept for compatibility) ---


class _FakeSweepClient:
    def __init__(self, *, accepted: bool = True, settled: bool = True, telemetry=None) -> None:
        self.accepted = accepted
        self.settled = settled
        self.telemetry = telemetry or (
            lambda angles: {
                "base_frame": _base_world_pose(),
                "joint_endpoint_solids": {
                    ARM_JOINTS[0]: {
                        "world_xyz_m": [0.0, 0.0, 1.0],
                        "world_orientation_matrix_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    },
                },
                "finger_endpoint_solids": {},
            }
        )
        self.target: dict[str, float] = {}

    def apply(self, action):
        if not self.accepted:
            return ApplyResult(False, "joint limit")
        self.target = dict(action.q)
        return ApplyResult(True)

    def step(self, n_steps: int = 1):
        q = dict(self.target)
        if not self.settled:
            q = {name: value + 1.0 for name, value in q.items()}
        angles = {name: q.get(name, 0.0) for name in ARM_JOINTS}
        return SimpleNamespace(
            t=0.0,
            robot=SimpleNamespace(
                joint_pos=q,
                diagnostics={"kinematics_telemetry": self.telemetry(angles)},
            ),
        )


def _sweep_client(client: _FakeSweepClient):
    return cast("ZmqBackendClient", client)


def test_drive_pose_collects_a_settled_safe_sample():
    client = _FakeSweepClient()
    target = {name: 0.2 * (index + 1) for index, name in enumerate(ARM_JOINTS)}
    outcome = _drive_pose(
        _sweep_client(client),
        target,
        TORSO_REF,
        max_steps=5,
        settle_tolerance=0.01,
    )
    assert outcome.sample is not None, outcome.reason
    assert outcome.sample.joints == pytest.approx(target)
    assert client.target[TORSO_JOINT] == TORSO_REF


def test_drive_pose_reports_rejections_and_unsettled_poses():
    target = {name: 0.1 for name in ARM_JOINTS}
    rejected = _drive_pose(
        _sweep_client(_FakeSweepClient(accepted=False)),
        target,
        TORSO_REF,
        max_steps=5,
        settle_tolerance=0.01,
    )
    assert rejected.sample is None
    assert rejected.reason is not None and "safety rejected" in rejected.reason

    unsettled = _drive_pose(
        _sweep_client(_FakeSweepClient(settled=False)),
        target,
        TORSO_REF,
        max_steps=3,
        settle_tolerance=0.01,
    )
    assert unsettled.sample is None
    assert unsettled.reason is not None and "did not settle" in unsettled.reason


def test_drive_pose_aborts_a_pose_that_fails_the_table_pre_check():
    centre_x, centre_y = TABLE_FOOTPRINT_CENTRE_XY_M

    def unsafe_telemetry(angles):
        return {
            "base_frame": _base_world_pose(),
            "joint_endpoint_solids": {
                ARM_JOINTS[0]: {
                    "world_xyz_m": [centre_x, centre_y, TABLE_TOP_HEIGHT_M + 0.01],
                    "world_orientation_matrix_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                },
            },
            "finger_endpoint_solids": {},
        }

    outcome = _drive_pose(
        _sweep_client(_FakeSweepClient(telemetry=unsafe_telemetry)),
        {name: 0.1 for name in ARM_JOINTS},
        TORSO_REF,
        max_steps=3,
        settle_tolerance=0.01,
    )
    assert outcome.sample is None
    assert outcome.reason is not None and "table pre-check" in outcome.reason
