import json

import pytest

from vlarobot.execution.trace_compare import compare_traces, joint_diagnostics

PREFIX = ("INIT", "PREGRASP", "DESCEND", "CLOSE", "LIFT")


def _trace(
    pregrasp_steps=(),
    *,
    joint_offset=0.0,
    pregrasp_ee_x=0.1,
    pregrasp_achieved_x=None,
    descend_x=0.2,
    lift_x=0.1,
    tilt=1.0,
):
    records = []
    targets = {
        "INIT": None,
        "PREGRASP": [0.1, 0.2, 0.3],
        "DESCEND": [0.2, 0.2, 0.3],
        "CLOSE": [0.2, 0.2, 0.3],
        "LIFT": [0.1, 0.2, 0.3],
    }
    for index, state in enumerate(targets):
        records.append(
            {
                "event": "state_boundary",
                "phase": "enter",
                "state": state,
                "commanded_joint_positions_rad": {"arm_1_joint": index / 10.0},
                "joint_limits_rad": {"arm_1_joint": [-2.0, 2.0]},
            }
        )
        if state == "PREGRASP":
            for value in pregrasp_steps:
                records.append(
                    {
                        "event": "state_boundary",
                        "phase": "waypoint",
                        "state": state,
                        "joint_target": {"arm_1_joint": value},
                    }
                )
        if state in {"DESCEND", "LIFT"}:
            target = targets[state]
            assert target is not None
            point = target.copy()
            point[0] = descend_x if state == "DESCEND" else lift_x
            records.append(
                {
                    "event": "waypoint",
                    "state": "INIT",
                    "waypoint": "descend_grasp" if state == "DESCEND" else "lift",
                    "base_xyz_m": point,
                    "ik_tilt_deg": tilt,
                }
            )
        joint_value = float(index) / 10.0 + (joint_offset if state == "PREGRASP" else 0.0)
        boundary_target = targets[state]
        if state == "PREGRASP" and boundary_target is not None:
            boundary_target = boundary_target.copy()
            boundary_target[0] = pregrasp_ee_x
        achieved_target = [0.0, 0.0, 0.0] if boundary_target is None else boundary_target.copy()
        if state == "PREGRASP" and pregrasp_achieved_x is not None:
            achieved_target[0] = pregrasp_achieved_x
        records.append(
            {
                "event": "state_boundary",
                "phase": "exit",
                "state": state,
                "commanded_joint_positions_rad": {"arm_1_joint": joint_value},
                "commanded_ee_base_xyz_m": boundary_target,
                "achieved_ee_base_xyz_m": achieved_target,
                "joint_limits_rad": {"arm_1_joint": [-2.0, 2.0]},
                "ik_tilt_deg": None if state == "INIT" else tilt,
            }
        )
    return records


def _write_trace(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _compare(tmp_path, first_records, second_records):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_trace(first, first_records)
    _write_trace(second, second_records)
    return compare_traces(first, second, PREFIX, 1e-3, 15.0)


def test_compare_ignores_joint_value_and_interpolation_differences(tmp_path):
    result = _compare(
        tmp_path,
        _trace((0.05, 0.1)),
        _trace((0.033, 0.066, 0.1), joint_offset=0.8),
    )

    assert result.passed
    assert "joint values are diagnostics" in result.reason


def test_compare_checks_boundary_ee_targets(tmp_path):
    result = _compare(tmp_path, _trace(), _trace(pregrasp_ee_x=0.11))

    assert not result.passed
    assert result.state == "PREGRASP"
    assert "boundary commanded_ee_base_xyz_m" in result.reason


def test_compare_checks_achieved_boundary_ee_positions(tmp_path):
    result = _compare(tmp_path, _trace(), _trace(pregrasp_achieved_x=0.11))

    assert not result.passed
    assert result.state == "PREGRASP"
    assert "boundary achieved_ee_base_xyz_m" in result.reason


@pytest.mark.parametrize(
    ("state", "offset"),
    [("DESCEND", {"descend_x": 0.21}), ("LIFT", {"lift_x": 0.11})],
)
def test_compare_checks_cartesian_targets(tmp_path, state, offset):
    result = _compare(tmp_path, _trace(), _trace(**offset))

    assert not result.passed
    assert result.state == state
    assert result.waypoint == 1
    assert "EE waypoint delta" in result.reason


def test_compare_checks_exact_state_sequence(tmp_path):
    first = _trace()
    second = _trace()
    second.insert(4, {"event": "state_boundary", "phase": "enter", "state": "RETRY"})

    result = _compare(tmp_path, first, second)

    assert not result.passed
    assert result.state == "DESCEND"
    assert "state sequence differs" in result.reason


def test_compare_allows_brain_trace_to_continue_after_compared_prefix(tmp_path):
    original = _trace()
    brain = _trace() + [{"event": "state_boundary", "phase": "enter", "state": "TRANSIT"}]

    result = _compare(tmp_path, original, brain)

    assert result.passed


def test_compare_enforces_existing_tilt_limit(tmp_path):
    result = _compare(tmp_path, _trace(), _trace(tilt=15.01))

    assert not result.passed
    assert "tool tilt" in result.reason


def test_compare_requires_tilt_diagnostics(tmp_path):
    first = _trace()
    second = _trace()
    for record in first + second:
        record.pop("ik_tilt_deg", None)

    result = _compare(tmp_path, first, second)

    assert not result.passed
    assert "no IK tilt diagnostics" in result.reason


def test_joint_diagnostics_reports_steps_margin_and_wrist_excursion(tmp_path):
    path = tmp_path / "joint-diagnostics.jsonl"
    limits = {"arm_6_joint": [-2.0, 2.0], "arm_7_joint": [-2.5, 2.5]}
    _write_trace(
        path,
        [
            {
                "event": "state_boundary",
                "phase": "enter",
                "state": "PREGRASP",
                "commanded_joint_positions_rad": {"arm_6_joint": 0.0, "arm_7_joint": 0.0},
                "joint_limits_rad": limits,
            },
            {
                "event": "state_boundary",
                "phase": "waypoint",
                "state": "PREGRASP",
                "joint_target": {"arm_6_joint": 0.3, "arm_7_joint": 1.2},
                "joint_limits_rad": limits,
            },
            {
                "event": "state_boundary",
                "phase": "exit",
                "state": "PREGRASP",
                "commanded_joint_positions_rad": {"arm_6_joint": 0.3, "arm_7_joint": 1.2},
                "joint_limits_rad": limits,
            },
        ],
    )

    result = joint_diagnostics(path)

    assert result.max_step_by_joint_rad == {"arm_6_joint": 0.3, "arm_7_joint": 1.2}
    assert result.min_limit_margin_rad == pytest.approx(1.3)
    assert result.wrist_roll_excursions == ("PREGRASP->PREGRASP: 1.2 rad",)


def test_joint_diagnostics_prefers_explicit_command_target_over_measured_alias(tmp_path):
    path = tmp_path / "explicit-joints.jsonl"
    limits = {"arm_7_joint": [-2.5, 2.5]}
    _write_trace(
        path,
        [
            {
                "event": "state_boundary",
                "phase": "enter",
                "state": "PREGRASP",
                "measured_joint_positions_rad": {"arm_7_joint": 0.0},
                "commanded_joint_positions_rad": {"arm_7_joint": 0.0},
                "joint_limits_rad": limits,
            },
            {
                "event": "state_boundary",
                "phase": "waypoint",
                "state": "PREGRASP",
                "measured_joint_positions_rad": {"arm_7_joint": 0.02},
                "commanded_joint_positions_rad": {"arm_7_joint": 0.02},
                "commanded_joint_target_rad": {"arm_7_joint": 0.15},
                "joint_limits_rad": limits,
            },
        ],
    )

    result = joint_diagnostics(path)

    assert result.max_step_by_joint_rad == {"arm_7_joint": 0.15}
