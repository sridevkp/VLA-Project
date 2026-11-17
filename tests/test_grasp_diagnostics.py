"""Physical diagnostic checks must distinguish enclosure proxies from a real lift."""

import json

import pytest

from vlarobot.eval.grasp_diagnostics import compare_numeric_rows, measure_sample, summarize_grasp


def test_rotated_fingers_measure_can_axis_and_off_axis_distance():
    row = {
        "obs": {
            "t": 1.0,
            "diag": {
                "finger_tip_world_xyz_m": {"left": [0, -0.04, 1], "right": [0, 0.04, 1]},
                "finger_midpoint_world_xyz_m": [0, 0, 1],
                "ee_fk_world_xyz_m": [0.05, 0, 1],
            },
        },
        "gt": {"CAN": {"pos": [0.02, 0.01, 0.98], "quat": [0, 0, 0, 1]}},
    }
    sample = measure_sample(row)
    assert sample["can_centre_between_fingers_proxy"] is True
    assert sample["can_axis_offset_m"] == pytest.approx(0.01)
    assert sample["can_distance_from_finger_axis_m"] == pytest.approx(0.02 * 2**0.5)
    assert sample["pad_minus_fk_world_xyz_m"] == [-0.05, 0, 0]
    assert "success" not in sample


def test_missing_pad_is_unavailable_instead_of_zero_error():
    sample = measure_sample(
        {
            "obs": {"t": 0},
            "gt": {
                "CAN": {"pos": [0, 0, 1], "quat": [0, 0, 0, 1]},
            },
        }
    )
    assert "pad_can_xy_error_m" not in sample
    assert sample["finger_positions_world_xyz_m"] is None


def test_trajectory_comparison_reports_early_and_length_divergence():
    result = compare_numeric_rows([{"q": {"joint": 1.0}}], [{"q": {"joint": 1.01}}])
    assert result["first_different_row"] == 0
    assert result["max_abs_difference_by_field"][".q.joint"] == pytest.approx(0.01)
    assert compare_numeric_rows([{}], [{}, {}])["first_different_row"] == 1


def test_home_without_can_motion_does_not_establish_lift_or_success(tmp_path):
    diag = tmp_path / "diag.jsonl"
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"obs": {"t": t}, "gt": {"CAN": {"pos": [0, 0, 0.81], "quat": [0, 0, 0, 1]}}}
        for t in [0.0, 1.0]
    ]
    diag.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    trace.write_text(
        json.dumps({"event": "state_boundary", "state": "HOME", "phase": "exit", "sim_time_s": 1}),
        encoding="utf-8",
    )
    report = summarize_grasp(diag, trace)
    assert report["max_can_rise_m"] == 0.0
    assert report["final_can_rise_m"] == 0.0
    assert report["boundaries"][0]["sample_time_error_s"] == 0.0
    assert "success" not in report
