import json

from vlarobot.execution.trace_compare import compare_traces


def _record(event, phase, state, *, xyz=None, joints=None):
    record = {"event": event, "phase": phase, "state": state}
    if phase == "waypoint":
        record["commanded_base_xyz_m"] = xyz
        record["joint_target"] = joints
    return record


def _write_trace(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_compare_traces_accepts_matching_state_and_waypoint_prefix(tmp_path):
    records = [
        _record("state_boundary", "enter", "INIT"),
        _record("state_boundary", "exit", "INIT"),
        _record("state_boundary", "enter", "PREGRASP"),
        _record(
            "state_boundary",
            "waypoint",
            "PREGRASP",
            xyz=[0.1, 0.2, 0.3],
            joints={"arm_1_joint": 0.2},
        ),
    ]
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_trace(first, records)
    _write_trace(second, records)

    result = compare_traces(first, second, ("INIT", "PREGRASP"), 1e-3, 1e-3)

    assert result.passed
    assert result.reason == "matched"


def test_compare_traces_reports_first_joint_waypoint_divergence(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_trace(
        first,
        [
            _record("state_boundary", "enter", "INIT"),
            _record("state_boundary", "enter", "PREGRASP"),
            _record(
                "state_boundary",
                "waypoint",
                "PREGRASP",
                xyz=[0.1, 0.2, 0.3],
                joints={"arm_1_joint": 0.2},
            ),
        ],
    )
    _write_trace(
        second,
        [
            _record("state_boundary", "enter", "INIT"),
            _record("state_boundary", "enter", "PREGRASP"),
            _record(
                "state_boundary",
                "waypoint",
                "PREGRASP",
                xyz=[0.1, 0.2, 0.3],
                joints={"arm_1_joint": 0.25},
            ),
        ],
    )

    result = compare_traces(first, second, ("INIT", "PREGRASP"), 1e-3, 1e-3)

    assert not result.passed
    assert result.state == "PREGRASP"
    assert result.waypoint == 1
    assert "joint waypoint delta" in result.reason


def test_compare_traces_reports_waypoint_count_difference(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_trace(
        first,
        [
            _record("state_boundary", "enter", "INIT"),
            _record("state_boundary", "enter", "PREGRASP"),
            _record(
                "state_boundary",
                "waypoint",
                "PREGRASP",
                xyz=[0.1, 0.2, 0.3],
                joints={"arm_1_joint": 0.2},
            ),
        ],
    )
    _write_trace(
        second,
        [
            _record("state_boundary", "enter", "INIT"),
            _record("state_boundary", "enter", "PREGRASP"),
        ],
    )

    result = compare_traces(first, second, ("INIT", "PREGRASP"), 1e-3, 1e-3)

    assert not result.passed
    assert result.state == "PREGRASP"
    assert result.waypoint == 1


def test_compare_traces_reports_first_state_sequence_divergence(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write_trace(
        first,
        [
            _record("state_boundary", "enter", "INIT"),
            _record("state_boundary", "enter", "DESCEND"),
        ],
    )
    _write_trace(
        second,
        [
            _record("state_boundary", "enter", "INIT"),
            _record("state_boundary", "enter", "PREGRASP"),
        ],
    )

    result = compare_traces(first, second, ("INIT", "PREGRASP"), 1e-3, 1e-3)

    assert not result.passed
    assert result.state == "PREGRASP"
    assert result.waypoint is None
    assert "state 2" in result.reason
