"""Injected controller-device and protocol checks with no simulator or socket connection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlarobot.backends.webots.finger_admission_probe import capture_finger_admission
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import SimServer
from vlarobot.core.wire import make_message

LEFT = "gripper_left_finger_joint"
RIGHT = "gripper_right_finger_joint"
CAPTURED_LEFT = -7.142594888992681e-07
CAPTURED_RIGHT = 7.171095938981087e-07


def _sensor(name, value):
    return SimpleNamespace(
        getName=lambda: name, getValue=lambda: value, getSamplingPeriod=lambda: 32
    )


def _motor(name, sensor):
    return SimpleNamespace(
        getName=lambda: name,
        getPositionSensor=lambda: sensor,
        getMinPosition=lambda: 0.0,
        getMaxPosition=lambda: 0.045,
    )


def _backend(*, missing=False, swapped=False):
    left = _sensor("left_sensor", CAPTURED_LEFT)
    right = _sensor("right_sensor", CAPTURED_RIGHT)
    motors = {LEFT: _motor(LEFT, left), RIGHT: _motor(RIGHT, None if missing else right)}
    return SimpleNamespace(
        diagnostic_step_capability=("webots", 1),
        diagnostic_step=lambda: pytest.fail("the probe must not advance physics"),
        _observation=lambda: pytest.fail("the probe must not build an observation"),
        _physics_step_count=4,
        io=SimpleNamespace(
            robot=SimpleNamespace(getTime=lambda: 0.128, getDevice=motors.get),
            fingers=[(motors[LEFT], right if swapped else left), (motors[RIGHT], right)],
            _joint_position_field=lambda motor, name: SimpleNamespace(
                getSFFloat=lambda: CAPTURED_LEFT
            ),
        ),
    )


@pytest.mark.parametrize(("missing", "swapped"), [(False, False), (True, False), (False, True)])
def test_raw_device_and_cached_mapping_remain_distinct(missing, swapped):
    snapshot = capture_finger_admission(_backend(missing=missing, swapped=swapped))
    left, right = snapshot["readings"]
    assert left["raw_coordinate_m"] == CAPTURED_LEFT
    assert left["supervisor_position_field_m"] == CAPTURED_LEFT
    assert left["admission"] == "outside_reviewed_limits"
    assert left["mapping"] == ("mismatch_or_unstable" if swapped else "matches")
    assert right["mapping"] == ("missing_sensor" if missing else "matches")
    if missing:
        assert "raw_coordinate_m" not in right


def test_finger_probe_requires_guard_and_preserves_owned_session(monkeypatch):
    server = SimServer(_backend())
    requests = []
    client = ZmqBackendClient()

    def exchange(operation, arguments, timeout_ms):
        requests.append(operation)
        reply = server.handle(make_message(operation, 0, arguments))
        if reply["error"] is not None:
            raise RuntimeError(reply["error"])
        return reply

    monkeypatch.setattr(client, "_exchange", exchange)
    try:
        with pytest.raises(ValueError, match="pinned"):
            client.inspect_fingers()
        assert "guarded" in server.handle(make_message("inspect_fingers", 0))["error"]
        client.require_diagnostic_session(expected_backend="webots")
        result = client.inspect_fingers()
        assert result["session_id"] == server.session_id
        assert result["physics_step_count"] == 4
        server.session_id = {**server.session_id, "nonce": "replacement"}
        with pytest.raises(RuntimeError, match="identity_mismatch"):
            client.inspect_fingers()
        with pytest.raises(ValueError, match="closed"):
            client.inspect_fingers()
        assert requests == ["ping", "diagnostic_session_request", "diagnostic_session_request"]
    finally:
        client.close()


def test_clock_change_invalidates_read_only_capture():
    backend = _backend()
    times = iter([0.128, 0.160])
    backend.io.robot.getTime = lambda: next(times)
    with pytest.raises(RuntimeError, match="physics boundary"):
        capture_finger_admission(backend)


def test_invalid_raw_type_is_not_coerced_to_a_valid_endpoint():
    backend = _backend()
    backend.io.fingers[0][1].getValue = lambda: False
    row = capture_finger_admission(backend)["readings"][0]
    assert row["raw_coordinate_m"] is None
    assert row["raw_reading_type"] == "bool"
    assert row["admission"] == "invalid_or_nonfinite"
