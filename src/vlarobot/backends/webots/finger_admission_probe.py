"""Getter-only finger binding probe for the owned diagnostic controller session."""

from __future__ import annotations

import math
from typing import Any, cast

from vlarobot.backends.webots.config import SETTINGS


def _bound(value: object, lower: float, upper: float) -> str:
    numeric = cast("int | float", value)
    if type(value) not in (int, float) or not math.isfinite(numeric):
        return "invalid_or_nonfinite"
    return "in_bounds" if lower <= numeric <= upper else "outside_reviewed_limits"


def capture_finger_admission(backend: Any) -> dict[str, Any]:
    """Compare named devices to cached pairs without building an observation or stepping."""
    io = backend.io
    robot = io.robot
    time_before = float(robot.getTime())
    steps_before = backend._physics_step_count
    lower, upper = SETTINGS.gripper_opening_limits
    readings = []
    for name in SETTINGS.finger_joints:
        motor = robot.getDevice(name)
        sensor = None if motor is None else motor.getPositionSensor()
        cached = [(m, s) for m, s in io.fingers if m.getName() == name]
        row: dict[str, Any] = {"expected_motor": name, "cached_pair_count": len(cached)}
        if motor is None or sensor is None:
            row.update(mapping="missing_motor" if motor is None else "missing_sensor")
            readings.append(row)
            continue

        actual_name = motor.getName()
        sensor_name = sensor.getName()
        reading = sensor.getValue()
        row.update(
            resolved_motor=actual_name,
            resolved_sensor=sensor_name,
            sampling_period_ms=sensor.getSamplingPeriod(),
            raw_coordinate_m=reading,
            motor_limits_m=[float(motor.getMinPosition()), float(motor.getMaxPosition())],
            admission=_bound(reading, lower, upper),
        )
        if type(reading) not in (int, float):
            row["raw_coordinate_m"] = None
            row["raw_reading_type"] = type(reading).__name__
        if len(cached) != 1 or cached[0][1] is None or actual_name != name:
            row["mapping"] = "unreconciled_cached_pair"
        else:
            cached_sensor = cached[0][1]
            cached_reading = cached_sensor.getValue()
            row.update(
                cached_sensor=cached_sensor.getName(),
                cached_raw_coordinate_m=(
                    cached_reading if type(cached_reading) in (int, float) else None
                ),
                mapping=(
                    "matches"
                    if cached_sensor.getName() == sensor_name
                    and type(cached_reading) in (int, float)
                    and type(reading) in (int, float)
                    and cached_reading == reading
                    else "mismatch_or_unstable"
                ),
            )
        try:
            # Read the field directly. The reset helper normalizes positions to legal limits.
            row["supervisor_position_field_m"] = float(
                io._joint_position_field(motor, name).getSFFloat()  # noqa: PLC2701
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            row["supervisor_position_field_error"] = f"{type(exc).__name__}: {exc}"
        readings.append(row)

    if float(robot.getTime()) != time_before or backend._physics_step_count != steps_before:
        raise RuntimeError("finger probe crossed a physics boundary")
    return {
        "scope": "read_only_finger_admission",
        "simulator_time_s": time_before,
        "physics_step_count": steps_before,
        "reviewed_limits_m": [lower, upper],
        "readings": readings,
    }


def idle_finger_step(backend: Any) -> dict[str, Any]:
    """Advance one isolated physics tick without visualization or observation callbacks."""
    before = backend._physics_step_count
    if backend.io.ts != 32 or type(before) is not int or not 0 <= before < 32:
        raise ValueError("idle finger step budget or time step mismatch")
    robot = backend.io.robot
    time_before = float(robot.getTime())
    try:
        status = robot.step(32)
    except Exception as exc:
        return {"confirmed_ticks": None, "error": str(exc)}
    if status != 0:
        return {"confirmed_ticks": None, "error": f"unexpected physics status {status!r}"}
    backend._physics_step_count = before + 1
    return {
        "confirmed_ticks": 1,
        "physics_step_count": before + 1,
        "simulator_time_s": float(robot.getTime()),
        "previous_simulator_time_s": time_before,
    }


def capture_idle_fingers(backend: Any) -> dict[str, Any]:
    """Add world-frame slider endpoint geometry using Supervisor getters only."""
    snapshot = capture_finger_admission(backend)
    robot = backend.io.robot
    snapshot["base_position_world_m"] = list(backend.io.me.getPosition())
    snapshot["base_orientation_world"] = list(backend.io.me.getOrientation())
    for row in snapshot["readings"]:
        motor = robot.getDevice(row["expected_motor"])
        node = robot.getFromDevice(motor._tag)
        while node is not None and node.getTypeName() != "SliderJoint":
            node = node.getParentNode()
        if node is None:
            raise RuntimeError("slider joint unavailable")
        endpoint = node.getField("endPoint").getSFNode()
        row["endpoint_position_world_m"] = list(endpoint.getPosition())
        row["endpoint_orientation_world"] = list(endpoint.getOrientation())
    if (
        robot.getTime() != snapshot["simulator_time_s"]
        or backend._physics_step_count != snapshot["physics_step_count"]
    ):
        raise RuntimeError("geometry capture crossed a physics boundary")
    return snapshot
