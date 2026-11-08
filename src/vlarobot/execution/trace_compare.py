"""Compare original-controller and brain traces at M1-06a contract points."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceComparison:
    passed: bool
    state: str | None
    waypoint: int | None
    reason: str
    first_pregrasp_realign: bool | None = None
    second_pregrasp_realign: bool | None = None


@dataclass(frozen=True)
class JointDiagnostics:
    max_step_by_joint_rad: dict[str, float]
    min_limit_margin_rad: float | None
    wrist_roll_excursions: tuple[str, ...]


@dataclass(frozen=True)
class StateBaseDrift:
    """Base drift observed while a state was active, relative to the first boundary."""

    state: str
    max_distance_m: float
    max_yaw_deg: float
    end_distance_m: float
    end_yaw_deg: float
    sim_time_s: float
    max_rate_m_per_s: float = 0.0
    max_tilt_deg: float = 0.0

    @property
    def rate_m_per_s(self) -> float:
        return self.max_distance_m / self.sim_time_s if self.sim_time_s > 0.0 else 0.0


def compare_traces(
    first_path: Path,
    second_path: Path,
    prefix: tuple[str, ...],
    tol_pos: float,
    max_tilt_deg: float = 15.0,
    init_torso_tolerance_m: float = 2e-3,
    init_head_tolerance_rad: float = 0.02,
) -> TraceComparison:
    if not prefix or any(not state for state in prefix):
        raise ValueError("prefix must contain one or more state names.")
    if tol_pos < 0.0 or max_tilt_deg < 0.0:
        raise ValueError("position and tilt limits must be non-negative.")
    if init_torso_tolerance_m < 0.0 or init_head_tolerance_rad < 0.0:
        raise ValueError("INIT posture tolerances must be non-negative.")
    first = _load_trace(first_path)
    second = _load_trace(second_path)
    first_flag = _pregrasp_realign(first)
    second_flag = _pregrasp_realign(second)
    if first_flag is not None and second_flag is not None and first_flag != second_flag:
        return TraceComparison(
            False,
            None,
            None,
            f"pregrasp_realign differs: A={first_flag} B={second_flag}; "
            "realign off keeps original semantics, on adds waypoints/events (ADR-0016)",
            first_flag,
            second_flag,
        )

    for label, records in (("A", first), ("B", second)):
        states = _state_sequence(records)
        valid_sequence = states == prefix if label == "A" else states[: len(prefix)] == prefix
        if not valid_sequence:
            index = next(
                (
                    i
                    for i, (actual, expected) in enumerate(zip(states, prefix, strict=False))
                    if actual != expected
                ),
                min(len(states), len(prefix)),
            )
            expected = prefix[index] if index < len(prefix) else None
            actual = states[index] if index < len(states) else None
            return TraceComparison(
                False,
                expected or actual,
                None,
                f"trace {label} state sequence differs at position {index + 1}: "
                f"expected {expected!r}, found {actual!r}; full sequence={states!r}",
                first_flag,
                second_flag,
            )

    for state in prefix:
        first_boundary = _boundary(first, state)
        second_boundary = _boundary(second, state)
        for field in ("commanded_ee_base_xyz_m", "achieved_ee_base_xyz_m"):
            first_value = first_boundary.get(field)
            second_value = second_boundary.get(field)
            if first_value is None and second_value is None and field.startswith("commanded_"):
                continue
            if first_value is None or second_value is None:
                return TraceComparison(
                    False,
                    state,
                    None,
                    f"boundary {field} is missing",
                    first_flag,
                    second_flag,
                )
            delta = _position_delta(
                {"commanded_base_xyz_m": first_value},
                {"commanded_base_xyz_m": second_value},
            )
            if delta > tol_pos:
                return TraceComparison(
                    False,
                    state,
                    None,
                    f"boundary {field} delta {delta:.9g} m exceeds {tol_pos:.9g} m",
                    first_flag,
                    second_flag,
                )

    if "INIT" in prefix:
        first_init = _boundary(first, "INIT")
        second_init = _boundary(second, "INIT")
        first_torso = first_init.get("torso_lift_joint_sensor_rad")
        second_torso = second_init.get("torso_lift_joint_sensor_rad")
        if isinstance(first_torso, (int, float)) and isinstance(second_torso, (int, float)):
            torso_delta = abs(float(first_torso) - float(second_torso))
            if torso_delta > init_torso_tolerance_m:
                return TraceComparison(
                    False,
                    "INIT",
                    None,
                    f"INIT torso sensor delta {torso_delta:.9g} m "
                    f"exceeds {init_torso_tolerance_m:.9g} m",
                    first_flag,
                    second_flag,
                )
        first_head = first_init.get("head_joint_sensor_rad")
        second_head = second_init.get("head_joint_sensor_rad")
        if isinstance(first_head, dict) and isinstance(second_head, dict):
            for name in sorted(set(first_head) & set(second_head)):
                first_value = first_head[name]
                second_value = second_head[name]
                if not isinstance(first_value, (int, float)) or not isinstance(
                    second_value, (int, float)
                ):
                    continue
                delta = abs(float(first_value) - float(second_value))
                if delta > init_head_tolerance_rad:
                    return TraceComparison(
                        False,
                        "INIT",
                        None,
                        f"INIT head joint {name} delta {delta:.9g} rad "
                        f"exceeds {init_head_tolerance_rad:.9g} rad",
                        first_flag,
                        second_flag,
                    )

    for state in ("DESCEND", "LIFT"):
        if state not in prefix:
            continue
        first_waypoints = _cartesian_waypoints(first, state)
        second_waypoints = _cartesian_waypoints(second, state)
        if len(first_waypoints) != len(second_waypoints):
            return TraceComparison(
                False,
                state,
                min(len(first_waypoints), len(second_waypoints)) + 1,
                f"waypoint count differs: {len(first_waypoints)} != {len(second_waypoints)}",
                first_flag,
                second_flag,
            )
        for index, (first_point, second_point) in enumerate(
            zip(first_waypoints, second_waypoints, strict=True), start=1
        ):
            delta = _position_delta(first_point, second_point)
            if delta > tol_pos:
                return TraceComparison(
                    False,
                    state,
                    index,
                    f"EE waypoint delta {delta:.9g} m exceeds {tol_pos:.9g} m",
                    first_flag,
                    second_flag,
                )

    for label, records in (("A", first), ("B", second)):
        tilts = [
            (record.get("state"), float(record["ik_tilt_deg"]))
            for record in records
            if record.get("state") in prefix and isinstance(record.get("ik_tilt_deg"), (int, float))
        ]
        if not tilts:
            return TraceComparison(
                False,
                None,
                None,
                f"trace {label} has no IK tilt diagnostics",
                first_flag,
                second_flag,
            )
        for state, tilt in tilts:
            if abs(tilt) > max_tilt_deg:
                return TraceComparison(
                    False,
                    state if isinstance(state, str) else None,
                    None,
                    f"trace {label} tool tilt {tilt:.9g} deg exceeds {max_tilt_deg:.9g} deg",
                    first_flag,
                    second_flag,
                )

    return TraceComparison(
        True,
        None,
        None,
        "matched state sequence, commanded/achieved boundary EE positions, and "
        "DESCEND/LIFT Cartesian targets; "
        "all recorded IK tilts are within the configured limit; joint values are diagnostics",
        first_flag,
        second_flag,
    )


def joint_diagnostics(path: Path, states: tuple[str, ...] | None = None) -> JointDiagnostics:
    """Summarize consecutive joint command steps and margins from a JSONL trace."""
    records = _load_trace(path)
    previous: dict[str, float] | None = None
    previous_state: str | None = None
    maximum: dict[str, float] = {}
    wrist_excursions: list[str] = []
    min_margin: float | None = None

    for record in records:
        if states is not None and record.get("state") not in states:
            continue
        sample = _joint_sample(record)
        if sample is None:
            continue
        if previous is not None:
            deltas = {
                name: abs(value - previous[name])
                for name, value in sample.items()
                if name in previous
            }
            for name, delta in deltas.items():
                maximum[name] = max(maximum.get(name, 0.0), delta)
            if deltas.get("arm_7_joint", 0.0) > 1.0:
                wrist_excursions.append(
                    f"{previous_state or '?'}->{record.get('state', '?')}: "
                    f"{deltas['arm_7_joint']:.6g} rad"
                )
        previous = sample
        previous_state = str(record.get("state", "?"))

        limits = record.get("joint_limits_rad")
        measured = record.get("measured_joint_positions_rad")
        if not isinstance(measured, dict):
            measured = record.get("joint_positions_rad")
        if not isinstance(measured, dict):
            measured = sample
        if isinstance(limits, dict):
            for name, value in measured.items():
                if name not in limits:
                    continue
                bound = limits.get(name)
                if isinstance(bound, list) and len(bound) == 2:
                    margin = min(value - float(bound[0]), float(bound[1]) - value)
                    min_margin = margin if min_margin is None else min(min_margin, margin)

    return JointDiagnostics(maximum, min_margin, tuple(wrist_excursions))


def base_drift_by_state(path: Path) -> dict[str, StateBaseDrift]:
    """Maximum base drift per state, relative to the first recorded base pose.

    Drift is planar distance and yaw relative to the first ``state_boundary`` record that
    contains ``base_world_xyz_m``. This works for older traces that predate the explicit
    ``base_drift_*`` fields. ``sim_time_s`` is the cumulative episode time at the last record
    for that state, matching the ``base_drift_rate_m_per_s`` field written by the policy.
    """
    records = _load_trace(path)
    boundaries = [
        record
        for record in records
        if record.get("event") == "state_boundary"
        and isinstance(record.get("base_world_xyz_m"), (list, tuple))
        and len(record["base_world_xyz_m"]) == 3
    ]
    if not boundaries:
        return {}
    reference = boundaries[0]["base_world_xyz_m"]
    reference_x, reference_y = float(reference[0]), float(reference[1])
    reference_yaw = _yaw_deg(
        boundaries[0].get("base_world_orientation_xyzw"),
        boundaries[0].get("base_world_orientation_matrix"),
    )

    accumulated: dict[str, dict[str, float]] = {}
    last_samples: dict[str, tuple[float, float]] = {}
    for record in boundaries:
        state = record.get("state")
        if not isinstance(state, str):
            continue
        xyz = record["base_world_xyz_m"]
        distance = math.hypot(float(xyz[0]) - reference_x, float(xyz[1]) - reference_y)
        yaw = _yaw_deg(
            record.get("base_world_orientation_xyzw"),
            record.get("base_world_orientation_matrix"),
        )
        if yaw is None or reference_yaw is None:
            yaw_delta = 0.0
        else:
            yaw_delta = math.degrees(
                math.atan2(math.sin(yaw - reference_yaw), math.cos(yaw - reference_yaw))
            )
        sim_time = record.get("sim_time_s")
        sim_time_s = float(sim_time) if isinstance(sim_time, (int, float)) else 0.0
        bucket = accumulated.setdefault(
            state,
            {
                "max_distance_m": 0.0,
                "max_yaw_deg": 0.0,
                "sim_time_s": 0.0,
                "max_rate_m_per_s": 0.0,
                "max_tilt_deg": 0.0,
            },
        )
        previous = last_samples.get(state)
        if previous is not None:
            previous_time, previous_distance = previous
            elapsed = sim_time_s - previous_time
            if elapsed > 0.0:
                rate = abs(distance - previous_distance) / elapsed
                bucket["max_rate_m_per_s"] = max(bucket["max_rate_m_per_s"], rate)
        last_samples[state] = (sim_time_s, distance)
        bucket["max_distance_m"] = max(bucket["max_distance_m"], distance)
        bucket["max_yaw_deg"] = max(bucket["max_yaw_deg"], abs(yaw_delta))
        bucket["end_distance_m"] = distance
        bucket["end_yaw_deg"] = yaw_delta
        bucket["sim_time_s"] = sim_time_s
        for key in ("base_tilt_roll_deg", "base_tilt_pitch_deg"):
            tilt = record.get(key)
            if isinstance(tilt, (int, float)):
                bucket["max_tilt_deg"] = max(bucket["max_tilt_deg"], abs(float(tilt)))

    return {
        state: StateBaseDrift(
            state=state,
            max_distance_m=bucket["max_distance_m"],
            max_yaw_deg=bucket["max_yaw_deg"],
            end_distance_m=bucket["end_distance_m"],
            end_yaw_deg=bucket["end_yaw_deg"],
            sim_time_s=bucket["sim_time_s"],
            max_rate_m_per_s=bucket["max_rate_m_per_s"],
            max_tilt_deg=bucket["max_tilt_deg"],
        )
        for state, bucket in accumulated.items()
    }


def _yaw_deg(quaternion: Any, matrix: Any = None) -> float | None:
    if isinstance(quaternion, (list, tuple)) and len(quaternion) == 4:
        x, y, z, w = (float(value) for value in quaternion)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if isinstance(matrix, (list, tuple)) and len(matrix) == 9:
        values = [float(value) for value in matrix]
        return math.atan2(values[3], values[0])
    return None


def _joint_sample(record: dict[str, Any]) -> dict[str, float] | None:
    target = record.get("commanded_joint_target_rad", record.get("joint_target"))
    if isinstance(target, dict):
        return {name: float(value) for name, value in target.items()}
    measured = record.get("measured_joint_positions_rad")
    if isinstance(measured, dict):
        return {name: float(value) for name, value in measured.items()}
    if record.get("event") == "state_boundary" and record.get("phase") == "enter":
        positions = record.get("measured_joint_positions_rad")
        if not isinstance(positions, dict):
            positions = record.get("commanded_joint_positions_rad")
        if isinstance(positions, dict):
            return {name: float(value) for name, value in positions.items()}
    return None


def _pregrasp_realign(records: list[dict[str, Any]]) -> bool | None:
    """Return the ``pregrasp_realign`` flag recorded in a trace, if present."""
    for record in records:
        value = record.get("pregrasp_realign")
        if isinstance(value, bool):
            return value
    return None


def _load_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: trace record must be a JSON object.")
            if record.get("event") in {"state_boundary", "waypoint"}:
                records.append(record)
    return records


def _state_sequence(records: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(record["state"])
        for record in records
        if record.get("event") == "state_boundary"
        and record.get("phase") == "enter"
        and isinstance(record.get("state"), str)
    )


def _boundary(records: list[dict[str, Any]], state: str) -> dict[str, Any]:
    exits = [
        record
        for record in records
        if record.get("state") == state
        and record.get("event") == "state_boundary"
        and record.get("phase") == "exit"
    ]
    return exits[-1] if exits else {}


def _cartesian_waypoints(records: list[dict[str, Any]], state: str) -> list[dict[str, Any]]:
    names = {"DESCEND": {"descend_grasp", "DESCEND"}, "LIFT": {"lift", "LIFT"}}[state]
    result = []
    for record in records:
        if record.get("event") != "waypoint":
            continue
        if record.get("waypoint") not in names and record.get("state") != state:
            continue
        point = record.get("base_xyz_m", record.get("commanded_base_xyz_m"))
        if isinstance(point, list) and len(point) == 3:
            result.append({"commanded_base_xyz_m": point})
    return result


def _position_delta(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_position = first.get("commanded_base_xyz_m", first.get("commanded_ee_base_xyz_m"))
    second_position = second.get("commanded_base_xyz_m", second.get("commanded_ee_base_xyz_m"))
    if not isinstance(first_position, (list, tuple)) or not isinstance(
        second_position, (list, tuple)
    ):
        raise ValueError("position records must include a three-coordinate target.")
    if len(first_position) != 3 or len(second_position) != 3:
        raise ValueError("position targets must contain three coordinates.")
    return (
        sum(
            (float(first_position[index]) - float(second_position[index])) ** 2
            for index in range(3)
        )
        ** 0.5
    )
