"""Compare controller and brain JSONL waypoint traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceComparison:
    passed: bool
    state: str | None
    waypoint: int | None
    reason: str


def compare_traces(
    first_path: Path,
    second_path: Path,
    prefix: tuple[str, ...],
    tol_joint: float,
    tol_pos: float,
) -> TraceComparison:
    if not prefix or any(not state for state in prefix):
        raise ValueError("prefix must contain one or more state names.")
    if tol_joint < 0.0 or tol_pos < 0.0:
        raise ValueError("comparison tolerances must be non-negative.")
    first = _load_trace(first_path)
    second = _load_trace(second_path)

    for label, records in (("A", first), ("B", second)):
        states = _state_sequence(records)
        for index, expected in enumerate(prefix):
            actual = states[index] if index < len(states) else None
            if actual != expected:
                return TraceComparison(
                    False,
                    expected,
                    None,
                    f"trace {label} differs at state {index + 1}: expected {expected!r}, "
                    f"found {actual!r}",
                )

    for state in prefix:
        first_waypoints = _waypoints(first, state)
        second_waypoints = _waypoints(second, state)
        if len(first_waypoints) != len(second_waypoints):
            return TraceComparison(
                False,
                state,
                min(len(first_waypoints), len(second_waypoints)) + 1,
                f"waypoint count differs: {len(first_waypoints)} != {len(second_waypoints)}",
            )
        for index, (first_point, second_point) in enumerate(
            zip(first_waypoints, second_waypoints, strict=True), start=1
        ):
            delta = _joint_delta(first_point, second_point)
            if delta is not None and delta > tol_joint:
                return TraceComparison(
                    False,
                    state,
                    index,
                    f"joint waypoint delta {delta:.9g} rad exceeds {tol_joint:.9g} rad",
                )
            position_delta = _position_delta(first_point, second_point)
            if position_delta > tol_pos:
                return TraceComparison(
                    False,
                    state,
                    index,
                    f"EE waypoint delta {position_delta:.9g} m exceeds {tol_pos:.9g} m",
                )
    return TraceComparison(True, None, None, "matched")


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
            if record.get("event") == "state_boundary":
                records.append(record)
    return records


def _state_sequence(records: list[dict[str, Any]]) -> tuple[str, ...]:
    states: list[str] = []
    for record in records:
        state = record.get("state")
        if record.get("phase") == "enter" and isinstance(state, str):
            if not states or states[-1] != state:
                states.append(state)
    return tuple(states)


def _waypoints(records: list[dict[str, Any]], state: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("state") == state and record.get("phase") == "waypoint"
    ]


def _joint_delta(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_joints = first.get("joint_target")
    second_joints = second.get("joint_target")
    if not isinstance(first_joints, dict) or not isinstance(second_joints, dict):
        raise ValueError("waypoint records must include joint_target mappings.")
    if first_joints.keys() != second_joints.keys():
        return float("inf")
    return max(
        (abs(float(first_joints[name]) - float(second_joints[name])) for name in first_joints),
        default=0.0,
    )


def _position_delta(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_position = first.get("commanded_base_xyz_m")
    second_position = second.get("commanded_base_xyz_m")
    if not isinstance(first_position, list) or not isinstance(second_position, list):
        raise ValueError("waypoint records must include commanded_base_xyz_m arrays.")
    if len(first_position) != 3 or len(second_position) != 3:
        raise ValueError("commanded_base_xyz_m must contain three coordinates.")
    return (
        sum(
            (float(first_position[index]) - float(second_position[index])) ** 2
            for index in range(3)
        )
        ** 0.5
    )
