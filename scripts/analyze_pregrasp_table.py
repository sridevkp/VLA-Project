"""Webots-free FK sweep of PREGRASP joint paths against the factory table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    MAX_JOINT_STEP_RAD,
    TABLE_CLEARANCE_REQUIRED_M,
    TABLE_FOOTPRINT_CENTRE_XY_M,
    TABLE_FOOTPRINT_SIZE_XY_M,
    TABLE_TOP_HEIGHT_M,
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import Pose3D, Vec3
from vlarobot.execution.scripted_can import (
    ScriptedCanPolicy,
    _arm_clearance_points,
    _base_to_world,
    _table_footprint_bounds,
)

DT_S = 0.032
MOTOR_SPEED_RAD_S = 0.5
SETTLE_TOLERANCE_RAD = 0.01


def _record_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _path(start: np.ndarray, goal: np.ndarray, step: float) -> list[np.ndarray]:
    count = max(
        1,
        math.ceil(max(abs(goal[index] - start[index]) for index in range(len(start))) / step),
    )
    return [start + (goal - start) * index / count for index in range(1, count + 1)]


def _independent_motor_path(
    kin: Kinematics, start: np.ndarray, goal: np.ndarray
) -> list[np.ndarray]:
    q = start.copy()
    result: list[np.ndarray] = []
    per_step = MOTOR_SPEED_RAD_S * DT_S
    while (
        max(abs(q[kin.idx[name]] - goal[kin.idx[name]]) for name in ARM_JOINTS)
        > SETTLE_TOLERANCE_RAD
    ):
        for name in ARM_JOINTS:
            index = kin.idx[name]
            delta = goal[index] - q[index]
            q[index] += math.copysign(min(abs(delta), per_step), delta) if delta else 0.0
        result.append(q.copy())
    return result


def _table_metrics(
    kin: Kinematics,
    samples: list[np.ndarray],
    base: Pose3D,
    torso: float,
) -> dict[str, object]:
    first = None
    minimum = math.inf
    minimum_detail = None
    x_min, x_max, y_min, y_max = _table_footprint_bounds()
    required_z = TABLE_TOP_HEIGHT_M + TABLE_CLEARANCE_REQUIRED_M
    for step_index, sample in enumerate(samples, start=1):
        q = sample.copy()
        q[kin.idx[TORSO_JOINT]] = torso
        for point_name, point_base, radius in _arm_clearance_points(kin, q):
            point_world = _base_to_world(point_base, base)
            inside = (
                x_min - radius <= point_world[0] <= x_max + radius
                and y_min - radius <= point_world[1] <= y_max + radius
            )
            if not inside:
                continue
            clearance = float(point_world[2] - radius - required_z)
            if clearance < minimum:
                minimum = clearance
                minimum_detail = {
                    "step": step_index,
                    "point": point_name,
                    "base_xyz_m": point_base.tolist(),
                    "world_xyz_m": point_world.tolist(),
                }
            if first is None and clearance < 0.0:
                first = {
                    "step": step_index,
                    "point": point_name,
                    "clearance_margin_m": clearance,
                    "base_xyz_m": point_base.tolist(),
                    "world_xyz_m": point_world.tolist(),
                }
    return {
        "first_intrusion": first,
        "minimum_margin_m": None if math.isinf(minimum) else minimum,
        "minimum_point": minimum_detail,
        "sample_count": len(samples),
    }


def _read_start_goal(
    rows: list[dict[str, object]], kin: Kinematics
) -> tuple[np.ndarray, np.ndarray, Pose3D]:
    start_row = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "INIT"
        and row.get("phase") == "enter"
    )
    waypoint_rows = [
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "PREGRASP"
        and row.get("phase") == "waypoint"
    ]
    if not waypoint_rows:
        raise ValueError("Trace has no PREGRASP joint waypoint records.")
    final_row = waypoint_rows[-1]
    q_start = np.zeros(len(kin.chain.links), dtype=float)
    q_goal = q_start.copy()
    for name, value in start_row["measured_joint_positions_rad"].items():
        if name in kin.idx:
            q_start[kin.idx[name]] = float(value)
    for name, value in final_row["commanded_joint_target_rad"].items():
        q_goal[kin.idx[name]] = float(value)
    base = Pose3D(
        Vec3(*start_row["base_world_xyz_m"]),
        tuple(start_row["base_world_orientation_xyzw"]),
    )
    return q_start, q_goal, base


def _safe_via(kin: Kinematics, start: np.ndarray, goal: np.ndarray, base: Pose3D) -> np.ndarray:
    policy = ScriptedCanPolicy(kin)
    policy._seed = 0
    via = policy._find_safe_via(start, goal, base)
    if via is None:
        raise RuntimeError("No safe_via posture found for this reset pose and torso height.")
    return via


def _route_summary(
    kin: Kinematics,
    segments: list[list[np.ndarray]],
    base: Pose3D,
    torso: float,
    start: np.ndarray,
) -> dict[str, object]:
    samples = [sample for segment in segments for sample in segment]
    max_dq = 0.0
    min_limit_margin = math.inf
    sim_time = 0.0
    previous = start
    for name in ARM_JOINTS:
        lo, hi = kin.bounds(name)
        value = start[kin.idx[name]]
        min_limit_margin = min(min_limit_margin, value - lo, hi - value)
    for segment in segments:
        for sample in segment:
            max_dq = max(
                max_dq,
                *(abs(sample[kin.idx[name]] - previous[kin.idx[name]]) for name in ARM_JOINTS),
            )
            for name in ARM_JOINTS:
                lo, hi = kin.bounds(name)
                value = sample[kin.idx[name]]
                min_limit_margin = min(min_limit_margin, value - lo, hi - value)
            largest_delta = max(
                abs(sample[kin.idx[name]] - previous[kin.idx[name]]) for name in ARM_JOINTS
            )
            steps = max(
                1,
                math.ceil(
                    max(0.0, largest_delta - SETTLE_TOLERANCE_RAD) / (MOTOR_SPEED_RAD_S * DT_S)
                ),
            )
            sim_time += steps * DT_S
            previous = sample
    metrics = _table_metrics(kin, samples, base, torso)
    return {
        "sample_count": len(samples),
        "max_abs_dq_rad": max_dq,
        "min_joint_limit_margin_rad": min_limit_margin,
        "estimated_sim_time_s": sim_time,
        **metrics,
    }


def analyze(trace_path: Path) -> dict[str, object]:
    rows = _record_rows(trace_path)
    kin = Kinematics()
    start, goal, base = _read_start_goal(rows, kin)
    x_min, x_max, y_min, y_max = _table_footprint_bounds()
    result: dict[str, object] = {
        "trace": str(trace_path),
        "geometry": {
            "source": (
                "configs/robot.yaml; table values read from the worlds/factory.wbt Table "
                "instance and the Webots R2025a Table.proto"
            ),
            "footprint_centre_xy_m": list(TABLE_FOOTPRINT_CENTRE_XY_M),
            "footprint_size_xy_m": list(TABLE_FOOTPRINT_SIZE_XY_M),
            "footprint_x_m": [x_min, x_max],
            "footprint_y_m": [y_min, y_max],
            "top_z_m": TABLE_TOP_HEIGHT_M,
            "clearance_threshold_m": TABLE_CLEARANCE_REQUIRED_M,
            "hand_radius_m": 0.08,
        },
        "paths": {},
    }
    for torso in (0.35, 0.0):
        interp = _path(start, goal, MAX_JOINT_STEP_RAD)
        original = _independent_motor_path(kin, start, goal)
        result["paths"][f"torso_{torso:.2f}"] = {
            "synchronized_0.15_rad": _table_metrics(kin, interp, base, torso),
            "independent_0.5_rad_s_32ms": _table_metrics(kin, original, base, torso),
        }
    torso = 0.35
    start_for_safe_route = start.copy()
    goal_for_safe_route = goal.copy()
    start_for_safe_route[kin.idx[TORSO_JOINT]] = torso
    goal_for_safe_route[kin.idx[TORSO_JOINT]] = torso
    via = _safe_via(kin, start_for_safe_route, goal_for_safe_route, base)
    via_segments = [
        _path(start_for_safe_route, via, MAX_JOINT_STEP_RAD),
        _path(via, goal_for_safe_route, MAX_JOINT_STEP_RAD),
    ]
    interp_segment = [_path(start_for_safe_route, goal_for_safe_route, MAX_JOINT_STEP_RAD)]
    result["candidate_routes_torso_0.35"] = {
        "interpolated": _route_summary(kin, interp_segment, base, torso, start_for_safe_route),
        "safe_via": _route_summary(kin, via_segments, base, torso, start_for_safe_route),
        "safe_via_joint_target_rad": {name: float(via[kin.idx[name]]) for name in ARM_JOINTS},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace), indent=2))


if __name__ == "__main__":
    main()
