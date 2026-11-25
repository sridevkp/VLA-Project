"""Run the Webots-free scripted-CAN continuity and placement audit."""

from __future__ import annotations

import math

import numpy as np

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
from vlarobot.core.types import Pose3D, Vec3
from vlarobot.execution.scripted_can import _world_to_base

BASE_WORLD = np.asarray((-0.53683, -3.70906, 0.0885438), dtype=float)
BASE_QUATERNION = (
    0.01121522834116144,
    -0.007522733160055142,
    -0.717650835270671,
    0.6962721492163582,
)
CAN_WORLD = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
TABLE_CENTER_XY = (-0.23882837, -4.8402928)
TABLE_SIZE_XY = (4.5, 1.5)
TABLE_TOP_M = 0.75
CAN_HALF_HEIGHT_M = 0.0611
CAN_RADIUS_M = 0.03175
HELD_TOOL_OFFSET_M = 0.02
GRID_STEP_M = 0.10
TABLE_EDGE_MARGIN_M = 0.10
OBJECT_CLEARANCE_M = 0.15
CAN_START_CLEARANCE_M = 0.15
SWEEP_STEP_RAD = 0.02
OBJECTS = (
    ("wrench", (-1.32774, -4.3478), 0.095, "Wrench.proto collision boxes; circumscribed XY radius"),
    (
        "hammer",
        (-0.544929, -4.5081),
        0.151,
        "Hammer.proto collision boxes/cylinder; circumscribed XY radius",
    ),
    ("apple", (-0.41, -4.44), 0.05, "Apple.proto bounding Sphere radius"),
    (
        "paint bucket",
        (-0.721679391155155, -5.178214147629225),
        0.20,
        "UNVERIFIED: conservative 0.20 m exclusion radius; remote PaintBucket.proto unavailable",
    ),
    (
        "screwdriver",
        (-0.07122996395380227, -5.200859419546002),
        0.115,
        "Screwdriver.proto collision cylinders; circumscribed XY radius",
    ),
)


def _rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _joint_delta(kinematics: Kinematics, first: np.ndarray, second: np.ndarray) -> list[float]:
    return [
        abs(float(second[kinematics.idx[name]] - first[kinematics.idx[name]]))
        for name in ARM_JOINTS
    ]


def _min_limit_margin(kinematics: Kinematics, q: np.ndarray) -> float:
    return min(
        min(
            q[kinematics.idx[name]] - kinematics.bounds(name)[0],
            kinematics.bounds(name)[1] - q[kinematics.idx[name]],
        )
        for name in ARM_JOINTS
    )


def _linear_ik_path(
    kinematics: Kinematics,
    start_q: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> list[np.ndarray]:
    start_position = kinematics.forward(start_q)[:3, 3].copy()
    count = max(2, int(np.linalg.norm(target - start_position) / 0.03))
    path = [start_q.copy()]
    q = start_q.copy()
    for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
        point = (1.0 - alpha) * start_position + alpha * target
        q, error, tilt = kinematics.solve(
            point,
            q,
            restarts=True,
            position_tolerance=tolerance,
            max_joint_step_rad=MAX_JOINT_STEP_RAD,
        )
        if error > tolerance or tilt > MAX_TILT_DEG:
            raise IKTrustRegionError(
                f"linear waypoint failed: error={error:.6f} m, tilt={tilt:.6f} degrees"
            )
        path.append(q.copy())
    return path


def _clearance_metrics(
    kinematics: Kinematics,
    path: list[np.ndarray],
    base_rotation: np.ndarray,
) -> dict[str, float]:
    max_tilt = 0.0
    min_margin = math.inf
    min_table = math.inf
    min_objects = {name: math.inf for name, _, _, _ in OBJECTS}
    max_step = 0.0
    for first, second in zip(path, path[1:], strict=False):
        joint_delta = max(_joint_delta(kinematics, first, second))
        max_step = max(max_step, joint_delta)
        samples = max(1, math.ceil(joint_delta / SWEEP_STEP_RAD))
        for index in range(1, samples + 1):
            q = first + (second - first) * index / samples
            transform = kinematics.forward(q)
            tool_position = transform[:3, 3]
            _, tilt = kinematics.score(tool_position, q)
            max_tilt = max(max_tilt, tilt)
            min_margin = min(min_margin, _min_limit_margin(kinematics, q))
            payload_world = (
                BASE_WORLD
                + base_rotation @ tool_position
                - base_rotation @ transform[:3, :3] @ np.asarray((0.0, 0.0, HELD_TOOL_OFFSET_M))
            )
            min_table = min(
                min_table,
                payload_world[2] - CAN_HALF_HEIGHT_M - TABLE_TOP_M,
            )
            for name, (object_x, object_y), object_radius, _ in OBJECTS:
                clearance = (
                    math.hypot(payload_world[0] - object_x, payload_world[1] - object_y)
                    - object_radius
                    - CAN_RADIUS_M
                )
                min_objects[name] = min(min_objects[name], clearance)
    return {
        "max_joint_step_rad": max_step,
        "min_joint_margin_rad": min_margin,
        "max_tool_tilt_deg": max_tilt,
        "min_payload_table_clearance_m": min_table,
        **{f"min_payload_{name}_clearance_m": value for name, value in min_objects.items()},
    }


def _candidate_scores(
    kinematics: Kinematics,
    lift_q: np.ndarray,
    base_pose: Pose3D,
    can_world: np.ndarray,
) -> tuple[int, list[tuple[float, float, np.ndarray]]]:
    x_min = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2 + TABLE_EDGE_MARGIN_M
    x_max = TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2 - TABLE_EDGE_MARGIN_M
    y_min = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2 + TABLE_EDGE_MARGIN_M
    y_max = TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2 - TABLE_EDGE_MARGIN_M
    can_z = TABLE_TOP_M + CAN_HALF_HEIGHT_M + 0.01
    candidates: list[tuple[float, float, np.ndarray]] = []
    geometric_count = 0
    for x in np.arange(x_min, x_max + 1e-9, GRID_STEP_M):
        for y in np.arange(y_min, y_max + 1e-9, GRID_STEP_M):
            if math.dist((x, y), can_world[:2]) < CAN_START_CLEARANCE_M:
                continue
            if any(
                math.dist((x, y), object_xy) - object_radius < OBJECT_CLEARANCE_M
                for _, object_xy, object_radius, _ in OBJECTS
            ):
                continue
            geometric_count += 1
            target_world = np.asarray((x, y, can_z), dtype=float)
            target_base = _world_to_base(target_world, base_pose)
            place_tip = target_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR), dtype=float)
            place_above = place_tip + np.asarray((0.0, 0.0, APPROACH_H), dtype=float)
            q_above, error, tilt = kinematics.solve(
                place_above,
                lift_q,
                restarts=False,
                position_tolerance=IK_TOL_TRANSIT,
            )
            if error > IK_TOL_TRANSIT or tilt > MAX_TILT_DEG:
                continue
            try:
                descent = _linear_ik_path(kinematics, q_above, place_tip, IK_TOL_GRASP)
            except IKTrustRegionError:
                continue
            transit = kinematics.interpolate_joint_path(lift_q, q_above, MAX_JOINT_STEP_RAD)
            route = [lift_q.copy(), *transit, *descent[1:], *reversed(descent[:-1])]
            margin = min(_min_limit_margin(kinematics, q) for q in route[1:])
            step = max(
                max(_joint_delta(kinematics, first, second))
                for first, second in zip(route, route[1:], strict=False)
            )
            candidates.append((margin, step, target_world))
    return geometric_count, sorted(candidates, key=lambda item: (-item[0], item[1]))


def run_audit() -> None:
    kinematics = Kinematics()
    kinematics.set_seed(0)
    base_pose = Pose3D(Vec3(*BASE_WORLD), BASE_QUATERNION)
    base_rotation = _rotation_matrix(BASE_QUATERNION)
    can_base = _world_to_base(CAN_WORLD, base_pose)
    place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose)
    pregrasp = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    grasp = can_base + np.asarray((0.0, 0.0, GRASP_Z))
    place_tip = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR))
    place_above = place_tip + np.asarray((0.0, 0.0, APPROACH_H))
    home_q = np.zeros(len(kinematics.chain.links), dtype=float)
    home_q[kinematics.idx[TORSO_JOINT]] = TORSO_HEIGHT
    home_q = kinematics.clip(home_q)

    q_pregrasp, _, _ = kinematics.solve(pregrasp, home_q, restarts=True, position_tolerance=IK_TOL)
    records: list[tuple[str, int, list[float], float]] = []
    records.append(("INIT", 1, [0.0] * len(ARM_JOINTS), _min_limit_margin(kinematics, home_q)))
    previous = home_q
    for index, q in enumerate(
        kinematics.interpolate_joint_path(home_q, q_pregrasp, MAX_JOINT_STEP_RAD), start=1
    ):
        records.append(
            (
                "PREGRASP",
                index,
                _joint_delta(kinematics, previous, q),
                _min_limit_margin(kinematics, q),
            )
        )
        previous = q
    q = q_pregrasp
    for state, target, tolerance in (
        ("DESCEND", grasp, IK_TOL_GRASP),
        ("LIFT", pregrasp, IK_TOL_GRASP),
        ("TRANSIT", place_above, IK_TOL_TRANSIT),
        ("PLACE_DESCEND", place_tip, IK_TOL_GRASP),
        ("RETREAT", place_above, IK_TOL_TRANSIT),
    ):
        path = _linear_ik_path(kinematics, q, target, tolerance)
        for index, (first, second) in enumerate(zip(path, path[1:], strict=False), start=1):
            records.append(
                (
                    state,
                    index,
                    _joint_delta(kinematics, first, second),
                    _min_limit_margin(kinematics, second),
                )
            )
        q = path[-1]
        if state == "DESCEND":
            records.append(("CLOSE", 1, [0.0] * len(ARM_JOINTS), _min_limit_margin(kinematics, q)))
        elif state == "PLACE_DESCEND":
            records.append(
                ("RELEASE", 1, [0.0] * len(ARM_JOINTS), _min_limit_margin(kinematics, q))
            )
    home_path = kinematics.interpolate_joint_path(q, home_q, MAX_JOINT_STEP_RAD)
    for index, (first, second) in enumerate(
        zip([q, *home_path[:-1]], home_path, strict=True), start=1
    ):
        records.append(
            (
                "HOME",
                index,
                _joint_delta(kinematics, first, second),
                _min_limit_margin(kinematics, second),
            )
        )

    print("Continuity: per-joint |dq| order is " + ", ".join(ARM_JOINTS))
    print("state,waypoint,dq_rad,max_abs_dq_rad,min_limit_margin_rad,flag")
    for state, index, deltas, margin in records:
        flag = max(deltas) > MAX_JOINT_STEP_RAD + 1e-9 or margin < 0.05
        print(
            f"{state},{index},"
            f"{' '.join(f'{value:.6f}' for value in deltas)},"
            f"{max(deltas):.6f},{margin:.6f},{'FLAG' if flag else ''}"
        )
    flagged = sum(
        max(deltas) > MAX_JOINT_STEP_RAD + 1e-9 or margin < 0.05 for _, _, deltas, margin in records
    )
    print(f"continuity_summary waypoints={len(records)} flagged={flagged}")

    lift_path = _linear_ik_path(kinematics, q_pregrasp, grasp, IK_TOL_GRASP)
    lift_path = _linear_ik_path(kinematics, lift_path[-1], pregrasp, IK_TOL_GRASP)
    lift_q = lift_path[-1]
    above_q, endpoint_error, endpoint_tilt = kinematics.solve(
        place_above, lift_q, restarts=True, position_tolerance=IK_TOL_TRANSIT
    )
    cartesian_path = _linear_ik_path(kinematics, lift_q, place_above, IK_TOL_TRANSIT)
    joint_path = [
        lift_q.copy(),
        *kinematics.interpolate_joint_path(lift_q, above_q, MAX_JOINT_STEP_RAD),
    ]
    print(
        f"transit_endpoint error_m={endpoint_error:.9f} tilt_deg={endpoint_tilt:.9f} "
        f"lift_to_place_above_max_dq_rad={max(_joint_delta(kinematics, lift_q, above_q)):.9f}"
    )
    print(
        "geometry_clearances status='UNVERIFIED (assumed radii)' "
        "paint_bucket_exclusion_radius_m=0.20"
    )
    for name, _, radius, source in OBJECTS:
        print(f"geometry_object name={name!r} exclusion_radius_m={radius:.6f} source={source}")
    for label, path in (
        ("cartesian_transit_sweep", cartesian_path),
        ("joint_transit_sweep", joint_path),
    ):
        metrics = _clearance_metrics(kinematics, path, base_rotation)
        print(
            f"{label} motion_max_dq_rad={metrics['max_joint_step_rad']:.9f} "
            f"min_joint_margin_rad={metrics['min_joint_margin_rad']:.9f} "
            f"max_tilt_deg={metrics['max_tool_tilt_deg']:.9f}"
        )
        for name, value in metrics.items():
            if "clearance" in name:
                print(f"{label} {name} status='UNVERIFIED (assumed radii)' value_m={value:.9f}")

    current_route_margin = min(_min_limit_margin(kinematics, point) for point in cartesian_path[1:])
    print(f"current_target_route_margin_rad={current_route_margin:.9g} target_applied=false")


if __name__ == "__main__":
    run_audit()
