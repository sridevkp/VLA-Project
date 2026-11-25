"""Webots-free analysis of posture, torso and base settings; never changes robot settings."""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np
from audit_can_path import (
    BASE_QUATERNION,
    BASE_WORLD,
    CAN_WORLD,
    GRID_STEP_M,
    _candidate_scores,
    _joint_delta,
    _min_limit_margin,
)

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
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import IKTrustRegionError, Kinematics
from vlarobot.core.types import Pose3D, Vec3
from vlarobot.execution.scripted_can import _world_to_base

RESTARTS = 100
TORSO_VALUES = tuple(round(value * 0.05, 2) for value in range(8))
BASE_OFFSETS = tuple(round(value * 0.05, 2) for value in range(-3, 4))
YAW_OFFSETS_DEG = (-10, -5, 0, 5, 10)
FLOW_STATES = ("PREGRASP", "DESCEND", "LIFT", "TRANSIT", "PLACE_DESCEND", "RETREAT", "HOME")


@dataclass
class FlowResult:
    setting: tuple[float, float, float, float]
    complete: bool
    failure: str
    min_margin: float
    max_dq: float
    max_ik_error: float
    max_tilt: float
    selected_pregrasp: np.ndarray
    lift_q: np.ndarray
    profile: list[tuple[str, int, float, float, float, float, np.ndarray]]
    base_pose: Pose3D


def _qmul(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _base_pose(dx: float, dy: float, dyaw_deg: float) -> Pose3D:
    yaw = math.radians(dyaw_deg) / 2.0
    offset = (0.0, 0.0, math.sin(yaw), math.cos(yaw))
    return Pose3D(
        Vec3(float(BASE_WORLD[0] + dx), float(BASE_WORLD[1] + dy), float(BASE_WORLD[2])),
        _qmul(offset, BASE_QUATERNION),
    )


def _home(kinematics: Kinematics, torso: float) -> np.ndarray:
    home = np.zeros(len(kinematics.chain.links), dtype=float)
    home[kinematics.idx[TORSO_JOINT]] = torso
    return kinematics.clip(home)


def _select_pregrasp(
    kinematics: Kinematics,
    target: np.ndarray,
    home_q: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    feasible: list[tuple[float, float, np.ndarray, float, float]] = []
    for _ in range(RESTARTS):
        initial = home_q.copy()
        for name in ARM_JOINTS:
            lower, upper = kinematics.bounds(name)
            initial[kinematics.idx[name]] = rng.uniform(lower, upper)
        q = kinematics.chain.inverse_kinematics(
            target,
            target_orientation=[0, 0, 1],
            orientation_mode="Z",
            initial_position=initial,
        )
        error, tilt = kinematics.score(target, q)
        if error <= IK_TOL and tilt <= MAX_TILT_DEG:
            margin = _min_limit_margin(kinematics, q)
            distance = float(
                np.linalg.norm(
                    [q[kinematics.idx[name]] - home_q[kinematics.idx[name]] for name in ARM_JOINTS]
                )
            )
            feasible.append((margin, -distance, q, error, tilt))
    if not feasible:
        raise RuntimeError(f"no feasible pregrasp among {RESTARTS} deterministic restarts")
    margin, _, q, error, tilt = max(feasible, key=lambda candidate: (candidate[0], candidate[1]))
    return q, error, tilt


def _linear_path(
    kinematics: Kinematics,
    start: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    start_position = kinematics.forward(start)[:3, 3].copy()
    count = max(2, int(np.linalg.norm(target - start_position) / 0.03))
    path = [start.copy()]
    metrics = [(0.0, 0.0)]
    q = start.copy()
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
                f"error={error:.6f} m exceeds {tolerance:.6f} m or tilt={tilt:.6f} deg"
            )
        path.append(q.copy())
        metrics.append((error, tilt))
    return path, metrics


def _evaluate(setting: tuple[float, float, float, float], mode: str, seed: int) -> FlowResult:
    torso, dx, dy, dyaw = setting
    kinematics = Kinematics()
    kinematics.rng = np.random.default_rng(seed)
    base_pose = _base_pose(dx, dy, dyaw)
    home_q = _home(kinematics, torso)
    can_base = _world_to_base(CAN_WORLD, base_pose)
    grasp = can_base + np.asarray((0.0, 0.0, GRASP_Z))
    pregrasp = grasp + np.asarray((0.0, 0.0, APPROACH_H))
    place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose)
    place_tip = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR))
    place_above = place_tip + np.asarray((0.0, 0.0, APPROACH_H))

    q_pre = home_q.copy()
    pre_error = pre_tilt = math.inf
    try:
        if mode == "current":
            q_pre, pre_error, pre_tilt = kinematics.solve(
                pregrasp, home_q, restarts=True, position_tolerance=IK_TOL
            )
        else:
            q_pre, pre_error, pre_tilt = _select_pregrasp(kinematics, pregrasp, home_q, seed)
        states: list[tuple[str, list[np.ndarray], list[tuple[float, float]]]] = []
        states.append(
            (
                "PREGRASP",
                kinematics.interpolate_joint_path(home_q, q_pre, MAX_JOINT_STEP_RAD),
                [(pre_error, pre_tilt)]
                * len(kinematics.interpolate_joint_path(home_q, q_pre, MAX_JOINT_STEP_RAD)),
            )
        )
        q = q_pre
        for state, target, tolerance in (
            ("DESCEND", grasp, IK_TOL_GRASP),
            ("LIFT", pregrasp, IK_TOL_GRASP),
            ("TRANSIT", place_above, IK_TOL_TRANSIT),
            ("PLACE_DESCEND", place_tip, IK_TOL_GRASP),
            ("RETREAT", place_above, IK_TOL_TRANSIT),
        ):
            path, metrics = _linear_path(kinematics, q, target, tolerance)
            states.append((state, path[1:], metrics[1:]))
            q = path[-1]
        lift_q = states[2][1][-1]
        home_path = kinematics.interpolate_joint_path(q, home_q, MAX_JOINT_STEP_RAD)
        states.append(("HOME", home_path, [(0.0, 0.0)] * len(home_path)))
    except (RuntimeError, ValueError) as error:
        return FlowResult(
            setting,
            False,
            str(error),
            math.nan,
            math.nan,
            pre_error,
            pre_tilt,
            q_pre,
            q_pre,
            [],
            base_pose,
        )

    records: list[tuple[str, int, float, float, float, float, np.ndarray]] = [
        ("INIT", 1, _min_limit_margin(kinematics, home_q), 0.0, 0.0, 0.0, home_q.copy())
    ]
    previous = home_q
    max_error = 0.0
    max_tilt = 0.0
    for state, path, metrics in states:
        for index, (q, (error, tilt)) in enumerate(zip(path, metrics, strict=True), start=1):
            delta = max(_joint_delta(kinematics, previous, q))
            margin = _min_limit_margin(kinematics, q)
            records.append((state, index, margin, delta, error, tilt, q.copy()))
            previous = q
            max_error = max(max_error, error)
            max_tilt = max(max_tilt, tilt)
    return FlowResult(
        setting,
        True,
        "",
        min(record[2] for record in records),
        max(record[3] for record in records),
        max_error,
        max_tilt,
        q_pre,
        lift_q,
        records,
        base_pose,
    )


def _show_top(title: str, results: list[FlowResult]) -> list[FlowResult]:
    complete = sorted(
        (result for result in results if result.complete),
        key=lambda result: (-result.min_margin, result.max_dq, result.max_ik_error),
    )
    print(f"\n{title}: completed={len(complete)}/{len(results)} target_min_margin_rad=0.05")
    for rank, result in enumerate(complete[:3], start=1):
        torso, dx, dy, dyaw = result.setting
        print(
            f"rank={rank} torso_m={torso:.2f} base_offset_m=({dx:.2f},{dy:.2f}) "
            f"dyaw_deg={dyaw:.0f} min_margin_rad={result.min_margin:.6f} "
            f"max_abs_dq_rad={result.max_dq:.6f} max_ik_error_m={result.max_ik_error:.6f} "
            f"max_tilt_deg={result.max_tilt:.6f} meets_0.05={result.min_margin >= 0.05}"
        )
    failures = [result for result in results if not result.complete]
    for result in failures:
        print(f"failed_setting={result.setting} reason={result.failure}")
    return complete


def _show_margin_profile(result: FlowResult) -> None:
    kinematics = Kinematics()
    print("\nbest_case_joint_margin_profile: per joint limit distance (rad)")
    print("state,waypoint,arm_6_margin_rad,arm_7_margin_rad,all_arm_min_margin_rad,max_abs_dq_rad")
    for state, index, margin, delta, _, _, q in result.profile:
        arm6_margin = _joint_margin(kinematics, "arm_6_joint", q)
        arm7_margin = _joint_margin(kinematics, "arm_7_joint", q)
        print(f"{state},{index},{arm6_margin:.6f},{arm7_margin:.6f},{margin:.6f},{delta:.6f}")


def _joint_margin(kinematics: Kinematics, name: str, q: np.ndarray) -> float:
    lower, upper = kinematics.bounds(name)
    value = float(q[kinematics.idx[name]])
    return min(value - lower, upper - value)


def run_study() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-base-sweep",
        action="store_true",
        help="Run studies a-c only; the full d grid is computationally expensive.",
    )
    args = parser.parse_args()
    print("study=a_current_behavior", flush=True)
    results_a = [_evaluate((0.35, 0.0, 0.0, 0.0), "current", seed=0)]
    print("study=b_selected_pregrasp", flush=True)
    results_b = [_evaluate((0.35, 0.0, 0.0, 0.0), "selected", seed=1000)]
    results_c = []
    for index, torso in enumerate(TORSO_VALUES):
        results_c.append(_evaluate((torso, 0.0, 0.0, 0.0), "selected", seed=2000 + index))
        print(f"torso_sweep_progress={index + 1}/{len(TORSO_VALUES)}", flush=True)

    _show_top("a_current_behavior", results_a)
    _show_top("b_selected_pregrasp", results_b)
    best_c = _show_top("c_selected_pregrasp_torso_sweep", results_c)
    if args.skip_base_sweep:
        print("d_posture_torso_base_sweep=NOT RUN (--skip-base-sweep)")
        if best_c:
            _show_margin_profile(best_c[0])
        return

    count_d = len(TORSO_VALUES) * len(BASE_OFFSETS) ** 2 * len(YAW_OFFSETS_DEG)
    print(f"base_pose_sweep_settings={count_d} restarts_per_setting={RESTARTS}")
    results_d = []
    for index, (torso, dx, dy, dyaw) in enumerate(
        itertools.product(TORSO_VALUES, BASE_OFFSETS, BASE_OFFSETS, YAW_OFFSETS_DEG)
    ):
        results_d.append(_evaluate((torso, dx, dy, float(dyaw)), "selected", seed=3000 + index))
        if (index + 1) % 50 == 0:
            print(f"base_pose_sweep_progress={index + 1}/{count_d}", flush=True)

    best_d = _show_top("d_posture_torso_base_sweep", results_d)
    if not best_d:
        print("candidate_search skipped: no complete full-flow base setting")
        return

    best = best_d[0]
    _show_margin_profile(best)
    kinematics = Kinematics()
    kinematics.rng = np.random.default_rng(9000)
    geometric, candidates = _candidate_scores(kinematics, best.lift_q, best.base_pose, CAN_WORLD)
    print(
        f"\ngeometry_screen_candidates setting={best.setting} geometric={geometric} "
        f"ik={len(candidates)} status='UNVERIFIED (assumed radii)' "
        f"paint_bucket_exclusion_radius_m=0.20 grid_step_m={GRID_STEP_M:.2f}"
    )
    from audit_can_path import OBJECTS

    for name, _, radius, source in OBJECTS:
        print(f"geometry_object name={name!r} exclusion_radius_m={radius:.6f} source={source}")
    for rank, (margin, max_step, world) in enumerate(candidates[:3], start=1):
        base = _world_to_base(world, best.base_pose)
        print(
            f"candidate_rank={rank} route_min_margin_rad={margin:.6f} "
            f"max_abs_dq_rad={max_step:.6f} world_m={world.tolist()} base_m={base.tolist()} "
            "applied=false"
        )


if __name__ == "__main__":
    run_study()
