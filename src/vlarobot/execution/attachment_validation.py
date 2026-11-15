"""Oracle-initialized conditional carry prediction against recorded TRAIN CAN poses."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState, Vec3
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.linearized_ik import pose_error
from vlarobot.world_model.rollout import compose, quaternion_product, rotate


def conditional_attachment_errors(
    collection: Path,
    forward: Callable[[dict[str, float]], Pose3D],
    matrix_to_quaternion: Callable[[np.ndarray], tuple[float, float, float, float]],
    *,
    joint_names: Sequence[str],
    speed_rad_s: float,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """No grasp inference: explicitly condition on TRANSIT entry of audited successful carries.

    Initialize object/tool attachment and full base transform from the initial oracle row.
    Future joint commands are replay inputs; future CAN/robot observations are metrics only.
    Never use this oracle-initialized diagnostic as learned-policy input or task success.
    """
    predictor = VelocityLimitedKinematics({n: speed_rad_s for n in joint_names}, forward)
    grouped = {}
    accepted = json.loads((collection / "accepted.json").read_text(encoding="utf-8"))["episodes"]
    for entry in accepted:
        if entry["scene"]["metadata"]["split"] != "train":
            raise ValueError("TRAIN diagnostic only")
        raw = Path(entry["raw_dir"])
        rows = [
            json.loads(s) for s in (raw / "policy.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        startrow = next(r for r in rows if r["state"] == "TRANSIT" and r["phase"] == "enter")
        stoprow = next(r for r in rows if r["state"] == "PLACE_DESCEND" and r["phase"] == "enter")
        base = Pose3D(
            Vec3(*startrow["base_world_xyz_m"]),
            (
                float(startrow["base_world_orientation_xyzw"][0]),
                float(startrow["base_world_orientation_xyzw"][1]),
                float(startrow["base_world_orientation_xyzw"][2]),
                float(startrow["base_world_orientation_xyzw"][3]),
            ),
        )
        initial_can = Pose3D(
            Vec3(*startrow["can_base_xyz_m"]),
            matrix_to_quaternion(np.asarray(startrow["can_base_orientation_matrix"]).reshape(3, 3)),
        )
        q0 = dict(startrow["measured_joint_positions_rad"])
        tool = forward(q0)
        inverse_q = np.r_[-np.asarray(tool.orientation)[:3], tool.orientation[3]]
        relative_q = quaternion_product(inverse_q, np.asarray(initial_can.orientation))
        relative = Pose3D(
            Vec3(
                *map(
                    float,
                    rotate(inverse_q, initial_can.position.as_array() - tool.position.as_array()),
                )
            ),
            (
                float(relative_q[0]),
                float(relative_q[1]),
                float(relative_q[2]),
                float(relative_q[3]),
            ),
        )
        commands = [
            json.loads(s)
            for s in (raw / "episode-00.actions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        family = entry["scene"]["metadata"]["family"]
        group = grouped.setdefault(family, {h: [] for h in horizons})
        with np.load(raw / "episode-00.steps.npz", allow_pickle=False) as a:
            names = a["joint_names"].tolist()
            start = int(np.argmin(np.abs(a["t"] - startrow["sim_time_s"])))
            if abs(a["t"][start] - startrow["sim_time_s"]) > 1e-6:
                raise ValueError("oracle boundary/frame alignment missing")
            active = dict(zip(names, map(float, a["joint_pos"][0]), strict=True))
            targets = []
            j = 0
            for t in a["t"]:
                while j < len(commands) and commands[j]["t"] <= t + 1e-9:
                    active.update(commands[j]["action"].get("q", {}))
                    j += 1
                targets.append({n: active[n] for n in joint_names})
            can = a["gt_names"].tolist().index("CAN")
            for h in horizons:
                end = start + h
                if end >= len(a["t"]) or a["t"][end] >= stoprow["sim_time_s"]:
                    continue
                robot = RobotState(q0, {}, Pose2D(0, 0, 0), tool, float(a["gripper"][start]), True)
                for i in range(start, end):
                    robot = predictor.step(
                        robot, JointTarget(targets[i]), float(a["t"][i + 1] - a["t"][i])
                    )
                prediction = compose(base, compose(robot.ee_pose, relative))
                observed = Pose3D(
                    Vec3(*map(float, a["gt_xyz"][end, can])),
                    (
                        float(a["gt_quat"][end, can, 0]),
                        float(a["gt_quat"][end, can, 1]),
                        float(a["gt_quat"][end, can, 2]),
                        float(a["gt_quat"][end, can, 3]),
                    ),
                )
                delta = pose_error(prediction, observed)
                static = a["gt_xyz"][start, can] - a["gt_xyz"][end, can]
                group[h].append(
                    (
                        float(np.linalg.norm(delta[:3])),
                        float(np.linalg.norm(delta[3:])),
                        float(static @ static),
                    )
                )
    return {
        "status": "MEASURED oracle-initialized conditional rigid carry diagnostic",
        "not_grasp_success_prediction": True,
        "not_placement_prediction": True,
        "episodes": len(accepted),
        "initialization": "TRAIN TRANSIT-entry oracle poses, conditioned on successful carry",
        "future_input": "recorded commanded joints only",
        "base_motion_model": "fixed initial base pose; drift contributes to error",
        "test_accessed": False,
        "brier_score": None,
        "probabilities_calibrated": False,
        "families": {
            f: {
                str(h): {
                    "windows": len(rows),
                    "can_world_position_rmse_m": float(np.sqrt(np.mean([r[0] ** 2 for r in rows])))
                    if rows
                    else None,
                    "can_world_position_mae_m": float(np.mean([r[0] for r in rows]))
                    if rows
                    else None,
                    "can_world_orientation_rmse_rad": float(
                        np.sqrt(np.mean([r[1] ** 2 for r in rows]))
                    )
                    if rows
                    else None,
                    "stationary_can_world_rmse_m": float(np.sqrt(np.mean([r[2] for r in rows])))
                    if rows
                    else None,
                }
                for h, rows in groups.items()
            }
            for f, groups in grouped.items()
        },
    }
