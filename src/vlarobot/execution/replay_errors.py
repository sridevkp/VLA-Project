"""Read-only TRAIN action replay, quantity/horizon errors and source provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState
from vlarobot.world_model.kinematic import VelocityLimitedKinematics


def recorded_chunk_errors(
    collection: Path,
    forward: Callable[[dict[str, float]], Pose3D],
    *,
    joint_names: Sequence[str],
    speed_rad_s: float,
    horizons: Sequence[int],
    stride: int,
) -> dict[str, Any]:
    """Condition on recorded future commands, never future observations or fitted dynamics."""
    if not horizons or min(horizons) < 1 or stride < 1:
        raise ValueError("positive explicit horizons and stride required")
    entries = json.loads((collection / "accepted.json").read_text(encoding="utf-8"))["episodes"]
    predictor = VelocityLimitedKinematics({n: speed_rad_s for n in joint_names}, forward)
    groups: dict[str, dict[int, list[tuple[float, ...]]]] = {}
    sources = []
    for entry in entries:
        if entry["scene"]["metadata"]["split"] != "train":
            raise ValueError("recorded development only accepts TRAIN traces")
        raw = Path(entry["raw_dir"])
        meta = raw / "episode-00.meta.json"
        sources.append(
            {
                "attempt": entry["attempt"],
                "meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
            }
        )
        commands = [
            json.loads(line)
            for line in (raw / "episode-00.actions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if any(c["accepted"] is not True for c in commands):
            raise ValueError("replay requires accepted actual commands")
        with np.load(raw / "episode-00.steps.npz", allow_pickle=False) as a:
            names = a["joint_names"].tolist()
            active = dict(zip(names, map(float, a["joint_pos"][0]), strict=True))
            targets = []
            j = 0
            for t in a["t"]:
                while j < len(commands) and commands[j]["t"] <= t + 1e-9:
                    active.update(commands[j]["action"].get("q", {}))
                    j += 1
                targets.append({n: active[n] for n in joint_names})
            family = entry["scene"]["metadata"]["family"]
            by_horizon = groups.setdefault(family, {h: [] for h in horizons})
            can = a["gt_names"].tolist().index("CAN")
            for start in range(0, len(a["t"]) - 1, stride):
                for horizon in horizons:
                    end = start + horizon
                    if end >= len(a["t"]):
                        continue
                    q = dict(zip(names, map(float, a["joint_pos"][start]), strict=True))
                    robot = RobotState(
                        q, {}, Pose2D(0, 0, 0), forward(q), float(a["gripper"][start])
                    )
                    for i in range(start, end):
                        robot = predictor.step(
                            robot, JointTarget(targets[i]), float(a["t"][i + 1] - a["t"][i])
                        )
                    delta = robot.ee_pose.position.as_array() - a["ee_xyz"][end]
                    q_error = np.array(
                        [
                            robot.joint_pos[n] - a["joint_pos"][end, names.index(n)]
                            for n in joint_names
                        ]
                    )
                    still = a["ee_xyz"][start] - a["ee_xyz"][end]
                    obj = a["gt_xyz"][start, can] - a["gt_xyz"][end, can]
                    by_horizon[horizon].append(
                        (
                            float(delta @ delta),
                            float(np.linalg.norm(delta)),
                            float(q_error @ q_error),
                            float(still @ still),
                            float(obj @ obj),
                        )
                    )
    return {
        "status": "MEASURED offline recorded TRAIN replay; not autonomous task success",
        "episodes": len(entries),
        "sources": sources,
        "stride": stride,
        "horizons": list(horizons),
        "future_input": "recorded commanded joint targets only; observed future states are metrics",
        "object_quantity": "stationary CAN world-frame baseline, not contact/grasp prediction",
        "probabilities_calibrated": False,
        "brier_score": None,
        "test_accessed": False,
        "families": {
            family: {
                str(h): {
                    "windows": len(rows),
                    "tcp_rmse_m": float(np.sqrt(np.mean([r[0] for r in rows]))),
                    "tcp_mae_m": float(np.mean([r[1] for r in rows])),
                    "joint_rmse_rad": float(
                        np.sqrt(np.mean([r[2] for r in rows]) / len(joint_names))
                    ),
                    "stationary_tcp_rmse_m": float(np.sqrt(np.mean([r[3] for r in rows]))),
                    "stationary_can_world_rmse_m": float(np.sqrt(np.mean([r[4] for r in rows]))),
                }
                for h, rows in horizons_data.items()
            }
            for family, horizons_data in groups.items()
        },
    }
