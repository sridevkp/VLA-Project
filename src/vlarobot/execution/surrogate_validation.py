"""Read-only one-step nominal TCP/joint prediction errors from real successful TRAIN logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.webots.backend import _rotation_to_quaternion
from vlarobot.backends.webots.config import ARM_JOINTS
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState, Vec3
from vlarobot.world_model.kinematic import VelocityLimitedKinematics


def recorded_kinematic_errors(collection: Path, *, commanded_speed_rad_s: float) -> dict[str, Any]:
    """Evaluate causal accepted commands without fitting parameters or simulator rollouts."""
    selected = json.loads((collection / "accepted.json").read_text(encoding="utf-8"))["episodes"]
    kin = Kinematics()
    provenance = json.loads((collection / "provenance.json").read_text(encoding="utf-8"))
    if kin.registration_sha256 != provenance["registration_sha256"]:
        raise ValueError("surrogate validation requires the same fixed registered geometry")

    def forward(joints: dict[str, float]) -> Pose3D:
        vector = np.zeros(len(kin.chain.links))
        for name, value in joints.items():
            vector[kin.idx[name]] = value
        matrix = kin.forward(vector)
        return Pose3D(Vec3(*map(float, matrix[:3, 3])), _rotation_to_quaternion(matrix[:3, :3]))

    predictor = VelocityLimitedKinematics({n: commanded_speed_rad_s for n in ARM_JOINTS}, forward)
    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    for entry in selected:
        raw = Path(entry["raw_dir"])
        commands = [
            json.loads(line)
            for line in (raw / "episode-00.actions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if any(row["accepted"] is not True for row in commands):
            raise ValueError("surrogate validation requires explicit accepted commands")
        with np.load(raw / "episode-00.steps.npz", allow_pickle=False) as arrays:
            names = arrays["joint_names"].tolist()
            active = dict(zip(names, map(float, arrays["joint_pos"][0]), strict=True))
            index = 0
            family = entry["scene"]["metadata"]["family"]
            values = grouped.setdefault(family, [])
            for i, t in enumerate(arrays["t"][:-1]):
                while index < len(commands) and commands[index]["t"] <= t + 1e-9:
                    active.update(commands[index]["action"].get("q", {}))
                    index += 1
                joints = dict(zip(names, map(float, arrays["joint_pos"][i]), strict=True))
                state = RobotState(
                    joints, {}, Pose2D(0, 0, 0), forward(joints), float(arrays["gripper"][i])
                )
                prediction = predictor.step(
                    state,
                    JointTarget({n: active[n] for n in ARM_JOINTS}),
                    float(arrays["t"][i + 1] - t),
                )
                error = prediction.ee_pose.position.as_array() - arrays["ee_xyz"][i + 1]
                observed = arrays["ee_xyz"][i + 1] - arrays["ee_xyz"][i]
                joint_error = np.array(
                    [
                        prediction.joint_pos[n] - arrays["joint_pos"][i + 1, names.index(n)]
                        for n in ARM_JOINTS
                    ]
                )
                values.append(
                    (
                        float(error @ error),
                        float(np.linalg.norm(error)),
                        float(joint_error @ joint_error),
                        float(observed @ observed),
                    )
                )
    return {
        "status": "MEASURED read-only recorded TRAIN nominal TCP tracking prediction",
        "episodes": len(selected),
        "commanded_joint_speed_rad_s": commanded_speed_rad_s,
        "parameter_source": "frozen RobotIO sets arm velocity to 0.5 rad/s; no fitted coefficients",
        "quantity": "nominal registered TCP/joint tracking, not object/grasp/contact prediction",
        "brier_score": None,
        "probabilities_calibrated": False,
        "families": {
            family: {
                "one_step_samples": len(rows),
                "tool_position_rmse_m": float(np.sqrt(np.mean([r[0] for r in rows]))),
                "tool_position_mae_m": float(np.mean([r[1] for r in rows])),
                "joint_rmse_rad": float(np.sqrt(np.mean([r[2] for r in rows]) / len(ARM_JOINTS))),
                "zero_motion_tool_rmse_m": float(np.sqrt(np.mean([r[3] for r in rows]))),
            }
            for family, rows in grouped.items()
        },
    }
