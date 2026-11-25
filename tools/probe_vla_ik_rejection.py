"""Replay an observed V2 TRAIN IK rejection within the original trust bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vlarobot.backends.webots.pose_solver import BoundedPoseSolver, quaternion_matrix
from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState, Vec3
from vlarobot.execution.vla_campaign import make_vla_adapter
from vlarobot.vla.action_adapter import as_quaternion, quaternion_product, rotation_delta_quaternion


def probe(directory: Path, stem: str) -> dict:
    """Compare frozen IK with a normalized residual solve as an offline feasibility witness."""
    meta = json.loads((directory / f"{stem}.meta.json").read_text())
    outcome = meta["outcome"]
    if meta["scenario"]["metadata"]["split"] != "train" or outcome["failure_reason"] != (
        "VLA IK pose residual exceeds configured tolerances"
    ):
        raise ValueError("a measured TRAIN IK residual rejection is required")
    records = [
        json.loads(line) for line in (directory / f"{stem}.latency.jsonl").read_text().splitlines()
    ]
    i = outcome["executed_actions"] - sum(len(r["actions"]) for r in records[:-1])
    values = np.asarray(records[-1]["actions"][i])
    with np.load(directory / f"{stem}.steps.npz", allow_pickle=False) as arrays:
        state = RobotState(
            dict(zip(arrays["joint_names"].tolist(), arrays["joint_pos"][-1], strict=True)),
            {},
            Pose2D(*arrays["base_pose"][-1]),
            Pose3D(Vec3(*arrays["ee_xyz"][-1]), as_quaternion(arrays["ee_quat"][-1])),
            float(arrays["gripper"][-1]),
        )
    position = state.ee_pose.position.as_array() + values[:3]
    orientation = as_quaternion(
        quaternion_product(
            rotation_delta_quaternion(values[3:6]), np.asarray(state.ee_pose.orientation)
        )
    )
    pose = Pose3D(Vec3(*position), orientation)
    rotation = quaternion_matrix(orientation)
    adapter = make_vla_adapter()
    solver = adapter.solver
    assert isinstance(solver, BoundedPoseSolver)
    present = solver.solve(pose, state)
    names = tuple(present.joints)
    kin = solver.kinematics
    seed = np.zeros(len(kin.chain.links))
    for name, value in state.joint_pos.items():
        if name in kin.idx:
            seed[kin.idx[name]] = value
    indices = [kin.idx[n] for n in names]
    joints = np.asarray([state.joint_pos[n] for n in names])
    bounds = np.asarray([kin.bounds(n) for n in names])
    lower = np.maximum(bounds[:, 0], joints - solver.max_joint_step_rad)
    upper = np.minimum(bounds[:, 1], joints + solver.max_joint_step_rad)

    def residual(q: np.ndarray) -> np.ndarray:
        x = seed.copy()
        x[indices] = q
        actual = kin.forward(x)
        return np.r_[
            (actual[:3, 3] - position) / adapter.position_tolerance_m,
            Rotation.from_matrix(rotation @ actual[:3, :3].T).as_rotvec()
            / adapter.orientation_tolerance_rad,
        ]

    def evaluate(q: np.ndarray) -> dict:
        r = residual(q)
        verdict = adapter.safety_monitor.validate(
            JointTarget(dict(zip(names, map(float, q), strict=True))), state
        )
        return {
            "max_joint_motion_rad": float(np.max(np.abs(q - joints))),
            "position_residual_m": float(np.linalg.norm(r[:3]) * adapter.position_tolerance_m),
            "orientation_residual_rad": float(
                np.linalg.norm(r[3:]) * adapter.orientation_tolerance_rad
            ),
            "within_existing_pose_gates": np.linalg.norm(r[:3]) <= 1 and np.linalg.norm(r[3:]) <= 1,
            "motor_precheck": verdict.accepted,
        }

    original = np.array([present.joints[n] for n in names])
    witnesses = []
    for start in (joints, original):
        result = least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper))
        witnesses.append(evaluate(result.x))
    return {
        "scope": "offline feasibility diagnostic on recorded TRAIN rejection",
        "episode": stem,
        "action_index_within_chunk": i,
        "observed_actions": outcome["executed_actions"],
        "requested_delta_m": float(np.linalg.norm(values[:3])),
        "requested_delta_rad": float(np.linalg.norm(values[3:6])),
        "existing_solver": evaluate(original),
        "gate_normalized_residual_witnesses": witnesses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("episode")
    args = parser.parse_args()
    print(json.dumps(probe(args.run, args.episode), indent=2, default=bool))
