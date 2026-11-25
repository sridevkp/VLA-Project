"""Teacher-forced SmolVLA probe on aligned, archived TRAIN demonstration frames."""

from __future__ import annotations

import argparse
import io
import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from vlarobot.core.types import Pose2D, Pose3D, RobotState, Vec3
from vlarobot.execution.vla_campaign import make_vla_adapter
from vlarobot.vla.action_adapter import as_quaternion, quaternion_product, rotation_delta_quaternion
from vlarobot.vla.model_service import PortableChunkModel


def observation_rows(dataset: Path, episode: int) -> list[dict]:
    """Read only matching image/state/action rows and retain their original indices."""
    import pyarrow.parquet as parquet

    rows = []
    for shard in sorted((dataset / "data").rglob("*.parquet")):
        table = parquet.read_table(shard, filters=[("episode_index", "=", episode)])
        rows.extend(table.to_pylist())
        if rows and table.num_rows == 0:
            break
    rows.sort(key=lambda row: row["frame_index"])
    if not rows or [row["frame_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("noncontiguous demonstration frames")
    return rows


def run(export: Path, data: Path, demo: Path, output: Path, index: int, release_only: bool) -> None:
    """Compare trained checkpoint outputs with recorded expert labels and bounded IK."""
    trace = demo / "policy.jsonl"
    phase_rows = [
        row
        for line in trace.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("event") == "state_boundary"
        and row.get("phase") == "enter"
    ]
    meta = json.loads((demo / "episode-00.meta.json").read_text(encoding="utf-8"))
    if meta["scenario"]["metadata"]["split"] != "train" or not meta["outcome"]["success"]:
        raise ValueError("probe requires a successful archived TRAIN demonstration")
    rows = observation_rows(data, index)
    adapter = make_vla_adapter()
    model = PortableChunkModel(export, cpu_threads=4)
    with np.load(demo / "episode-00.steps.npz", allow_pickle=False) as steps:
        timestamps = steps["t"]
        joint_names = steps["joint_names"].tolist()
        positions, orientations = steps["ee_xyz"], steps["ee_quat"]
        joints = steps["joint_pos"]
    if len(timestamps) != len(rows):
        raise ValueError("dataset and native recordings differ")
    if release_only:
        release = next(r["sim_time_s"] for r in phase_rows if r["state"] == "RELEASE")
        retreat = next(r["sim_time_s"] for r in phase_rows if r["state"] == "RETREAT")
        frame = next(
            i
            for i, (t, row) in enumerate(zip(timestamps, rows, strict=True))
            if release <= t <= retreat and row["action"][6] >= np.float32(0.045)
        )
        phase_rows = [{"state": "RELEASE_FIRST_OPEN", "sim_time_s": timestamps[frame]}]

    def ik(values: np.ndarray, state: RobotState) -> dict:
        try:
            result = adapter.decode(values, state)
            return {
                "accepted": True,
                "joints": result.joint_target.q,
                "max_motion_rad": max(
                    abs(v - state.joint_pos[k]) for k, v in result.joint_target.q.items()
                ),
                "position_residual_m": result.position_error_m,
                "orientation_residual_rad": result.orientation_error_rad,
            }
        except (ValueError, RuntimeError) as error:
            return {"accepted": False, "reason": str(error)}

    instruction = "Pick up the can and place it at the target location."
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        for phase_record in phase_rows:
            phase = phase_record["state"]
            frame = int(np.argmin(np.abs(timestamps - phase_record["sim_time_s"])))
            row = rows[frame]
            expected = np.r_[joints[frame, :7], row["observation.state"][-1]]
            np.testing.assert_allclose(row["observation.state"], expected, atol=1e-7)
            img = row["observation.images.camera1"]["bytes"]
            rgb = np.asarray(Image.open(io.BytesIO(img)).convert("RGB"))
            model.reset(int(meta["seed"]))
            predicted = model.predict(
                np.asarray(row["observation.state"], np.float32), rgb, instruction
            )
            expert = np.array([item["action"] for item in rows[frame : frame + len(predicted)]])
            matched = predicted[: len(expert)].astype(float)
            position_error = np.linalg.norm(matched[:, :3] - expert[:, :3], axis=1)
            rotation_error = []
            for p, e in zip(matched[:, 3:6], expert[:, 3:6], strict=True):
                q = rotation_delta_quaternion(p)
                r = rotation_delta_quaternion(e)
                d = quaternion_product(q, np.r_[-r[:3], r[3]])
                rotation_error.append(2 * math.atan2(np.linalg.norm(d[:3]), abs(d[3])))
            state = RobotState(
                dict(zip(joint_names, joints[frame].tolist(), strict=True)),
                {},
                Pose2D(0, 0, 0),
                Pose3D(Vec3(*positions[frame]), as_quaternion(orientations[frame])),
                float(row["observation.state"][-1]),
            )
            finite_first = predicted[0].copy()
            finite_first[6] = np.clip(finite_first[6], 0, 0.045)
            report = {
                "scope": "teacher-forced TRAIN observation; no backend",
                "phase": phase,
                "frame": frame,
                "episode_index": index,
                "scene": meta["scenario"]["name"],
                "seed": meta["seed"],
                "image_sha256": sha256(img).hexdigest(),
                "model": model.metadata,
                "aligned_actions": len(expert),
                "translation_mae_m": float(np.mean(position_error)),
                "orientation_mae_rad": float(np.mean(rotation_error)),
                "learned_gripper_outside": int(
                    np.count_nonzero(
                        (predicted[:, 6] < 0.0) | (predicted[:, 6] > np.float32(0.045))
                    )
                ),
                "learned_first_ik": ik(predicted[0], state),
                "bounded_first_ik_for_diagnosis": ik(finite_first, state),
                "expert_first_ik": ik(expert[0], state),
                "expert_actions": expert.tolist(),
                "learned_actions": predicted.tolist(),
            }
            stream.write(json.dumps(report, allow_nan=False) + "\n")
            print(phase, frame, report["learned_gripper_outside"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "data", "demo", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--release-only", action="store_true")
    args = parser.parse_args()
    run(args.export, args.data, args.demo, args.output, args.index, args.release_only)
