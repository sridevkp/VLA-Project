"""Decompose the arm_7 FK-vs-GT (endPoint Solid) offset from existing JSONL traces.

Webots-free: reads recorded traces, rebuilds FK orientations/positions from the measured
joint angles with the project kinematics (URDF/ikpy, no simulator imports), and expresses

    u7 = GT(arm_7 endPoint Solid)      - FK(arm_7 link origin)
    u6 = GT(arm_6 endPoint Solid)      - FK(arm_6 link origin)
    u4 = GT(arm_4 endPoint Solid)      - FK(arm_4 link origin)

in (a) the world frame, (b) the robot base frame, and (c) the arm_7 link frame (and the
wrist_ft_tool "tool" frame), so "constant in the tool frame = geometry/offset error" vs
"constant in base/world = frame/origin error" vs "varies with pose = kinematic model
error" can be decided from data. Comparing u4/u6/u7 localizes *where* the FK error first
appears along the chain.

Usage: uv run python scripts/decompose_tool_offset.py [trace.jsonl ...]
       (defaults to every trace under runs/ that carries the GT fields)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import TORSO_HEIGHT, TORSO_JOINT
from vlarobot.backends.webots.kinematics import Kinematics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_GLOB = "runs/*.jsonl"

TOOL_KEY = "wrist_ft_tool_link_front_joint"
JOINT_KEYS = ("arm_4_joint", "arm_6_joint", "arm_7_joint", TOOL_KEY)
LINK_FRAMES = ("base", "world", "arm4_link", "arm6_link", "arm7_link", "tool")
VECTOR_KEYS = (
    *LINK_FRAMES,
    "u4_world",
    "u4_base",
    "u6_world",
    "u6_base",
    "arm6_arm7diff",
    "arm4_arm7diff",
)


def quaternion_to_matrix(quaternion: list[float]) -> np.ndarray:
    """Rotation matrix from a (x, y, z, w) quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def fk_snapshot(kin: Kinematics, row: dict) -> dict[str, np.ndarray] | None:
    """Base-frame FK positions and world-ready local rotations for relevant frames."""
    joints = row.get("joint_positions_rad") or {}
    q = np.zeros(len(kin.chain.links), dtype=float)
    for name, value in joints.items():
        index = kin.idx.get(name)
        if index is not None:
            q[index] = float(value)
    if kin.idx.get(TORSO_JOINT) is not None and TORSO_JOINT not in joints:
        q[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    try:
        transforms = kin.chain.forward_kinematics(q, full_kinematics=True)
    except Exception:
        return None
    positions: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}
    for key in JOINT_KEYS:
        index = kin.idx.get(key)
        if index is None:
            return None
        transform = np.asarray(transforms[index], dtype=float)
        positions[key] = transform[:3, 3]
        rotations[key] = transform[:3, :3]
    return {"positions": positions, "rotations": rotations}


def frame_stats(vectors: list[np.ndarray]) -> dict[str, object]:
    """Mean vector and per-component standard deviation of a list of vectors."""
    if not vectors:
        return {}
    stacked = np.vstack(vectors)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return {
        "n": len(vectors),
        "mean_xyz_m": mean.tolist(),
        "std_xyz_m": std.tolist(),
        "mean_norm_m": float(np.linalg.norm(mean)),
        "max_std_m": float(std.max()),
    }


def print_table(per_run: dict[str, dict[str, list[np.ndarray]]]) -> None:
    """Print the per-run/per-frame mean/std table."""
    header = (
        f"{'run':34s} {'frame':13s} {'n':>4s} {'mean_norm_mm':>12s} {'std_xyz_mm':>26s} mean_xyz_mm"
    )
    print(header)
    for run in sorted(per_run):
        for frame in VECTOR_KEYS:
            stats = frame_stats(per_run[run].get(frame, []))
            if not stats:
                continue
            mean_mm = [round(1000.0 * v, 2) for v in stats["mean_xyz_m"]]
            std_mm = [round(1000.0 * v, 3) for v in stats["std_xyz_m"]]
            print(
                f"{run:34s} {frame:13s} {stats['n']:4d} "
                f"{1000.0 * stats['mean_norm_m']:12.2f} "
                f"({std_mm[0]:6.2f},{std_mm[1]:6.2f},{std_mm[2]:6.2f}) "
                f"({mean_mm[0]:7.2f},{mean_mm[1]:7.2f},{mean_mm[2]:7.2f})"
            )
    print()


def print_global(per_run: dict[str, dict[str, list[np.ndarray]]]) -> None:
    """Print all-run aggregates per frame (the primary verdict input, n = all boundaries)."""
    merged: dict[str, list[np.ndarray]] = defaultdict(list)
    for run in sorted(per_run):
        for frame, vectors in per_run[run].items():
            merged[frame].extend(vectors)
    print(f"ALL RUNS aggregated (n={len(merged.get('world', []))} boundaries per frame):")
    for frame in VECTOR_KEYS:
        stats = frame_stats(merged.get(frame, []))
        if not stats:
            continue
        mean_mm = [round(1000.0 * v, 2) for v in stats["mean_xyz_m"]]
        std_mm = [round(1000.0 * v, 3) for v in stats["std_xyz_m"]]
        print(
            f"  {frame:13s} n={stats['n']:4d} mean_norm={1000.0 * stats['mean_norm_m']:8.2f}mm "
            f"std=({std_mm[0]:6.2f},{std_mm[1]:6.2f},{std_mm[2]:6.2f})mm "
            f"mean=({mean_mm[0]:7.2f},{mean_mm[1]:7.2f},{mean_mm[2]:7.2f})mm"
        )
    print()


def print_per_state(per_run_state: dict) -> None:
    """Print per state/waypoint arm7_link-frame residual stats per run."""
    print("Per state/waypoint, arm7_link frame residual (mm), per run:")
    for run in sorted(per_run_state):
        lines = []
        for state_phase in sorted(per_run_state[run]):
            stats = frame_stats(per_run_state[run][state_phase].get("arm7_link", []))
            if not stats:
                continue
            mean = stats["mean_xyz_m"]
            std = stats["std_xyz_m"]
            lines.append(
                f"{state_phase[0]}/{state_phase[1]}: "
                f"mean=({1000.0 * mean[0]:6.1f},{1000.0 * mean[1]:6.1f},{1000.0 * mean[2]:6.1f}) "
                f"std=({1000.0 * std[0]:5.2f},{1000.0 * std[1]:5.2f},{1000.0 * std[2]:5.2f})"
            )
        if lines:
            print(f"  {run}:\n    " + "\n    ".join(lines))
    print()


def collect(paths: list[Path], kin: Kinematics) -> tuple[dict, dict, int]:
    """Read traces and bucket residual vectors per run and per run/state."""
    per_run: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    per_run_state: dict[str, dict[tuple[str, str], dict[str, list[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    skipped = 0
    for path in paths:
        run = path.name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != "state_boundary":
                    continue
                gt = row.get("arm_7_joint_ground_truth_world_xyz_m")
                base_t = row.get("base_world_xyz_m")
                base_q = row.get("base_world_orientation_xyzw")
                links = row.get("supervisor_arm_link_world_xyz_m") or {}
                if not (
                    isinstance(gt, list)
                    and len(gt) == 3
                    and isinstance(base_t, list)
                    and len(base_t) == 3
                    and isinstance(base_q, list)
                    and len(base_q) == 4
                ):
                    skipped += 1
                    continue
                snapshot = fk_snapshot(kin, row)
                if snapshot is None:
                    skipped += 1
                    continue
                base_rot = quaternion_to_matrix(base_q)
                base_trans = np.array(base_t, dtype=float)
                positions = snapshot["positions"]
                rotations = snapshot["rotations"]
                gt_arm7 = np.array(gt, dtype=float)
                gt_arm6 = np.array(links.get("wrist", gt_arm7), dtype=float)
                gt_arm4 = np.array(links.get("elbow", gt_arm7), dtype=float)

                def to_world(
                    local_point: np.ndarray, _rot=base_rot, _trans=base_trans
                ) -> np.ndarray:
                    return _rot @ local_point + _trans

                # Residual vectors u = GT endPoint Solid - FK link origin (world frame).
                u7_world = gt_arm7 - to_world(positions["arm_7_joint"])
                u6_world = gt_arm6 - to_world(positions["arm_6_joint"])
                u4_world = gt_arm4 - to_world(positions["arm_4_joint"])

                frames: dict[str, np.ndarray] = {"world": u7_world, "base": base_rot.T @ u7_world}
                for frame in ("arm4_link", "arm6_link", "arm7_link", "tool"):
                    key = {
                        "arm4_link": "arm_4_joint",
                        "arm6_link": "arm_6_joint",
                        "arm7_link": "arm_7_joint",
                        "tool": TOOL_KEY,
                    }[frame]
                    r_world = base_rot @ rotations[key]
                    frames[frame] = r_world.T @ u7_world
                state_bucket = per_run_state[run][(row.get("state"), row.get("phase"))]
                run_bucket = per_run[run]
                for key, vector in frames.items():
                    run_bucket[key].append(vector)
                    state_bucket[key].append(vector)
                run_bucket["u4_world"].append(u4_world)
                run_bucket["u4_base"].append(base_rot.T @ u4_world)
                run_bucket["u6_world"].append(u6_world)
                run_bucket["u6_base"].append(base_rot.T @ u6_world)
                run_bucket["arm6_arm7diff"].append(u7_world - u6_world)
                run_bucket["arm4_arm7diff"].append(u7_world - u4_world)
    return per_run, per_run_state, skipped


def main(paths: list[Path]) -> int:
    kin = Kinematics()
    per_run, per_run_state, skipped = collect(paths, kin)
    print_table(per_run)
    print_global(per_run)
    print_per_state(per_run_state)
    print(f"skipped boundaries without complete FK/GT/base fields: {skipped}")
    return 0


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted(PROJECT_ROOT.glob(DEFAULT_TRACE_GLOB))
    raise SystemExit(main(args))
