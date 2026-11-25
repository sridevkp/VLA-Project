"""Reconcile the frozen VLA TRAIN failure paths without loading a policy or simulator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np


def audit(archive: Path, manifest: Path) -> dict:
    """Check recorded gripper proposals and reconstruct the original HOME exits."""
    pins = json.loads(manifest.read_text(encoding="utf-8"))
    if not pins:
        raise ValueError("empty archive manifest")
    for name, pin in pins.items():
        path = archive / name
        if (
            path.stat().st_size != pin["bytes"]
            or sha256(path.read_bytes()).hexdigest() != pin["sha256"]
        ):
            raise ValueError(f"archived input failed verification: {name}")

    counts: Counter[str] = Counter()
    commands = []
    home_exits = []
    for meta_path in sorted(archive.glob("episode-*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["scenario"]["metadata"]["split"] != "train":
            raise ValueError("audit requires the historical TRAIN campaign")
        stem = meta_path.name.removesuffix(".meta.json")
        lines = (archive / f"{stem}.latency.jsonl").read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError("expected exactly one historical chunk")
        proposal = np.asarray(json.loads(lines[0])["actions"], dtype=np.float64)
        if proposal.shape != (50, 7) or not np.all(np.isfinite(proposal)):
            raise ValueError("invalid archived action matrix")
        opening = proposal[:, 6]
        commands.extend(opening.tolist())
        # The historical adapter canonicalizes only the exact float32 representation.
        admitted_opening = np.where(opening == float(np.float32(0.045)), 0.045, opening)
        invalid = np.flatnonzero((admitted_opening < 0.0) | (admitted_opening > 0.045))
        counts["episodes"] += 1
        counts["successes"] += int(meta["outcome"]["success"])
        counts["out_of_bounds_targets"] += len(invalid)
        counts["chunks_with_out_of_bounds_target"] += bool(len(invalid))
        counts[meta["outcome"]["failure_state"]] += 1
        counts["accepted_actions"] += meta["outcome"]["executed_actions"]

        if meta["outcome"]["home_reached"]:
            with np.load(archive / f"{stem}.steps.npz", allow_pickle=False) as steps:
                names = steps["joint_names"].tolist()
                indices = [names.index(f"arm_{index}_joint") for index in range(1, 8)]
                joints = steps["joint_pos"][:, indices]
                err = np.max(np.abs(joints - joints[0]), axis=1)
                left = np.flatnonzero(err > 0.02)
                if not len(left):
                    raise ValueError(f"missing recorded HOME departure: {stem}")
                returned = np.flatnonzero(err[left[0] + 1 :] <= 0.02)
                if not len(returned):
                    raise ValueError(f"missing recorded HOME return: {stem}")
                home = int(left[0] + 1 + returned[0])
                if home != meta["outcome"]["executed_actions"]:
                    raise ValueError(f"HOME predicate does not reconcile: {stem}")
                obj = steps["gt_names"].tolist().index("CAN")
                rise = float(np.max(steps["gt_xyz"][:, obj, 2] - steps["gt_xyz"][0, obj, 2]))
                home_exits.append(
                    {
                        "episode": stem,
                        "steps": home,
                        "depart_error_rad": float(err[left[0]]),
                        "return_error_rad": float(err[home]),
                        "can_rise_m": rise,
                        "first_invalid_proposal_index": int(invalid[0]) if len(invalid) else None,
                    }
                )
    if counts["episodes"] != 24:
        raise ValueError("expected the frozen 24-trial TRAIN evidence")
    return {
        "scope": "historical TRAIN read-only audit",
        "verified_archive_files": len(pins),
        "counts": dict(counts),
        "opening_range_m": [min(commands), max(commands)],
        "home_exits": home_exits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.manifest), indent=2))
