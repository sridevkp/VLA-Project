"""Postprocess physically sampled chunks without conflating closure/carry/release with success."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def audit_candidates(root: Path) -> dict[str, Any]:
    cases = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = folder / "episode-00.meta.json"
        if not meta.exists():
            continue
        outcome = json.loads(meta.read_text(encoding="utf-8"))["outcome"]
        if outcome.get("not_a_demonstration") is not True or outcome.get("success") is not False:
            raise ValueError("diagnostic trajectories must not be successful demonstrations")
        row = {"kind": folder.name, "outcome": outcome}
        predictions = folder / "episode-00.predicted.jsonl"
        if predictions.exists():
            expected = [json.loads(s) for s in predictions.read_text(encoding="utf-8").splitlines()]
            with np.load(folder / "episode-00.steps.npz", allow_pickle=False) as a:
                errors = []
                for prediction in expected:
                    index = int(np.argmin(np.abs(a["t"] - prediction["t"])))
                    if abs(a["t"][index] - prediction["t"]) > 1e-8:
                        raise ValueError("unaligned prediction/observation")
                    errors.append(prediction["predicted"]["gripper_m"] - float(a["gripper"][index]))
                if errors:
                    row["gripper_feedback_rmse_m"] = float(np.sqrt(np.mean(np.square(errors))))
                    row["gripper_feedback_mae_m"] = float(np.mean(np.abs(errors)))
                can = a["gt_names"].tolist().index("CAN")
                row["max_can_rise_from_episode_start_m"] = float(
                    a["gt_xyz"][:, can, 2].max() - a["gt_xyz"][0, can, 2]
                )
                row["final_can_world_xyz_m"] = a["gt_xyz"][-1, can].tolist()
        row["files_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.glob("episode-00.*")
            if p.is_file()
        }
        cases.append(row)
    return {
        "status": "MEASURED independently sampled physical VALIDATION chunks",
        "completed_cases": len(cases),
        "cases": cases,
        "test_accessed": False,
        "training_admission": False,
        "brier_score": None,
        "limitations": [
            "one sample per staged condition; not a task success estimate",
            "oracle object initialization; future observations only for scoring",
            "conditional carry, closure, release and placement are separate quantities",
            "no grasp probability calibration or collision-safety certificate",
        ],
    }
