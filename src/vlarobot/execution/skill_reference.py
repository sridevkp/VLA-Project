"""CPU demonstration reference fitted on TRAIN scene groups."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.skill_demonstrations import partition_demonstrations
from vlarobot.execution.skills import Skill


def train_reference(dataset: Path, output: Path) -> dict[str, Any]:
    """Fit demonstrated skill preferences from decision-time geometry and aperture."""
    if output.exists():
        raise FileExistsError(output)
    raw = dataset.read_bytes()
    payload = json.loads(raw)
    if payload["split"] != "train":
        raise ValueError("Reference fitting requires TRAIN decisions")
    rows = payload["decisions"]
    partition = partition_demonstrations(rows)
    if partition != payload["partition"]:
        raise ValueError("TRAIN partition mismatch")
    skills = tuple(skill.value for skill in Skill)
    features = []
    for row in rows:
        before = row["observed_before"]
        tool = np.asarray(before["tool_base_m"])
        can = np.asarray(before["oracle_can_base_m"])
        features.append(np.r_[tool, can - tool, before["gripper_opening_m"]])
    x = np.asarray(features)
    labels = np.asarray([skills.index(row["skill"]) for row in rows])
    fit = np.asarray([row["scene_group"] in partition["fit_groups"] for row in rows])
    mean, scale = x[fit].mean(axis=0), x[fit].std(axis=0)
    scale[scale == 0] = 1.0
    z = np.c_[np.ones(len(x)), (x - mean) / scale]
    target = np.eye(len(skills))[labels]
    design = z[fit]
    weights = np.linalg.solve(design.T @ design + np.eye(z.shape[1]), design.T @ target[fit])
    predicted = np.argmax(z @ weights, axis=1)
    confusion = np.zeros((len(skills), len(skills)), dtype=int)
    np.add.at(confusion, (labels[~fit], predicted[~fit]), 1)
    result = {
        "identity": "v4-demonstration-reference-ridge-v1",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "fit_n": int(fit.sum()),
        "internal_train_evaluation_n": int((~fit).sum()),
        "fit_accuracy": float((labels[fit] == predicted[fit]).mean()),
        "internal_train_accuracy": float((labels[~fit] == predicted[~fit]).mean()),
        "internal_confusion_rows_actual_columns_predicted": confusion.tolist(),
        "skills": skills,
        "partition": partition,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "oracle_feature": "CAN position in robot base frame",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
