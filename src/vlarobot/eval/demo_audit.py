"""Physical TRAIN demonstration admission; independent of the frozen evaluation success flag."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemoAuditSpec:
    place_world_m: tuple[float, float, float]
    place_tolerance_m: float
    height_tolerance_m: float
    tilt_limit_deg: float
    lift_min_m: float
    can_radius_m: float
    close_align_tolerance_m: float
    open_command_m: float
    open_tolerance_m: float
    sample_dt_s: float


def audit_arrays(
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    diagnostics: list[dict[str, Any]],
    spec: DemoAuditSpec,
) -> dict[str, Any]:
    """Fail closed on incomplete motion evidence, false successes and non-TRAIN provenance."""
    reasons: list[str] = []
    scenario = meta.get("scenario") or {}
    if scenario.get("metadata", {}).get("split") != "train" or not scenario.get(
        "name", ""
    ).startswith("train-"):
        reasons.append("not an explicitly TRAIN scenario")
    outcome = meta.get("outcome") or {}
    if outcome.get("success") is not True:
        reasons.append("policy outcome is not successful")
    if not all(
        outcome.get(k) is True
        for k in ("home_reached", "place_distance_ok", "can_height_ok", "can_tilt_ok")
    ):
        reasons.append("frozen final criteria are not all explicitly satisfied")
    if outcome.get("rejections", 0) != 0:
        reasons.append("episode contains safety rejections")
    t = np.asarray(arrays["t"])
    if len(t) < 2 or not np.all(np.isfinite(t)):
        return {"accepted": False, "reasons": [*reasons, "missing finite time stream"]}
    if not np.allclose(np.diff(t), spec.sample_dt_s, atol=1e-8, rtol=0):
        reasons.append("recording is not at the declared lockstep sample cadence")
    if meta.get("frame_count") != len(t):
        reasons.append("RGB does not cover every training observation")
    names = arrays["gt_names"].tolist()
    if "CAN" not in names or len(diagnostics) != len(t):
        return {"accepted": False, "reasons": [*reasons, "missing aligned CAN/pad measurements"]}
    can = np.asarray(arrays["gt_xyz"][:, names.index("CAN"), :])
    quat = np.asarray(arrays["gt_quat"][:, names.index("CAN"), :])
    if not np.all(np.isfinite(can)) or not np.all(np.isfinite(quat)):
        return {"accepted": False, "reasons": [*reasons, "nonfinite CAN evidence"]}
    z_rise = can[:, 2] - can[0, 2]
    max_rise = float(z_rise.max())
    distance = float(np.linalg.norm(can[-1] - spec.place_world_m))
    height_error = float(can[-1, 2] - can[0, 2])
    tilt = float(
        np.degrees(np.arccos(np.clip(1 - 2 * (quat[-1, 0] ** 2 + quat[-1, 1] ** 2), -1, 1)))
    )
    checks = {
        "grasp": False,
        "lift": max_rise >= spec.lift_min_m,
        "transport": False,
        "placement": distance <= spec.place_tolerance_m
        and abs(height_error) <= spec.height_tolerance_m
        and tilt < spec.tilt_limit_deg,
    }
    held_indices = []
    finger_extrema = []
    for i, row in enumerate(diagnostics):
        if abs(float(row.get("t", float("inf"))) - float(t[i])) > 1e-8:
            reasons.append("physical diagnostics and observations are not time aligned")
            break
        fingers = row.get("finger_positions_m")
        pads = row.get("pad_positions_world_m")
        if (
            not isinstance(fingers, list)
            or len(fingers) != 2
            or not isinstance(pads, list)
            or len(pads) != 2
        ):
            continue
        q = np.asarray(fingers, dtype=float)
        p = np.asarray(pads, dtype=float)
        if p.shape != (2, 3) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            continue
        finger_extrema.extend(q.tolist())
        axis = p[1] - p[0]
        separation = float(np.linalg.norm(axis))
        if separation <= 0:
            continue
        midpoint = np.mean(p, axis=0)
        delta = can[i] - midpoint
        between = abs(float(delta @ axis / separation)) <= separation / 2
        if (
            z_rise[i] >= spec.lift_min_m
            and between
            and np.linalg.norm(delta[:2]) <= spec.close_align_tolerance_m
            and np.mean(np.abs(q)) < spec.open_command_m - spec.open_tolerance_m
        ):
            held_indices.append(i)
    transport = 0.0
    if len(held_indices) >= 2:
        carried = can[held_indices, :2]
        transport = float(np.linalg.norm(carried - carried[0], axis=1).max())
        checks["grasp"] = True
        checks["transport"] = transport >= spec.can_radius_m
    for check, okay in checks.items():
        if not okay:
            reasons.append(f"missing physical {check} evidence")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "checks": checks,
        "max_can_rise_m": max_rise,
        "carried_horizontal_distance_m": transport,
        "final_target_distance_m": distance,
        "final_height_error_m": height_error,
        "final_tilt_deg": tilt,
        "finger_sensor_min_m": min(finger_extrema) if finger_extrema else None,
        "finger_sensor_max_m": max(finger_extrema) if finger_extrema else None,
        "sensor_soft_limit_excursions_are_not_command_labels": True,
    }


def audit_demo(meta_path: Path, spec: DemoAuditSpec) -> dict[str, Any]:
    """Read only newly collected evidence; no training data or success is manufactured."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    steps_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".steps.npz"))
    diag_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".physical.jsonl"))
    with np.load(steps_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    diagnostics = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines()]
    return audit_arrays(meta, arrays, diagnostics, spec)
