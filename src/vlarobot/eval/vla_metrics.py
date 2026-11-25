"""Frozen physical VLA final criteria and literal paired gate, separate from control inputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vlarobot.core.types import Pose3D
from vlarobot.eval.settings import load_eval_settings


def final_can_outcome(
    initial: Pose3D,
    final: Pose3D,
    *,
    place_world_m: tuple[float, float, float],
    distance_tolerance_m: float,
    height_tolerance_m: float,
    tilt_limit_deg: float,
    home_reached: bool,
    failure_state: str | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    """Apply the same four baseline criteria to measured CAN pose; HOME alone never passes."""
    xyz = final.position.as_array()
    distance = float(np.linalg.norm(xyz - place_world_m))
    height = final.position.z - initial.position.z
    x, y, _, _ = final.orientation
    tilt = math.degrees(math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
    checks = {
        "home_reached": home_reached,
        "place_distance_ok": distance <= distance_tolerance_m,
        "can_height_ok": abs(height) <= height_tolerance_m,
        "can_tilt_ok": tilt < tilt_limit_deg,
    }
    success = all(checks.values())
    return {
        "success": success,
        "final_state": "home" if home_reached else "FAILED",
        **checks,
        "failure_state": None if success else (failure_state or "PLACEMENT"),
        "failure_reason": None if success else (failure_reason or "frozen final criteria failed"),
        "final_target_distance": distance,
        "can_height_error_m": height,
        "can_tilt_deg": tilt,
        "rejections": 0,  # The executor supplies the measured backend-veto counter.
        "fk_min_clearance_m": None,
    }


def paired_vla_gate(baseline: dict[str, Any], vla: dict[str, Any]) -> dict[str, Any]:
    """Only completed measured TRAIN 8×3 summaries can decide proposer role."""
    for summary in (baseline, vla):
        if (
            summary["measurement"] != "measured"
            or summary["split"] != "train"
            or summary["n_episodes"] != 24
            or summary["seeds"] != [0, 1, 2]
            or len(summary["per_scene"]) != 8
            or any(r["seeds"] != [0, 1, 2] for r in summary["per_scene"])
        ):
            raise ValueError("literal gate requires both measured frozen 24-episode TRAIN arms")
        if summary["success"]["total"] != 24 or not math.isclose(
            summary["success"]["rate"], summary["success"]["successes"] / 24
        ):
            raise ValueError("gate success rate does not match complete episode counts")
    if baseline["pipeline"] != "baseline" or vla["pipeline"] != "vla_only":
        raise ValueError("gate arms must be scripted baseline and VLA-only")
    if [(r["scene"], r["seeds"]) for r in baseline["per_scene"]] != [
        (r["scene"], r["seeds"]) for r in vla["per_scene"]
    ]:
        raise ValueError("gate requires identical paired scenes/seeds")
    rate = baseline["success"]["rate"]
    threshold = load_eval_settings().gate.threshold(rate)
    if rate == 0:
        return {
            "threshold": threshold,
            "passed": None,
            "primary": "OWNER_DECISION_REQUIRED",
            "reason": "non-discriminating zero scripted baseline",
        }
    passed = vla["success"]["rate"] >= threshold
    return {
        "threshold": threshold,
        "passed": passed,
        "primary": "vla" if passed else "scripted",
        "vla_role": "primary" if passed else "MEASURED partial result",
        "rule": "VLA_rate >= min(0.40, 0.60 * repaired_scripted_rate)",
    }
