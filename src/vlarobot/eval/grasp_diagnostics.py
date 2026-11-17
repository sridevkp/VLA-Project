"""Read-only physical measurements from privileged grasp diagnostic recordings.

These measurements never command the robot or certify task success. A can between the
finger centres is only a geometric enclosure proxy; contact and sustained lift are separate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a completed UTF-8 JSONL recording (malformed/truncated rows fail loudly)."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def measure_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Measure pad/can geometry in world axes, retaining missing telemetry as unavailable."""
    obs = row["obs"]
    diag = obs.get("diag", {})
    can = np.asarray(row["gt"]["CAN"]["pos"], dtype=float)
    sample: dict[str, Any] = {
        "t": obs["t"],
        "can_world_xyz_m": can.tolist(),
        "can_quat_world": row["gt"]["CAN"]["quat"],
        "finger_positions_world_xyz_m": diag.get("finger_tip_world_xyz_m"),
        "finger_sensor_positions_m": diag.get("finger_joint_sensor_rad"),
        "gripper_opening_m": obs.get("gripper_opening"),
        "contact_nodes": diag.get("contact_nodes"),
        "arm7_fk_error_m": diag.get("arm_7_joint_fk_ground_truth_error_m"),
    }
    pad = diag.get("finger_midpoint_world_xyz_m")
    if pad is not None:
        pad = np.asarray(pad, dtype=float)
        relative = can - pad
        sample.update(
            pad_midpoint_world_xyz_m=pad.tolist(),
            can_minus_pad_world_xyz_m=relative.tolist(),
            pad_can_xy_error_m=float(np.linalg.norm(relative[:2])),
        )
        fk = diag.get("ee_fk_world_xyz_m")
        if fk is not None:
            sample["pad_minus_fk_world_xyz_m"] = (pad - np.asarray(fk)).tolist()
    fingers = diag.get("finger_tip_world_xyz_m")
    if isinstance(fingers, dict) and len(fingers) == 2:
        left, right = np.asarray(list(fingers.values()), dtype=float)
        separation = float(np.linalg.norm(right - left))
        sample["finger_separation_m"] = separation
        if separation > 0:
            axis = (right - left) / separation
            midpoint = (right + left) / 2
            along = float((can - midpoint) @ axis)
            sample["can_axis_offset_m"] = along
            sample["can_distance_from_finger_axis_m"] = float(
                np.linalg.norm(can - midpoint - along * axis)
            )
            sample["can_centre_between_fingers_proxy"] = abs(along) <= separation / 2
    return sample


def summarize_grasp(diag_path: Path, trace_path: Path) -> dict[str, Any]:
    """Join state boundaries to physical samples; report lift without HOME-based success."""
    rows = [row for row in read_rows(diag_path) if "obs" in row]
    if not rows:
        raise ValueError("No physical samples recorded")
    samples = [measure_sample(row) for row in rows]
    boundaries = []
    for event in read_rows(trace_path):
        if event.get("event") != "state_boundary":
            continue
        t = event["sim_time_s"]
        nearest = min(samples, key=lambda sample: abs(sample["t"] - t))
        boundaries.append(
            {
                "state": event["state"],
                "phase": event["phase"],
                "sample_time_error_s": abs(nearest["t"] - t),
                "measurement": nearest,
            }
        )
    initial_z = samples[0]["can_world_xyz_m"][2]
    return {
        "provenance": "PRIVILEGED DIAGNOSTIC: Supervisor geometry/ground truth; metrics only",
        "enclosure_note": "Centre-between-fingers is a geometry proxy, not proof of contact/grasp",
        "sample_count": len(samples),
        "initial_can_z_m": initial_z,
        "max_can_rise_m": max(s["can_world_xyz_m"][2] for s in samples) - initial_z,
        "final_can_rise_m": samples[-1]["can_world_xyz_m"][2] - initial_z,
        "boundaries": boundaries,
        "final_measurement": samples[-1],
    }


def compare_numeric_rows(
    reference: list[dict[str, Any]], other: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare step/action recordings, preserving the first exact divergence and numeric max."""

    def flatten(value: Any, prefix: str = "") -> dict[str, float]:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result.update(flatten(item, f"{prefix}.{key}"))
            return result
        if isinstance(value, list):
            result = {}
            for index, item in enumerate(value):
                result.update(flatten(item, f"{prefix}[{index}]"))
            return result
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return {prefix: float(value)}
        return {}

    first = None
    maxima: dict[str, float] = {}
    for index, (a, b) in enumerate(zip(reference, other, strict=False)):
        if a != b and first is None:
            first = index
        fa, fb = flatten(a), flatten(b)
        for key in fa.keys() & fb.keys():
            maxima[key] = max(maxima.get(key, 0.0), abs(fa[key] - fb[key]))
    if len(reference) != len(other) and first is None:
        first = min(len(reference), len(other))
    return {
        "reference_rows": len(reference),
        "other_rows": len(other),
        "first_different_row": first,
        "identical": reference == other,
        "max_abs_difference_by_field": maxima,
    }
