"""November 24 table and non-target clearance predicates for the arm-5 control."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vlarobot.world_model.analytic import segment_box_clearance


def joint_limit_margin(
    joints: Mapping[str, float], limits: Mapping[str, tuple[float, float]], tolerance_rad: float
) -> float:
    """RiskGate signed slack only. The motor monitor continues to use its original limits."""
    if (
        not limits
        or not math.isfinite(tolerance_rad)
        or tolerance_rad < 0
        or any(name not in joints or not math.isfinite(joints[name]) for name in limits)
    ):
        raise ValueError("complete finite joint-limit evidence required")
    return (
        min(min(joints[n] - lo, hi - joints[n]) for n, (lo, hi) in limits.items()) + tolerance_rad
    )


def geometric_margins(
    config: Mapping[str, Any],
    world_points_m: NDArray[np.float64],
    phases: Sequence[str],
    boxes: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return signed slack by phase and box plus the two contact endpoint slacks."""
    if (
        not phases
        or world_points_m.shape != (len(phases) + 1, 3)
        or not np.isfinite(world_points_m).all()
        or set(boxes) != {"HAM", "APP"}
    ):
        raise ValueError("complete aligned world points and non-target boxes required")
    table, objects, contacts = {}, {}, {}
    heights = world_points_m[:, 2] - config["table_top_world_z_m"]
    for i, phase in enumerate(phases):
        contact = config["contact_height_m"].get(phase)
        floor = contact["path_lower_m"] if contact else config["table_margin_m"].get(phase, 0.0)
        slack = float(min(heights[i], heights[i + 1]) - floor)
        table[phase] = min(table.get(phase, math.inf), slack)
        if contact and (i == len(phases) - 1 or phases[i + 1] != phase):
            slack = float(
                contact["tolerance_m"]
                - abs(heights[i + 1] - contact["intended_height_above_top_m"])
            )
            contacts[phase] = min(contacts.get(phase, math.inf), slack)
    for name, (starts, ends, half) in boxes.items():
        if (
            starts.shape != (len(phases), 3)
            or ends.shape != starts.shape
            or half.shape != (3,)
            or not all(np.isfinite(x).all() for x in (starts, ends, half))
            or (half <= 0).any()
        ):
            raise ValueError(f"invalid aligned oriented box coordinates: {name}")
        for phase, start, end in zip(phases, starts, ends, strict=True):
            clearance = segment_box_clearance(start, end, -half, half, config["tool_tip_radius_m"])
            key = f"{phase}:{name}"
            objects[key] = min(objects.get(key, math.inf), clearance - config["object_margin_m"])
    return {"table": table, "objects": objects}, contacts
