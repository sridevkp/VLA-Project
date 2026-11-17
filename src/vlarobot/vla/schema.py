"""TIAGo feature names and gripper semantics; no ML or simulator dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CAMERA_KEY = "observation.images.camera1"
STATE_NAMES = tuple(f"arm_{i}_joint_rad" for i in range(1, 8)) + ("gripper_half_aperture_m",)
ACTION_NAMES = ("dx_m", "dy_m", "dz_m", "drx_rad", "dry_rad", "drz_rad", "gripper_target_m")


@dataclass(frozen=True)
class GripperMapping:
    """Symmetric per-finger target; feedback may exceed command soft limits under load."""

    closed_position_m: float
    open_position_m: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.closed_position_m, self.open_position_m)):
            raise ValueError("gripper endpoint coordinates must be finite")
        if self.closed_position_m == self.open_position_m:
            raise ValueError("gripper stroke must be positive")

    def target_position(self, half_aperture_m: float) -> float:
        """Map an absolute half-aperture command to both slider targets, without clipping."""
        stroke = abs(self.open_position_m - self.closed_position_m)
        if not math.isfinite(half_aperture_m) or not 0 <= half_aperture_m <= stroke:
            raise ValueError("gripper command exceeds finite half-aperture soft limits")
        if stroke == 0:
            raise ValueError("gripper stroke must be positive")
        return (
            self.closed_position_m
            + (self.open_position_m - self.closed_position_m) * half_aperture_m / stroke
        )

    def measured_half_aperture(self, finger_positions_m: tuple[float, float]) -> float:
        """Keep actual feedback, including external-force excursions; never infer a command."""
        if not all(math.isfinite(q) for q in finger_positions_m):
            raise ValueError("finger feedback must be finite")
        return sum(abs(q - self.closed_position_m) for q in finger_positions_m) / 2


def observation_state(joint_pos: dict[str, float], gripper_half_aperture_m: float) -> np.ndarray:
    """Ordered seven arm angles plus measured average per-finger opening (8 float32 values)."""
    state = np.array(
        [joint_pos[f"arm_{i}_joint"] for i in range(1, 8)] + [gripper_half_aperture_m],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(state)):
        raise ValueError("VLA observation state must be finite")
    return state
