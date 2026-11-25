"""Deterministic supplemental metrics derived from recorded episode arrays.

Frame and unit evidence
-----------------------
``episode-XX.steps.npz`` does not store frame metadata. The recorded arrays are written by
``eval/recorder.py::record_step`` from ``RobotState``: ``ee_xyz`` is copied from
``state.ee_pose.position`` and ``t`` from ``observation.t``. The project convention in
``AGENTS.md`` is SI units (m, rad, s) and poses in the robot base frame unless named
``_world``; ADR-0031 records the ``steps.npz`` file contract. Accordingly these helpers
accept only ``frame="base"`` and ``units="m"`` and require callers to state both explicitly.
No world-frame or unspecified-frame path length is inferred.

These are supplemental read-only derivations. They do not alter ``EpisodeMetrics``,
``summary.json``, ``episodes.csv``, provenance, or any frozen campaign result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

BaseFrame = Literal["base"]
Meters = Literal["m"]
Seconds = Literal["s"]


def _confirm_frame_and_units(frame: str, units: str) -> None:
    if frame != "base":
        raise ValueError(
            f"ee_xyz frame must be explicitly confirmed as 'base', got {frame!r}; "
            "the NPZ carries no frame metadata"
        )
    if units != "m":
        raise ValueError(
            f"ee_xyz units must be explicitly confirmed as 'm', got {units!r}; "
            "the NPZ carries no unit metadata"
        )


def _as_numeric_array(values: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float64]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"expected a numeric array, got dtype {array.dtype!r}")
    return array.astype(np.float64, copy=False)


def end_effector_path_length_m(
    ee_xyz: NDArray[np.floating] | NDArray[np.integer],
    *,
    frame: BaseFrame,
    units: Meters,
) -> float:
    """Return the end-effector path length in metres, in the robot base frame.

    Definition: the sum of Euclidean norms of consecutive ``ee_xyz`` differences.
    Edge cases: an empty trajectory is rejected (no value is invented); a single
    recorded sample has zero path length. Non-finite coordinates are rejected.
    """
    _confirm_frame_and_units(frame, units)
    array = _as_numeric_array(ee_xyz)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"ee_xyz must have shape (n, 3), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("ee_xyz is empty; path length is undefined, not zero")
    if not np.all(np.isfinite(array)):
        raise ValueError("ee_xyz contains non-finite values")
    if array.shape[0] == 1:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def simulation_duration_s(t: NDArray[np.floating] | NDArray[np.integer]) -> float:
    """Return recorded simulation duration in seconds as ``t[-1] - t[0]``.

    This is not wall-clock execution time. Edge cases: an empty series is rejected;
    a single timestamp has zero duration. Non-finite, repeated, or decreasing
    timestamps are rejected rather than silently repaired.
    """
    array = _as_numeric_array(t)
    if array.ndim != 1:
        raise ValueError(f"t must have shape (n,), got {array.shape}")
    if array.size == 0:
        raise ValueError("t is empty; duration is undefined, not zero")
    if not np.all(np.isfinite(array)):
        raise ValueError("t contains non-finite values")
    if array.size == 1:
        return 0.0
    if np.any(np.diff(array) <= 0):
        raise ValueError("t must be strictly increasing")
    return float(array[-1] - array[0])


@dataclass(frozen=True)
class SupplementalTrajectoryMetrics:
    """Supplemental values derived only from an existing ``steps.npz`` recording."""

    end_effector_path_length_m: float
    simulation_duration_s: float
    sample_count: int
    ee_frame: BaseFrame
    ee_units: Meters
    time_units: Seconds


def derive_supplemental_trajectory_metrics(
    *,
    ee_xyz: NDArray[np.floating] | NDArray[np.integer],
    t: NDArray[np.floating] | NDArray[np.integer],
    frame: BaseFrame,
    units: Meters,
) -> SupplementalTrajectoryMetrics:
    """Derive path length and simulation duration from arrays sharing one time axis."""
    positions = _as_numeric_array(ee_xyz)
    timestamps = _as_numeric_array(t)
    if positions.ndim == 2 and timestamps.ndim == 1 and positions.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"ee_xyz has {positions.shape[0]} rows but t has {timestamps.shape[0]} samples"
        )
    path_length = end_effector_path_length_m(positions, frame=frame, units=units)
    duration = simulation_duration_s(timestamps)
    return SupplementalTrajectoryMetrics(
        end_effector_path_length_m=path_length,
        simulation_duration_s=duration,
        sample_count=int(positions.shape[0]),
        ee_frame=frame,
        ee_units=units,
        time_units="s",
    )
