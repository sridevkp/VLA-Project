"""Geometry-only projection for a reviewed finger displacement reference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def slider_displacement_m(
    parent_position_world_m: Sequence[float],
    parent_rotation_world: Sequence[float],
    marker_position_world_m: Sequence[float],
    *,
    zero_marker_parent_m: Sequence[float],
    slider_axis_parent: Sequence[float],
) -> float:
    """Project an independently located marker onto a signed parent-frame slider axis.

    The caller supplies the zero reference from pinned geometry or metrology. The projection
    uses neither the joint position field nor the motor target.
    """
    parent, marker, zero, axis = (
        np.asarray(value, dtype=float)
        for value in (
            parent_position_world_m,
            marker_position_world_m,
            zero_marker_parent_m,
            slider_axis_parent,
        )
    )
    rotation = np.asarray(parent_rotation_world, dtype=float)
    if (
        any(
            value.shape != (3,) or not np.isfinite(value).all()
            for value in (parent, marker, zero, axis)
        )
        or rotation.shape != (9,)
        or not np.isfinite(rotation).all()
    ):
        raise ValueError("finite 3D reference vectors and row-major rotation required")
    rotation = rotation.reshape(3, 3)
    if not (
        np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
        and np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-8)
        and np.isclose(axis @ axis, 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("reference requires a proper rotation and a unit slider axis")
    return float(axis @ (rotation.T @ (marker - parent) - zero))
