"""Static URDF-root registration from measured TRAIN geometry, without runtime oracle input."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from ikpy.chain import Chain
from ikpy.link import URDFLink


def load_registration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load T_robot_base_urdf_root; reject non-rigid, reflected or unprovenanced transforms."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("source_split") != "train":
        raise ValueError("URDF registration requires TRAIN calibration provenance")
    if not value.get("evidence"):
        raise ValueError("URDF registration requires measurement evidence")
    matrix = np.asarray(value.get("T_robot_base_urdf_root"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("URDF registration must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("URDF registration must be a proper rigid transform")
    # URDFLink uses fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll).
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) < 1e-8:
        raise ValueError("URDF registration Euler representation is singular")
    rpy = np.array(
        [
            math.atan2(rotation[2, 1], rotation[2, 2]),
            pitch,
            math.atan2(rotation[1, 0], rotation[0, 0]),
        ]
    )
    return matrix[:3, 3], rpy


def register_chain(chain: Chain, path: Path) -> Chain:
    """Prepend a fixed registration so FK, IK, scoring and joint bounds share one frame."""
    translation, rpy = load_registration(path)
    registration = URDFLink(
        name="robot_base_to_urdf_root",
        origin_translation=translation,
        origin_orientation=rpy,
        joint_type="fixed",
    )
    # Keep the existing identity root and all original link objects/limits. Kinematics
    # reassigns the arm-only active mask by name after construction.
    return Chain(
        name=chain.name,
        links=[chain.links[0], registration, *chain.links[1:]],
        active_links_mask=[False] * (len(chain.links) + 1),
    )
