"""Coordinate transforms and current joint/tool state."""

import math

import numpy as np


def compose_can_reset_pose(translation, rotation, offset) -> tuple[list[float], list[float]]:
    """Compose a world-frame ``(dx, dy, yaw)`` CAN offset onto the world-file pose (ADR-0025).

    ``translation``/``rotation`` are the Webots ``SFVec3f``/``SFRotation`` (axis-angle) values
    read from the world; the yaw is applied about the world Z axis on top of the initial
    orientation: ``R_new = Rz(yaw) @ R_old``. Pure function so tests can import it without
    Webots.
    """
    dx, dy, yaw = offset
    shifted = [
        float(translation[0]) + float(dx),
        float(translation[1]) + float(dy),
        float(translation[2]),
    ]
    if yaw == 0.0:
        return shifted, [float(value) for value in rotation]
    composed = _rotation_z(float(yaw)) @ _axis_angle_to_matrix(rotation)
    return shifted, _matrix_to_axis_angle(composed)


def _axis_angle_to_matrix(rotation) -> np.ndarray:
    """Webots SFRotation (axis-angle) -> 3x3 rotation matrix."""
    axis = np.asarray(rotation[:3], dtype=float)
    angle = float(rotation[3])
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        if angle != 0.0:
            raise ValueError("Zero-length rotation axis with a non-zero angle.")
        return np.eye(3)
    x, y, z = axis / norm
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=float)


def _matrix_to_axis_angle(matrix: np.ndarray) -> list[float]:
    """3x3 rotation matrix -> Webots SFRotation, angle in [0, pi]."""
    rotation = np.asarray(matrix, dtype=float)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return [0.0, 0.0, 1.0, 0.0]
    if abs(angle - math.pi) < 1e-7:
        # sin(angle) vanishes near pi: recover the axis from the symmetric part.
        x = math.sqrt(max((rotation[0, 0] + 1.0) / 2.0, 0.0))
        y = math.sqrt(max((rotation[1, 1] + 1.0) / 2.0, 0.0))
        z = math.sqrt(max((rotation[2, 2] + 1.0) / 2.0, 0.0))
        if x >= y and x >= z:
            if x > 0.0:
                y, z = rotation[0, 1] / (2.0 * x), rotation[0, 2] / (2.0 * x)
        elif y >= z:
            if y > 0.0:
                x, z = rotation[0, 1] / (2.0 * y), rotation[1, 2] / (2.0 * y)
        elif z > 0.0:
            x, y = rotation[0, 2] / (2.0 * z), rotation[1, 2] / (2.0 * z)
        return [float(x), float(y), float(z), angle]
    scale = 2.0 * math.sin(angle)
    return [
        float(rotation[2, 1] - rotation[1, 2]) / scale,
        float(rotation[0, 2] - rotation[2, 0]) / scale,
        float(rotation[1, 0] - rotation[0, 1]) / scale,
        angle,
    ]


class Frames:
    def __init__(self, robot_io, kinematics):
        self.io, self.ik = robot_io, kinematics

    def base_to_world(self, point):
        R = np.array(self.io.me.getOrientation()).reshape(3, 3)
        return R @ np.asarray(point, dtype=float) + np.array(self.io.me.getPosition())

    def world_to_base(self, point):
        R = np.array(self.io.me.getOrientation()).reshape(3, 3)
        return R.T @ (np.asarray(point, dtype=float) - np.array(self.io.me.getPosition()))

    def current_q(self):
        return [
            self.io.sensors[link.name].getValue() if link.name in self.io.sensors else 0.0
            for link in self.ik.chain.links
        ]

    def tool_position(self):
        return self.ik.forward(self.current_q())[:3, 3]

    def can_base_position(self):
        return self.world_to_base(self.io.can_node.getPosition())
