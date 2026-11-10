"""Webots-side snapshot of simulator kinematics (ADR-0020).

Provides the input data for the sim-derived IKPy chain: world poses of the base
and of every torso/arm link Solid, joint axes read from ``jointParameters`` when
the API exposes them, and an optional small-motion posture probe for axes that
cannot be read. The transform algebra lives here as pure functions so the loader
in ``sim_kinematics.py`` can reuse and unit-test it without Webots.

Semantics enforced everywhere:

- ``getPosition()``/``getOrientation()`` are only called on Pose-derived nodes
  (``Solid``/``Robot``/``Pose`` ancestors of a joint device, and ``endPoint``
  child Solids). Joint and jointParameters nodes are never asked for their pose,
  mirroring ``robot_io.joint_endpoint_world_position``.
- The frame capture itself commands nothing. The posture probe moves one joint
  by at most ``PROBE_DELTA_RAD``/``PROBE_DELTA_M``, only when ``probe_axes=True``
  and only for unreadable axes, and restores the posture afterwards.

All numbers are SI (m, rad); orientations are row-major 3x3 in Webots order
(``getOrientation`` layout, i.e. world = R @ local + position).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

#: Webots node types on which ``getPosition``/``getOrientation`` are legal.
POSE_TYPES = frozenset({"Solid", "Robot", "Pose", "SolidReference"})

#: Posture-probe amplitude for unreadable hinges (rad) and sliders (m).
PROBE_DELTA_RAD = 0.05
PROBE_DELTA_M = 0.01
#: Default probe settle/cap parameters (ADR-0020 addendum). A joint is settled
#: when ``|delta sensor| < probe_stationary_delta_rad`` for
#: ``probe_stationarity_steps`` consecutive lockstep steps.
DEFAULT_PROBE_STATIONARY_DELTA_RAD = 0.0001
DEFAULT_PROBE_STATIONARITY_STEPS = 3
DEFAULT_PROBE_MAX_STEPS_PER_JOINT = 60
DEFAULT_PROBE_TOTAL_STEP_CAP = 2000
#: Probe posture-restore tolerance: the child Solid must return within this
#: distance (m) of its pre-probe pose before probing the next joint.
PROBE_RESTORE_TOL_M = 1e-4

SCHEMA_VERSION = 1

_JOINT_KINDS = {"HingeJoint": "hinge", "SliderJoint": "slider"}


@dataclass
class SolidPose:
    """World position/rotation of one Pose-derived node, or None parts."""

    position: list[float] | None = None
    rotation: list[float] | None = None  # row-major 3x3, Webots orientation order
    node_type: str | None = None
    label: str | None = None

    def to_payload(self) -> dict[str, object]:
        """JSON-safe record of the world pose (loader input, see sim_kinematics)."""
        return {
            "world_xyz_m": self.position,
            "world_orientation_matrix_row_major": self.rotation,
            "node_type": self.node_type,
            "node_label": self.label,
        }

    @property
    def complete(self) -> bool:
        return self.position is not None and self.rotation is not None


def _guard(call, warn: str) -> Any:
    """Call a Supervisor accessor and turn any failure into ``None`` + one warn."""
    try:
        return call()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        LOGGER.warning("%s (%s).", warn, error)
        return None


def _finite_values(raw: Any, count: int) -> list[float] | None:
    if raw is None:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if len(values) != count or not all(math.isfinite(value) for value in values):
        return None
    return values


def node_label(node: Any, fallback: str) -> str:
    """Best-effort stable label: DEF name, then ``name`` field, then type name."""
    if node is None:
        return fallback
    pieces: list[str] = []
    name = _guard(node.getDef, f"{fallback}: getDef failed")
    if isinstance(name, str) and name:
        pieces.append(name)
    field_node = _guard(lambda: node.getField("name"), f"{fallback}: name field failed")
    if field_node is not None:
        value = _guard(field_node.getSFString, f"{fallback}: name getSFString failed")
        if isinstance(value, str) and value and value not in pieces:
            pieces.append(value)
    type_name = _guard(node.getTypeName, f"{fallback}: getTypeName failed")
    if isinstance(type_name, str) and type_name:
        pieces.append(type_name)
    return "/".join(pieces) if pieces else fallback


def solid_pose(node: Any, fallback_label: str) -> SolidPose:
    """Read the world pose of a Pose-derived node; never raises."""
    if node is None:
        return SolidPose(label=fallback_label)
    pose = SolidPose()
    pose.node_type = _guard(node.getTypeName, f"{fallback_label}: getTypeName failed")
    pose.label = node_label(node, fallback_label)
    values = _finite_values(_guard(node.getPosition, f"{pose.label}: getPosition failed"), 3)
    if values is not None:
        pose.position = values
    values = _finite_values(_guard(node.getOrientation, f"{pose.label}: getOrientation failed"), 9)
    if values is not None:
        pose.rotation = values
    if not pose.complete:
        LOGGER.warning(
            "Solid %s did not yield a complete world pose; its FK references are None.",
            pose.label,
        )
    return pose


def read_joint_node(io: Any, name: str) -> Any | None:
    """The HingeJoint/SliderJoint node hosting the motor with ``name``."""
    motor = io.motors.get(name)
    joint_node = io._joint_node(name) if motor is not None else None
    if joint_node is None:
        LOGGER.warning("Export: no HingeJoint/SliderJoint node found for %s.", name)
    return joint_node


def child_solid(io: Any, name: str) -> SolidPose:
    """World pose of the Pose-derived ``endPoint`` child Solid of one joint."""
    joint_node = read_joint_node(io, name)
    if joint_node is None:
        return SolidPose(label=f"{name}.endPoint")
    endpoint_field = _guard(lambda: joint_node.getField("endPoint"), f"{name}: endPoint field")
    endpoint = (
        _guard(endpoint_field.getSFNode, f"{name}: endPoint getSFNode failed")
        if endpoint_field is not None
        else None
    )
    if endpoint is None:
        LOGGER.warning("Export: joint %s has no endPoint Solid (Pose-derived child node).", name)
        return SolidPose(label=f"{name}.endPoint")
    return solid_pose(endpoint, f"{name}.endPoint")


def parent_solid(joint_node: Any, name: str) -> SolidPose:
    """World pose of the first Pose-derived ancestor of a joint node.

    For ``arm_i`` with ``i >= 2`` this is the previous link's ``endPoint`` Solid;
    for ``arm_1`` and the torso joint it is the Solid subtree that hosts the
    device (or the Robot node for the torso). The loader verifies the recovered
    parent identity against the recorded child poses, so a wrong ancestor choice
    can never silently pass into the chain.
    """
    if joint_node is None:
        return SolidPose(label=f"{name}.parent")
    current = _guard(joint_node.getParentNode, f"{name}: getParentNode failed")
    while current is not None:
        type_name = _guard(current.getTypeName, f"{name}: ancestor type failed")
        found = isinstance(type_name, str) and type_name in POSE_TYPES
        if found:
            return solid_pose(current, f"{name}.parent")
        current = _guard(current.getParentNode, f"{name}: getParentNode failed")
    LOGGER.warning("Export: joint %s has no Pose-derived ancestor Solid.", name)
    return SolidPose(label=f"{name}.parent")


def joint_parameters(joint_node: Any, name: str) -> dict[str, Any]:
    """Read ``axis``/``anchor``/``position``/``minStop``/``maxStop`` fields from
    ``jointParameters``. Only fields that exist and evaluate are included."""
    if joint_node is None:
        return {}
    parameters_field = _guard(
        lambda: joint_node.getField("jointParameters"), f"{name}: jointParameters field"
    )
    if parameters_field is None:
        return {}
    parameters_node = _guard(parameters_field.getSFNode, f"{name}: getSFNode failed")
    if parameters_node is None:
        return {}
    values: dict[str, Any] = {}
    for key, getter in (
        ("axis", "getSFVec3f"),
        ("anchor", "getSFVec3f"),
        ("position", "getSFFloat"),
        ("minStop", "getSFFloat"),
        ("maxStop", "getSFFloat"),
    ):
        field_node = _guard(
            lambda key=key: parameters_node.getField(key),
            f"{name}: {key} field",  # noqa: B008
        )
        if field_node is None:
            continue
        raw = _guard(getattr(field_node, getter), f"{name}: {key} {getter} failed")
        if key in {"axis", "anchor"}:
            values[key] = _finite_values(raw, 3)
        elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            values[key] = float(raw)
    return values


def joint_contexts(
    io: Any, joint_order: tuple[str, ...]
) -> list[tuple[str, str | None, Any | None]]:
    """Per-joint ``(name, kind, motor)`` records in capture order.

    ``kind`` is ``hinge``/``slider`` when the Webots node type is recognized;
    the motors surface drives the posture probe later.
    """
    contexts: list[tuple[str, str | None, Any | None]] = []
    for name in joint_order:
        joint_node = read_joint_node(io, name)
        kind = None
        if joint_node is not None:
            type_name = _guard(joint_node.getTypeName, f"{name}: getTypeName failed")
            kind = _JOINT_KINDS.get(type_name) if isinstance(type_name, str) else None
            if kind is None:
                LOGGER.warning(
                    "Export: joint node for %s is a %r; posture-probe fallback disabled.",
                    name,
                    type_name,
                )
        motor = io.motors.get(name)
        contexts.append((name, kind, motor))
    return contexts


# --------------------------------------------------------------------------
# Pure transform algebra (shared with the sim_kinematics loader; Webots-free).
# --------------------------------------------------------------------------


def parent_from_child_world(parent_pose: SolidPose, child_pose: SolidPose) -> np.ndarray | None:
    """Parent->child transform (4x4) from the recorded world poses of both."""
    if not (parent_pose.complete and child_pose.complete):
        return None
    assert parent_pose.rotation is not None and parent_pose.position is not None
    assert child_pose.rotation is not None and child_pose.position is not None
    R_world_parent = np.asarray(parent_pose.rotation, dtype=float).reshape(3, 3)
    p_world_parent = np.asarray(parent_pose.position, dtype=float)
    R_world_child = np.asarray(child_pose.rotation, dtype=float).reshape(3, 3)
    p_world_child = np.asarray(child_pose.position, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = R_world_parent.T @ R_world_child
    transform[:3, 3] = R_world_parent.T @ (p_world_child - p_world_parent)
    return transform


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis (rad)."""
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def translation_matrix(axis: np.ndarray, distance: float) -> np.ndarray:
    """Homogeneous translation of ``distance`` (m) along the unit ``axis``."""
    axis = axis / np.linalg.norm(axis)
    transform = np.eye(4)
    transform[:3, 3] = axis * distance
    return transform


def rotation_axis(R: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Rotation axis (unit) and angle (rad) of a rotation matrix, None if ~identity."""
    trace = float(np.clip(np.trace(R), -1.0 + 1e-12, 3.0))
    angle = math.acos((trace - 1.0) / 2.0)
    if angle < 1e-9:
        return None
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None  # 180-degree degeneracy; the probe delta is far smaller anyway
    return axis / norm, angle


#: Proxy points (child-frame) used for fit residuals: origin + the three unit
#: corners; four 4-vectors stacked as columns.
_PROXY_POINTS = np.eye(4)


def hinge_axis_residual(
    transform_parent_child0: np.ndarray, transform_parent_child1: np.ndarray
) -> tuple[np.ndarray, float, float] | None:
    """Hinge axis (parent frame, unit), rotation angle (rad), fit residual (m).

    For a hinge, ``D = T1 @ T0^-1`` is a pure rotation about the joint axis in
    the parent frame (the fixed on-axis translations of both postures cancel in
    the conjugation). The residual is the largest proxy-point deviation between
    the measured ``T1`` and the model ``Rot(axis, angle) @ T0`` — an actual fit
    quality statement, not just the commanded delta.
    """
    delta = transform_parent_child1 @ np.linalg.inv(transform_parent_child0)
    solved = rotation_axis(delta[:3, :3])
    if solved is None:
        return None
    axis, angle = solved
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_matrix(axis, angle)
    modelled = (rotation @ transform_parent_child0) @ _PROXY_POINTS.T
    measured = transform_parent_child1 @ _PROXY_POINTS.T
    residual = float(np.max(np.linalg.norm(measured - modelled, axis=0)))
    return axis, angle, residual


def slider_axis_residual(
    transform_parent_child0: np.ndarray, transform_parent_child1: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Slider axis (parent frame, unit) and fit residual (m) from two postures.

    ``D = T1 @ T0^-1`` must be a pure translation along ``axis``; the residual
    covers both perpendicular drift and any small rotation part in ``D``.
    """
    delta = transform_parent_child1 @ np.linalg.inv(transform_parent_child0)
    rotation_norm = float(np.linalg.norm(delta[:3, :3] - np.eye(3)))
    direction = delta[:3, 3]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    axis = direction / norm
    modelled = (translation_matrix(axis, norm) @ transform_parent_child0) @ _PROXY_POINTS.T
    measured = transform_parent_child1 @ _PROXY_POINTS.T
    residual = float(np.max(np.linalg.norm(measured - modelled, axis=0))) + rotation_norm
    return axis, residual


# --------------------------------------------------------------------------
# Posture probe: small known move + stationarity settle + read, restore.
# (ADR-0020 addendum: no convergence-to-command; axis from measured delta.)
# --------------------------------------------------------------------------


def joint_limits(motor: Any) -> tuple[float, float] | None:
    low = _guard(motor.getMinPosition, "getMinPosition failed")
    high = _guard(motor.getMaxPosition, "getMaxPosition failed")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    return float(low), float(high)


def _settle_stationary(
    io: Any,
    sensor: Any,
    *,
    stationary_delta: float,
    stationary_steps: int,
    max_steps: int,
) -> tuple[bool, int, float | None, str]:
    """Step until the sensor is stationary, not until it reaches a command.

    Settled when ``|delta sensor| < stationary_delta`` for ``stationary_steps``
    consecutive lockstep steps. Returns
    ``(settled, steps_used, final_value, reason)``.
    """
    last = _guard(sensor.getValue, "sensor getValue failed")
    if not isinstance(last, (int, float)):
        return False, 0, None, "sensor_unreadable"
    consecutive = 0
    steps = 0
    for _ in range(max_steps):
        if not io.step_raw(1):
            return False, steps, float(last), "step_failed"
        steps += 1
        value = _guard(sensor.getValue, "sensor getValue failed")
        if not isinstance(value, (int, float)):
            return False, steps, float(last), "sensor_unreadable"
        moved = abs(float(value) - float(last))
        last = float(value)
        if moved < stationary_delta:
            consecutive += 1
            if consecutive >= stationary_steps:
                return True, steps, float(last), "stationary"
        else:
            consecutive = 0
    return False, steps, float(last), "step_cap"


def _drive_and_settle(
    io: Any,
    name: str,
    motor: Any,
    sensor: Any,
    target: float,
    *,
    stationary_delta: float,
    stationary_steps: int,
    max_steps: int,
) -> tuple[bool, int, float | None, str]:
    """Command ``target`` and settle by stationarity; never require target convergence."""
    try:
        motor.setPosition(target)
    except (AttributeError, TypeError, RuntimeError) as error:
        LOGGER.warning("Export: setPosition failed for %s (%s).", name, error)
        return False, 0, None, "set_position_failed"
    return _settle_stationary(
        io,
        sensor,
        stationary_delta=stationary_delta,
        stationary_steps=stationary_steps,
        max_steps=max_steps,
    )


def solid_pose_distance(a: SolidPose, b: SolidPose) -> float | None:
    if a.position is None or b.position is None:
        return None
    return float(
        np.linalg.norm(np.asarray(a.position, dtype=float) - np.asarray(b.position, dtype=float))
    )


def _restore_posture(
    io: Any,
    name: str,
    motor: Any,
    sensor: Any | None,
    theta0: float,
    initial_pose: SolidPose,
    *,
    stationary_delta: float,
    stationary_steps: int,
    max_steps: int,
) -> dict[str, object]:
    """Drive the joint back and verify the child Solid pose returned."""
    status: dict[str, object] = {
        "restored_pose_error_m": None,
        "restored": False,
        "restore_steps": 0,
        "restore_reason": "not_attempted",
    }
    if motor is None or sensor is None:
        status["restore_reason"] = "unavailable"
        return status
    settled, steps, _, reason = _drive_and_settle(
        io,
        name,
        motor,
        sensor,
        theta0,
        stationary_delta=stationary_delta,
        stationary_steps=stationary_steps,
        max_steps=max_steps,
    )
    status["restore_steps"] = steps
    status["restore_reason"] = reason if settled else f"restore_{reason}"
    restored = child_solid(io, name)
    error_m = solid_pose_distance(initial_pose, restored)
    status["restored_pose_error_m"] = error_m
    status["restored"] = bool(settled and error_m is not None and error_m < PROBE_RESTORE_TOL_M)
    if not status["restored"]:
        LOGGER.warning(
            "Export: %s posture restore not fully verified (settled=%s error_m=%s reason=%s).",
            name,
            settled,
            error_m,
            status["restore_reason"],
        )
    return status


def probe_axis(
    io: Any,
    name: str,
    kind: str | None,
    motor: Any,
    sensor: Any | None,
    parent_pose: SolidPose,
    transform0: np.ndarray,
    initial_child_pose: SolidPose,
    *,
    stationary_delta: float = DEFAULT_PROBE_STATIONARY_DELTA_RAD,
    stationary_steps: int = DEFAULT_PROBE_STATIONARITY_STEPS,
    max_steps: int = DEFAULT_PROBE_MAX_STEPS_PER_JOINT,
) -> dict[str, object]:
    """Probe one joint and return a result payload (always, with status).

    Axis and angle come from the *measured* sensor change and measured poses;
    the commanded delta is recorded for traceability but never assumed reached.
    Settle is stationarity-based, so steady-state tracking error cannot stall the probe.
    """
    result: dict[str, object] = {
        "axis_source": "unreadable",
        "readable_axis": None,
        "derived_axis": None,
        "probe": None,
        "probe_residual": None,
        "probe_status": {
            "steps_used": 0,
            "measured_angle_moved_rad": None,
            "measured_delta_m": None,
            "settled": False,
            "reason": "not_attempted",
        },
    }
    if motor is None or sensor is None or kind != "hinge":
        status = result["probe_status"]
        assert isinstance(status, dict)
        status["reason"] = "not_probeable"
        return result
    try:
        theta0 = float(sensor.getValue())
    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
        LOGGER.warning("Export: sensor read failed for %s (%s).", name, error)
        status = result["probe_status"]
        assert isinstance(status, dict)
        status["reason"] = "sensor_unreadable"
        return result
    delta_cmd = PROBE_DELTA_RAD if kind == "hinge" else PROBE_DELTA_M
    limits = joint_limits(motor)
    target = theta0 + delta_cmd
    if limits is not None and not limits[0] + 1e-6 <= target <= limits[1] - 1e-6:
        target = theta0 - delta_cmd
    settled, steps, theta1, reason = _drive_and_settle(
        io,
        name,
        motor,
        sensor,
        target,
        stationary_delta=stationary_delta,
        stationary_steps=stationary_steps,
        max_steps=max_steps,
    )
    if theta1 is None:
        theta1 = theta0
    measured_delta = float(theta1 - theta0)
    status = result["probe_status"]
    assert isinstance(status, dict)
    status.update(
        {
            "steps_used": steps,
            "settled": settled,
            "reason": reason,
            "measured_angle_moved_rad" if kind == "hinge" else "measured_delta_m": measured_delta,
        }
    )
    child_probe_pose = child_solid(io, name)
    restore = _restore_posture(
        io,
        name,
        motor,
        sensor,
        theta0,
        initial_child_pose,
        stationary_delta=stationary_delta,
        stationary_steps=stationary_steps,
        max_steps=max_steps,
    )
    result["probe"] = {
        "commanded_delta": float(target - theta0),
        "theta0": theta0,
        "theta1": float(theta1),
        "theta_delta": measured_delta,
        "child_pose_probe": child_probe_pose.to_payload(),
        "restore": restore,
    }
    if not settled:
        return result
    if abs(measured_delta) < 1e-9:
        status["reason"] = "no_measured_motion"
        return result
    transform1 = parent_from_child_world(parent_pose, child_probe_pose)
    if transform1 is None:
        status["reason"] = "incomplete_pose"
        return result
    solved = hinge_axis_residual(transform0, transform1)
    if solved is None:
        status["reason"] = "axis_unsolved"
        return result
    axis = float_vec(solved[0])
    result.update(
        {
            "axis_source": "posture_probe",
            "derived_axis": axis,
            "probe_residual": {
                "rotation_angle_rad": float(solved[1]),
                "pose_residual_m": float(solved[2]),
            },
        }
    )
    status["reason"] = "ok"
    return result


def capture_snapshot(
    io: Any,
    joint_order: tuple[str, ...],
    *,
    probe_axes: bool = True,
    probe_settings: dict[str, float | int] | None = None,
) -> dict[str, object]:
    """Capture the full simulator kinematics snapshot at the current posture.

    ``joint_order`` must list the chain joints from root to tip (torso first,
    then arm joints), matching the loader's parent-child traversal. World poses
    are read for the Robot node base frame, the parent/child Solids of every
    joint, and both finger ``endPoint`` Solids. Axes come from
    ``jointParameters.axis`` when readable, else a posture probe when allowed.
    """
    base_pose = solid_pose(io.me, "RobotNode")
    joints: list[dict[str, object]] = []
    readable: list[str] = []
    pose_inferred: list[str] = []
    unreadable: list[str] = []
    for name, kind, motor in joint_contexts(io, joint_order):
        entry = _capture_joint_entry(
            io, name, kind, motor, probe_axes=probe_axes, probe_settings=probe_settings
        )
        joints.append(entry)
        source = str(entry["axis_source"])
        if source == "posture_probe":
            pose_inferred.append(name)
            if entry.get("readable_axis") is not None:
                readable.append(name)
        else:
            unreadable.append(name)
    fingers = {
        motor.getName(): child_solid(io, motor.getName()).to_payload()
        for motor, _sensor in getattr(io, "fingers", [])
    }
    snapshot: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "units": {"length": "m", "angle": "rad"},
        "source": "Webots Supervisor, Pose-derived Solid world poses",
        "webots_version": webots_version(io),
        "sim_time_s": sim_time_s(io),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_frame": base_pose.to_payload(),
        "joints": joints,
        "finger_endpoint_solids": fingers,
        "readability_report": {
            "jointParameters": readable,
            "posture_probe": pose_inferred,
            "unreadable": unreadable,
        },
        "posture_reference": posture_reference(io, joint_order),
    }
    return snapshot


def _capture_joint_entry(
    io: Any,
    name: str,
    kind: str | None,
    motor: Any | None,
    *,
    probe_axes: bool,
    probe_settings: dict[str, float | int] | None = None,
) -> dict[str, object]:
    joint_node = read_joint_node(io, name)
    child_pose = child_solid(io, name)
    parent_pose = parent_solid(joint_node, name)
    sensor = io.sensors.get(name)
    settings = probe_settings or {}
    entry: dict[str, object] = {
        "name": name,
        "kind": kind,
        "parent": parent_pose.to_payload(),
        "child": child_pose.to_payload(),
        "limits": _limits_payload(motor),
        "probe": None,
        "joint_parameters": {},
        "axis_source": "unreadable",
        "readable_axis": None,
        "derived_axis": None,
        "probe_status": None,
    }
    if joint_node is None:
        return entry
    entry["joint_parameters"] = joint_parameters(joint_node, name)
    readable_axis = _raw_axis(entry["joint_parameters"])
    if readable_axis is not None:
        entry.update({"readable_axis": readable_axis})
    if probe_axes and kind == "hinge":
        transform0 = parent_from_child_world(parent_pose, child_pose)
        if transform0 is not None:
            probed = probe_axis(
                io,
                name,
                kind,
                motor,
                sensor,
                parent_pose,
                transform0,
                child_pose,
                stationary_delta=float(
                    settings.get("stationary_delta", DEFAULT_PROBE_STATIONARY_DELTA_RAD)
                ),
                stationary_steps=int(
                    settings.get("stationarity_steps", DEFAULT_PROBE_STATIONARITY_STEPS)
                ),
                max_steps=int(
                    settings.get("max_steps_per_joint", DEFAULT_PROBE_MAX_STEPS_PER_JOINT)
                ),
            )
            entry["probe_status"] = probed.get("probe_status")
            if probed.get("axis_source") == "posture_probe":
                entry.update(
                    {
                        k: probed[k]
                        for k in ("axis_source", "derived_axis", "probe", "probe_residual")
                    }
                )
                entry["readable_axis"] = readable_axis
                entry["axis_cross_check"] = _axis_cross_check(readable_axis, entry["derived_axis"])
                return entry
            if probed.get("probe") is not None:
                entry["probe"] = probed["probe"]
            if probed.get("probe_residual") is not None:
                entry["probe_residual"] = probed["probe_residual"]
    if readable_axis is not None:
        entry["axis_source"] = "jointParameters_unverified"
    return entry


def float_vec(values: Any) -> list[float]:
    array = np.asarray(values, dtype=float)
    return [float(value) for value in array]


def webots_version(io: Any) -> str | None:
    robot = getattr(io, "robot", None)
    if robot is None:
        return None
    value = _guard(getattr(robot, "getVersion", None), "getVersion failed")
    return value if isinstance(value, str) else None


def sim_time_s(io: Any) -> float | None:
    robot = getattr(io, "robot", None)
    if robot is None:
        return None
    value = _guard(robot.getTime, "getTime failed")
    return float(value) if isinstance(value, (int, float)) else None


def _axis_cross_check(readable_axis, derived_axis) -> dict[str, float] | None:
    """Angle between the jointParameters axis and the derived axis (report only)."""
    if readable_axis is None or derived_axis is None:
        return None
    a = np.asarray(readable_axis, dtype=float) / np.linalg.norm(readable_axis)
    b = np.asarray(derived_axis, dtype=float) / float(np.linalg.norm(derived_axis))
    angle = float(np.degrees(np.arccos(np.clip(float(abs(np.dot(a, b))), -1.0, 1.0))))
    return {"angle_between_deg": angle}


def _raw_axis(parameters: dict[str, Any]) -> list[float] | None:
    """The jointParameters axis when finite and non-degenerate (None otherwise)."""
    axis = parameters.get("axis")
    if not isinstance(axis, list) or len(axis) != 3:
        return None
    values = [float(item) for item in axis]
    if not all(math.isfinite(value) for value in values):
        return None
    if float(np.linalg.norm(values)) < 1e-9:
        return None
    return values


def posture_reference(io: Any, joint_order: tuple[str, ...]) -> dict[str, float | None]:
    """Sensor values at export time (the absolute-angle reference per joint)."""
    reference: dict[str, float | None] = {}
    for name in joint_order:
        sensor = io.sensors.get(name)
        value = _guard(sensor.getValue, f"{name}: sensor getValue failed") if sensor else None
        reference[name] = float(value) if isinstance(value, (int, float)) else None
    return reference


def _limits_payload(motor: Any | None) -> dict[str, float | None]:
    limits = joint_limits(motor) if motor is not None else None
    if limits is None:
        return {"min_position": None, "max_position": None}
    return {"min_position": limits[0], "max_position": limits[1]}


def to_json_bytes(snapshot: dict[str, object]) -> bytes:
    """Serialize one snapshot deterministically (sorted keys, UTF-8, no NaN)."""
    return json.dumps(snapshot, sort_keys=True, allow_nan=False, indent=2).encode("utf-8")
