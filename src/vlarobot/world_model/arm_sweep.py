"""Conservative full-arm swept transition check with explicit intermediate states.

This module adds a fail-closed wrapper around the existing broad-phase :class:`SweptArmCheck`.
It does NOT certify the TIAGo arm: the URDF-derived geometry provider is deliberately
unregistered, and production still omits ``check_transition`` so every prediction carries the
``unverified-arm-sweep`` hard veto until an owner-reviewed certificate exists. Unit tests only
exercise synthetic geometry.

The check is a broad AABB sweep with a point-speed certificate, not triangle-mesh contact
simulation, not a probability and not a substitute for the live SafetyMonitor veto.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.core.types import WorldState
from vlarobot.world_model.arm_geometry import Capsule, SweptArmCheck


class SweepGeometryError(ValueError):
    """Required geometry, a transform or a contact type is missing or unsupported."""


def required_substeps(
    *,
    speed_bound_m_s: float,
    dt_s: float,
    margin_m: float,
    minimum: int = 1,
    maximum: int = 4096,
    geometry_error_bound_m: float = 0.0,
    registration_uncertainty_bound_m: float = 0.0,
) -> int:
    """Conservative count keeping bounded motion plus support error below the margin.

    Eight fixed substeps at the pinned 28.49 m/s point-speed bound and 32 ms tick allow
    0.114 m of travel per sub-interval, which is larger than the 0.02 m margin. The count
    returned here is ``ceil(speed * dt / usable_margin)``, where ``usable_margin`` subtracts
    explicit geometry/registration error bounds. Missing error bounds must fail closed at the
    caller's certificate layer; passing zero is only valid for synthetic exact geometry.
    """
    if (
        isinstance(speed_bound_m_s, bool)
        or not math.isfinite(speed_bound_m_s)
        or speed_bound_m_s <= 0
        or isinstance(dt_s, bool)
        or not math.isfinite(dt_s)
        or dt_s <= 0
        or isinstance(margin_m, bool)
        or not math.isfinite(margin_m)
        or margin_m <= 0
        or isinstance(minimum, bool)
        or minimum < 1
        or isinstance(maximum, bool)
        or maximum < 1
    ):
        raise SweepGeometryError("adaptive subdivision requires finite positive inputs")
    for value, name in (
        (geometry_error_bound_m, "geometry error bound"),
        (registration_uncertainty_bound_m, "registration uncertainty bound"),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise SweepGeometryError(f"{name} must be finite and nonnegative")
    usable_margin = margin_m - geometry_error_bound_m - registration_uncertainty_bound_m
    if usable_margin <= 0.0:
        raise SweepGeometryError("geometry and registration error consume the collision margin")
    required = max(minimum, int(math.ceil(speed_bound_m_s * dt_s / usable_margin)))
    if required > maximum:
        raise SweepGeometryError(
            f"required {required} substeps exceeds the conservative cap {maximum}"
        )
    return required


@dataclass(frozen=True)
class SweepSafetyBound:
    """Explicit no-false-clear inequality for one certified transition contract."""

    margin_m: float
    point_speed_bound_m_s: float
    dt_s: float
    substeps: int
    point_motion_allowance_m: float
    joint_motion_allowance_m: float | None
    geometry_approximation_error_bound_m: float | None
    registration_uncertainty_bound_m: float | None
    required_margin_m: float | None
    residual_m: float | None
    certified: bool
    blockers: tuple[str, ...]

    def report(self) -> dict[str, object]:
        return {
            "margin_m": self.margin_m,
            "point_speed_bound_m_s": self.point_speed_bound_m_s,
            "dt_s": self.dt_s,
            "substeps": self.substeps,
            "point_motion_allowance_m": self.point_motion_allowance_m,
            "joint_motion_allowance_m": self.joint_motion_allowance_m,
            "geometry_approximation_error_bound_m": self.geometry_approximation_error_bound_m,
            "registration_uncertainty_bound_m": self.registration_uncertainty_bound_m,
            "required_margin_m": self.required_margin_m,
            "residual_m": self.residual_m,
            "inequality": (
                "margin > max(point_speed*dt/substeps, joint_displacement*reach) "
                "+ geometry_error + registration_error"
            ),
            "certified": self.certified,
            "blockers": list(self.blockers),
        }


def evaluate_sweep_safety_bound(
    *,
    margin_m: float,
    point_speed_bound_m_s: float,
    dt_s: float,
    substeps: int,
    max_joint_displacement_bound_rad: float | None = None,
    joint_reach_bound_m: float | None = None,
    geometry_approximation_error_bound_m: float | None = None,
    registration_uncertainty_bound_m: float | None = None,
) -> SweepSafetyBound:
    """Return a fail-closed bound; ``certified`` is true only for a strict positive residual.

    A false-clear is impossible only when every term is bounded and the collision margin
    strictly exceeds motion, geometry-approximation and registration uncertainty. Unknown terms
    are never replaced by zero. Joint displacement is converted to a point bound only when both
    the displacement and a finite reach are supplied.
    """
    if (
        isinstance(margin_m, bool)
        or not math.isfinite(margin_m)
        or margin_m <= 0
        or isinstance(point_speed_bound_m_s, bool)
        or not math.isfinite(point_speed_bound_m_s)
        or point_speed_bound_m_s <= 0
        or isinstance(dt_s, bool)
        or not math.isfinite(dt_s)
        or dt_s <= 0
        or isinstance(substeps, bool)
        or not isinstance(substeps, int)
        or substeps < 1
    ):
        raise SweepGeometryError("safety bound requires finite positive motion inputs")
    blockers: list[str] = []
    point_allowance = point_speed_bound_m_s * dt_s / substeps
    joint_allowance: float | None = None
    if max_joint_displacement_bound_rad is None or joint_reach_bound_m is None:
        blockers.append("joint displacement/reach bound is missing")
    else:
        for value, name in (
            (max_joint_displacement_bound_rad, "joint displacement bound"),
            (joint_reach_bound_m, "joint reach bound"),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise SweepGeometryError(f"{name} must be finite and nonnegative")
        joint_allowance = max_joint_displacement_bound_rad * joint_reach_bound_m
    motion_allowance = max(point_allowance, joint_allowance or 0.0)
    error_terms: dict[str, float | None] = {
        "geometry approximation error": geometry_approximation_error_bound_m,
        "registration uncertainty": registration_uncertainty_bound_m,
    }
    for name, value in error_terms.items():
        if value is None:
            blockers.append(f"{name} bound is missing")
        elif isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise SweepGeometryError(f"{name} bound must be finite and nonnegative")
    required = (
        motion_allowance
        + (geometry_approximation_error_bound_m or 0.0)
        + (registration_uncertainty_bound_m or 0.0)
        if not blockers
        else None
    )
    residual = margin_m - required if required is not None else None
    if residual is not None and residual <= 0.0:
        blockers.append("collision margin does not strictly dominate all bounded error terms")
    return SweepSafetyBound(
        margin_m=margin_m,
        point_speed_bound_m_s=point_speed_bound_m_s,
        dt_s=dt_s,
        substeps=substeps,
        point_motion_allowance_m=point_allowance,
        joint_motion_allowance_m=joint_allowance,
        geometry_approximation_error_bound_m=geometry_approximation_error_bound_m,
        registration_uncertainty_bound_m=registration_uncertainty_bound_m,
        required_margin_m=required,
        residual_m=residual,
        certified=not blockers,
        blockers=tuple(blockers),
    )


@dataclass(frozen=True)
class ArmSweepContract:
    """Explicit geometry coverage contract; every field must be supplied by the caller."""

    required_links: tuple[str, ...]
    required_objects: frozenset[str]
    adjacent_pairs: frozenset[frozenset[str]]
    dt_s: float
    margin_m: float
    substeps: int
    scene_complete: bool = True
    permitted_contact_objects: frozenset[str] = frozenset()
    adaptive_speed_bound_m_s: float | None = None
    max_adaptive_substeps: int = 4096

    def __post_init__(self) -> None:
        if (
            not self.required_links
            or len(set(self.required_links)) != len(self.required_links)
            or not all(isinstance(name, str) and name for name in self.required_links)
            or self.substeps < 1
            or not math.isfinite(self.dt_s)
            or self.dt_s <= 0
            or not math.isfinite(self.margin_m)
            or self.margin_m <= 0
            or self.scene_complete is not True
            or isinstance(self.max_adaptive_substeps, bool)
            or self.max_adaptive_substeps < 1
        ):
            raise SweepGeometryError("incomplete or invalid full-arm sweep contract")
        if self.adaptive_speed_bound_m_s is not None and (
            isinstance(self.adaptive_speed_bound_m_s, bool)
            or not math.isfinite(self.adaptive_speed_bound_m_s)
            or self.adaptive_speed_bound_m_s <= 0
        ):
            raise SweepGeometryError("adaptive speed bound must be positive and finite")

    @property
    def effective_substeps(self) -> int:
        """Substeps after enlarging the configured minimum for the speed certificate."""
        if self.adaptive_speed_bound_m_s is None:
            return self.substeps
        required = required_substeps(
            speed_bound_m_s=self.adaptive_speed_bound_m_s,
            dt_s=self.dt_s,
            margin_m=self.margin_m,
            minimum=self.substeps,
            maximum=self.max_adaptive_substeps,
        )
        return required


def interpolate_world_states(
    before: WorldState, after: WorldState, substeps: int
) -> tuple[WorldState, ...]:
    """Return ``substeps + 1`` states from before to after, inclusive.

    Joint values and end-effector position are linearly interpolated; orientation uses a
    sign-corrected normalised lerp. Scene geometry must be static and identical in both states:
    any moving, missing or unsupported object raises :class:`SweepGeometryError` rather than
    being silently ignored.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise SweepGeometryError("interpolation requires a positive integer substep count")
    if not isinstance(before, WorldState) or not isinstance(after, WorldState):
        raise SweepGeometryError("interpolation requires two WorldState values")
    if before.robot.base_pose != after.robot.base_pose:
        raise SweepGeometryError("base motion invalidates the base-frame sweep")
    a, b = before.robot.joint_pos, after.robot.joint_pos
    if set(a) != set(b) or not a:
        raise SweepGeometryError("interpolation requires matching nonempty joint identities")
    if any(
        not math.isfinite(float(value)) or not math.isfinite(float(b[name]))
        for name, value in a.items()
    ):
        raise SweepGeometryError("nonfinite joint configuration")
    if [o.object_id for o in before.scene.objects] != [o.object_id for o in after.scene.objects]:
        raise SweepGeometryError("moving or reidentified scene objects are unsupported")
    for first, second in zip(before.scene.objects, after.scene.objects, strict=True):
        if first.pose != second.pose or first.dimensions != second.dimensions:
            raise SweepGeometryError("moving scene geometry requires its own certified bound")
    q_a = np.asarray(before.robot.ee_pose.orientation, dtype=np.float64)
    q_b = np.asarray(after.robot.ee_pose.orientation, dtype=np.float64)
    if (
        q_a.shape != (4,)
        or q_b.shape != (4,)
        or not np.isfinite(np.r_[q_a, q_b]).all()
        or not math.isclose(float(q_a @ q_a), 1.0, abs_tol=1e-8)
        or not math.isclose(float(q_b @ q_b), 1.0, abs_tol=1e-8)
    ):
        raise SweepGeometryError("invalid end-effector transform")
    if float(q_a @ q_b) < 0.0:
        q_b = -q_b
    p_a = before.robot.ee_pose.position.as_array()
    p_b = after.robot.ee_pose.position.as_array()
    if not np.isfinite(np.r_[p_a, p_b]).all():
        raise SweepGeometryError("nonfinite end-effector position")

    states: list[WorldState] = []
    for index in range(substeps + 1):
        alpha = index / substeps
        joints = {name: float(a[name]) + alpha * (float(b[name]) - float(a[name])) for name in a}
        position = p_a + alpha * (p_b - p_a)
        quaternion = (1.0 - alpha) * q_a + alpha * q_b
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise SweepGeometryError("degenerate interpolated orientation")
        robot = copy.deepcopy(before.robot)
        robot.joint_pos = joints
        interpolated_quaternion: tuple[float, float, float, float] = (
            float(quaternion[0]) / norm,
            float(quaternion[1]) / norm,
            float(quaternion[2]) / norm,
            float(quaternion[3]) / norm,
        )
        robot.ee_pose = type(before.robot.ee_pose)(
            type(before.robot.ee_pose.position)(*map(float, position)),
            interpolated_quaternion,
        )
        states.append(WorldState(copy.deepcopy(before.scene), robot))
    if states[0].robot.joint_pos != dict(a):
        raise SweepGeometryError("interpolation endpoints changed")
    return tuple(states)


class IntermediateSweptArmCheck:
    """Call the broad-phase swept check on every interpolated sub-interval.

    Returns ``True`` (clear), ``False`` (conservative collision rejection) or ``None``
    (geometry/transform/contact coverage is incomplete, so the hard veto must remain).
    """

    def __init__(
        self,
        geometry: Callable[[WorldState], tuple[Capsule, ...]],
        contract: ArmSweepContract,
        *,
        scene: Any = None,
        expected_base_pose_world: Any = None,
    ) -> None:
        self.contract = contract
        self.geometry = geometry
        self.substeps = contract.effective_substeps
        self.inner = SweptArmCheck(
            geometry,
            required_links=contract.required_links,
            required_objects=contract.required_objects,
            scene_complete=contract.scene_complete,
            dt_s=contract.dt_s / self.substeps,
            margin_m=contract.margin_m,
            adjacent_pairs=contract.adjacent_pairs,
            permitted_contact_objects=contract.permitted_contact_objects,
            scene=scene,
            expected_base_pose_world=expected_base_pose_world,
        )

    def __call__(self, before: WorldState, after: WorldState) -> bool | None:
        try:
            states = interpolate_world_states(before, after, self.substeps)
            results = [self.inner(states[i], states[i + 1]) for i in range(len(states) - 1)]
        except Exception:  # noqa: BLE001 - fail closed on any missing/invalid geometry
            return None
        if any(value is None for value in results):
            return None
        return all(value is True for value in results)


@dataclass(frozen=True)
class LinkCollisionPrimitive:
    """One URDF `<collision>` shape, conservatively bounded by a capsule in the link frame."""

    capsule: Capsule
    shape: str

    @property
    def certified_containment(self) -> bool:
        return self.shape in {"cylinder", "box", "sphere"}


def _rpy_axes(rpy: Sequence[float]) -> np.ndarray:
    if len(rpy) != 3 or not np.isfinite(np.asarray(rpy, dtype=float)).all():
        raise SweepGeometryError("invalid collision origin rpy")
    roll, pitch, yaw = map(float, rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def urdf_collision_capsules(
    urdf_path: Path,
    link_names: Sequence[str],
    *,
    max_point_speed_m_s: float,
) -> dict[str, tuple[LinkCollisionPrimitive, ...]]:
    """Derive conservative capsule bounds for EVERY supported collision shape of each link.

    Each cylinder is bounded by a capsule on its axis; each box is bounded by a capsule along
    its longest axis with radius equal to the half-diagonal of the other two axes; each sphere
    is a degenerate capsule. Unsupported shapes raise :class:`SweepGeometryError`. This is a
    containment bound, not a mesh-exact contact model, and the resulting capsules remain
    UNCERTIFIED until an owner reviews the frames and containment.
    """
    speed = max_point_speed_m_s
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise SweepGeometryError("point-speed certificate must be an explicit positive number")
    speed = float(speed)
    if not math.isfinite(speed) or speed <= 0.0:
        raise SweepGeometryError("point-speed certificate must be positive and finite")
    if not link_names:
        raise SweepGeometryError("complete relevant link list required")
    root = ET.parse(Path(urdf_path)).getroot()
    found: dict[str, tuple[LinkCollisionPrimitive, ...]] = {}
    for name in link_names:
        link = root.find(f'./link[@name="{name}"]')
        if link is None or link.findall("collision") == []:
            raise SweepGeometryError(f"missing collision geometry for arm link {name}")
        primitives: list[LinkCollisionPrimitive] = []
        for index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            origin = collision.find("origin")
            if geometry is None or origin is None:
                raise SweepGeometryError(f"incomplete collision origin for {name}")
            xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
            rotation = _rpy_axes(
                [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
            )
            center = np.asarray(xyz, dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise SweepGeometryError(f"invalid collision origin for {name}")
            shapes = [child for child in geometry if child.tag in {"cylinder", "box", "sphere"}]
            if len(shapes) != 1 or len(list(geometry)) != 1:
                raise SweepGeometryError(f"unsupported collision geometry for {name}")
            shape = shapes[0]
            if shape.tag == "cylinder":
                radius = float(shape.attrib["radius"])
                half = float(shape.attrib["length"]) / 2.0
                if radius <= 0.0 or half < 0.0:
                    raise SweepGeometryError(f"invalid cylinder geometry for {name}")
                axis = rotation @ np.asarray((0.0, 0.0, 1.0))
                start, end = center - half * axis, center + half * axis
                capsule_radius = radius
            elif shape.tag == "sphere":
                capsule_radius = float(shape.attrib["radius"])
                if capsule_radius <= 0.0:
                    raise SweepGeometryError(f"invalid sphere geometry for {name}")
                start = end = center
            else:
                dimensions = np.asarray(
                    [float(value) for value in shape.attrib["size"].split()], dtype=np.float64
                )
                if dimensions.shape != (3,) or not np.all(dimensions > 0.0):
                    raise SweepGeometryError(f"invalid box geometry for {name}")
                long_axis = int(np.argmax(dimensions))
                half = dimensions[long_axis] / 2.0
                others = [dimensions[i] / 2.0 for i in range(3) if i != long_axis]
                capsule_radius = math.hypot(*others)
                local = np.zeros(3)
                local[long_axis] = half
                axis = rotation @ local
                start, end = center - axis, center + axis
            start_xyz = (float(start[0]), float(start[1]), float(start[2]))
            end_xyz = (float(end[0]), float(end[1]), float(end[2]))
            capsule = Capsule(
                name=f"{name}:{index}",
                start=start_xyz,
                end=end_xyz,
                radius_m=float(capsule_radius),
                max_point_speed_m_s=speed,
            )
            primitives.append(LinkCollisionPrimitive(capsule, shape.tag))
        found[name] = tuple(primitives)
    if set(found) != set(link_names):
        raise SweepGeometryError("incomplete link geometry coverage")
    return found
