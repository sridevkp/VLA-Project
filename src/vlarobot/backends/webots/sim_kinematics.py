"""Sim-derived IKPy chain built from the Webots Supervisor snapshot (ADR-0020).

Loads the JSON written by ``uv run python -m vlarobot.cli export-sim-kinematics``
and builds an IKPy chain whose forward kinematics matches the measured
parent->child Solid poses. Everything here is Webots-free: it operates on the
JSON payload and the pure functions of ``supervisor_kinematics``.

Geometry model and conventions (all SI, orientations row-major 3x3):

- The chain root frame is the **Robot node frame** (``io.me``), which is exactly
  the frame ``frames.world_to_base`` uses — chain output lives in the base frame
  IK targets live in. This deliberately differs from the URDF chain, whose root
  ``Base link`` is displaced from the Robot node origin (the rigid upstream
  residual quantified in ADR-0019).
- For each measured hinge with export-posture relative transform
  ``T = T_pc(θ_ref)`` and axis ``â`` (parent frame), the joint family is
  ``T(θ) = Rot(â, θ) @ K`` with ``K = Rot(â, θ_ref)⁻¹ @ T``. This follows purely
  from the measured conjugation ``T(θ1) T(θ0)⁻¹ = Rot(â, θ1−θ0)``; no Webots
  axis-field convention is assumed (any readable ``jointParameters.axis`` is
  only cross-checked, never used).
- ikpy composes an element as ``origin @ Rot(axis, θ)``, i.e. translation first
  then rotation. The measured family is the opposite order, so the per-joint
  element sequence is the **rotation link first** (axis ``â`` in the joint's
  parent frame) followed by a fixed ``FixedLink(K)`` element; the variable is
  the **absolute** joint angle as reported by the sensors, and the recorded
  motor limits are used directly as IK bounds. Putting ``K`` first would encode
  a different hinge family (constant child-origin translation instead of a
  rotating anchor) — the perturbed-angle FK test pins the correct order.
- The torso slider is baked at its export posture (M1 keeps the torso at
  ``torso_height``, asserted at load and per FK call). Its element is the plain
  measured Robot->torsoChild transform, which needs no axis at all.
- The tool frame is the midpoint of the two finger pad centres expressed in the
  arm_7 endPoint Solid frame at the export posture (fixed element, identity
  rotation). The pad-centre offset along each finger Solid's local ``-z`` uses
  the existing ``finger_tip_offset_m`` config value; the finger endPoint Solid
  origins sit at the slider joints, not at the physical fingertip, and the
  ``finger_tip_offset_m`` geometry is itself UNVERIFIED against PROTO data —
  the load report carries ``tool_source`` for exactly this reason.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ikpy.chain import Chain
from ikpy.link import Link, OriginLink, URDFLink

from vlarobot.backends.webots.config import (
    CAN_RADIUS_M,
    FINGER_TIP_OFFSET_M,
    TORSO_HEIGHT,
    TORSO_JOINT,
)
from vlarobot.backends.webots.supervisor_kinematics import (
    SolidPose,
    hinge_axis_residual,
    parent_from_child_world,
    rotation_matrix,
)

LOGGER = logging.getLogger(__name__)

#: Loader input schema marker written by supervisor_kinematics.SCHEMA_VERSION.
REQUIRED_SCHEMA_VERSION = 1

#: Torso-baked-posture tolerance (m), matching the reset settle tolerance.
SIM_TORSO_TOLERANCE_M = 0.002
#: Required agreement between the export-posture torso sensor and config
#: torso_height (m); the baked torso element is invalid otherwise.
SIM_TORSO_EXPORT_TOL_M = 0.002
#: Parent/child world-pose gap: for arm_i (i>=2) the parent Solid IS the
#: previous child Solid, so a larger gap means a wrong ancestor decision (m).
PARENT_CHAIN_MATCH_TOL_M = 1e-4
#: Fit quality cap for the empirically derived hinge axis (proxy-point
#: residual of the two-posture rigid model, m).
DERIVED_AXIS_RESIDUAL_TOL_M = 1e-3
#: FK-at-export-posture round-trip residual cap (m): FK must reproduce the
#: recorded child poses essentially exactly; anything above is a defect.
EXPORT_POSTURE_FIT_TOL_M = 1e-6


@dataclass
class SimJoint:
    """One measured hinge joint, validated and ready for the chain."""

    name: str
    kind: str  # "hinge"
    axis: np.ndarray  # unit axis, parent frame
    transform_parent_child: np.ndarray  # at the export posture
    angle_reference: float  # absolute sensor value at the export posture
    axis_source: str  # "posture_probe" (derived) or fallback marker
    pose_residual_m: float | None  # two-posture rigid fit residual
    limits: tuple[float, float] | None  # absolute motor limits, used as IK bounds
    axis_cross_check_deg: float | None  # vs readable jointParameters axis, report only


@dataclass
class SimKinematics:
    """Validated simulator kinematics: arm joints + baked torso + tool frame."""

    joints: list[SimJoint]
    torso_reference: float  # torso_lift_joint sensor at the export posture
    torso_transform: np.ndarray  # Robot-node frame -> torso child Solid
    torso_gap: np.ndarray  # torso child frame -> arm_1 parent frame (fixed)
    tool_offset: np.ndarray  # arm_7 endPoint frame -> tool frame (identity rotation)
    tool_source: str
    expected_child_transforms: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]


def load_snapshot(path: Path) -> SimKinematics:
    """Load and validate a simulator kinematics JSON written by the exporter."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return kinematics_from_payload(payload, source=str(path))


def kinematics_from_payload(payload: Any, *, source: str = "<memory>") -> SimKinematics:
    """Validate one snapshot payload and return the ready-to-build model."""
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot {source} must be a JSON object.")
    version = payload.get("schema_version")
    if version != REQUIRED_SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot {source}: schema_version {version!r} unsupported "
            f"(required {REQUIRED_SCHEMA_VERSION})."
        )
    units = payload.get("units")
    if units != {"length": "m", "angle": "rad"}:
        raise ValueError(f"Snapshot {source}: expected SI units mapping, got {units!r}.")
    entries = payload.get("joints")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Snapshot {source}: 'joints' must be a non-empty list.")
    posture = payload.get("posture_reference")
    if not isinstance(posture, dict):
        raise ValueError(f"Snapshot {source}: 'posture_reference' must be an object.")

    joints: list[SimJoint] = []
    torso_reference: float | None = None
    torso_transform: np.ndarray | None = None
    torso_child_payload: dict[str, object] | None = None
    previous_child_payload: dict[str, object] | None = None
    previous_child_name: str | None = None
    first_arm_parent_payload: dict[str, object] | None = None
    arm_entries: list[dict[str, object]] = []
    chain_report: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        name, kind = _entry_identity(entry, index, source)
        value = posture.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Snapshot {source}: joint {name} has no finite posture reference.")
        parent_payload = _complete_pose(entry, "parent", source)
        child_payload = _complete_pose(entry, "child", source)
        transform = parent_from_child_world(
            _pose_from_payload(parent_payload), _pose_from_payload(child_payload)
        )
        if transform is None:
            raise ValueError(
                f"Snapshot {source}: joint {name} lacks a complete parent/child pose pair."
            )
        if previous_child_payload is not None:
            chain_report.append(
                {
                    "joint": name,
                    "gap_to_previous_child_m": _pose_gap(previous_child_payload, parent_payload),
                    "previous_joint": previous_child_name,
                }
            )
        previous_child_payload, previous_child_name = child_payload, name
        if kind == "slider":
            if torso_reference is not None:
                raise ValueError(f"Snapshot {source}: multiple slider joints detected.")
            torso_reference, torso_transform, torso_child_payload = (
                float(value),
                transform,
                child_payload,
            )
            continue
        if torso_reference is None:
            raise ValueError(
                f"Snapshot {source}: joint {name} precedes the torso slider in joint order."
            )
        joints.append(_validated_hinge(entry, name, transform, float(value), source))
        arm_entries.append(entry)
        if first_arm_parent_payload is None:
            first_arm_parent_payload = parent_payload
    if torso_reference is None or torso_transform is None or not joints:
        raise ValueError(f"Snapshot {source}: payload lacks a torso slider and arm hinges.")
    torso_reference_value = float(torso_reference)
    torso_height = float(TORSO_HEIGHT or 0.0)
    if abs(torso_reference_value - torso_height) > SIM_TORSO_EXPORT_TOL_M:
        raise ValueError(
            f"Snapshot {source}: export posture torso {torso_reference_value:.4f} m deviates "
            f"from configured torso_height {torso_height:.4f} m by more than "
            f"{SIM_TORSO_EXPORT_TOL_M:.4f} m; re-export the snapshot at that posture."
        )
    assert first_arm_parent_payload is not None
    torso_gap = _relative_transform(
        torso_child_payload, first_arm_parent_payload, "torso_child->arm_1_parent", source
    )
    fingers = _finger_payloads(payload, source)
    tool_offset, tool_source = _tool_offset(entries[-1], fingers, source)
    base_payload = _complete_pose({"base_frame": payload.get("base_frame")}, "base_frame", source)
    expected_child_transforms: list[np.ndarray] = []
    for arm_entry in arm_entries:
        child_payload = _complete_pose(arm_entry, "child", source)
        expected = parent_from_child_world(
            _pose_from_payload(base_payload), _pose_from_payload(child_payload)
        )
        if expected is None:
            raise ValueError(f"Snapshot {source}: cannot form the expected child transform.")
        expected_child_transforms.append(expected)
    model = SimKinematics(
        joints=joints,
        torso_reference=torso_reference,
        torso_transform=torso_transform,
        torso_gap=torso_gap,
        tool_offset=tool_offset,
        tool_source=tool_source,
        expected_child_transforms=expected_child_transforms,
        metadata={
            "source": source,
            "chain_report": chain_report,
            "webots_version": payload.get("webots_version"),
            "timestamp_utc": payload.get("timestamp_utc"),
            "sim_time_s": payload.get("sim_time_s"),
            "readability_report": payload.get("readability_report"),
        },
    )
    _validate_arm_parent_chain(model, source)
    return model


# ---------------------------------------------------------------------------
# Payload helpers.
# ---------------------------------------------------------------------------


def _entry_identity(entry: Any, index: int, source: str) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"Snapshot {source}: joint {index} is not an object.")
    name, kind = entry.get("name"), entry.get("kind")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Snapshot {source}: joint {index} lacks a name.")
    if kind not in {"hinge", "slider"}:
        raise ValueError(f"Snapshot {source}: joint {name} has unsupported kind {kind!r}.")
    return name, kind


def _pose_from_payload(payload: Any) -> SolidPose:
    data = payload if isinstance(payload, dict) else {}
    return SolidPose(
        position=data.get("world_xyz_m"),
        rotation=data.get("world_orientation_matrix_row_major"),
        node_type=data.get("node_type"),
        label=data.get("node_label"),
    )


def _complete_pose(entry: Any, key: str, source: str) -> dict[str, object]:
    """The ``key`` segment (parent/child/...) of ``entry``, required complete."""
    if not isinstance(entry, dict) or not isinstance(entry.get(key), dict):
        raise ValueError(f"Snapshot {source}: joint-level segment {key!r} missing.")
    payload = entry[key]
    pose = _pose_from_payload(payload)
    if not pose.complete:
        raise ValueError(f"Snapshot {source}: segment {key!r} lacks a complete world pose.")
    return payload


def _pose_gap(a: dict[str, object] | None, b: dict[str, object] | None) -> float | None:
    """World-position distance between two recorded Solid poses (m), or None."""
    if a is None or b is None:
        return None
    pa, pb = a.get("world_xyz_m"), b.get("world_xyz_m")
    if not isinstance(pa, list) or not isinstance(pb, list) or len(pa) != 3 or len(pb) != 3:
        return None
    return float(np.linalg.norm(np.asarray(pa, dtype=float) - np.asarray(pb, dtype=float)))


def _relative_transform(
    from_pose: dict[str, object] | None,
    to_pose: dict[str, object] | None,
    label: str,
    source: str,
) -> np.ndarray:
    """Fixed parent->child transform of two recorded world poses (chain element)."""
    if from_pose is None or to_pose is None:
        raise ValueError(f"Snapshot {source}: missing Solids for {label}.")
    transform = parent_from_child_world(_pose_from_payload(from_pose), _pose_from_payload(to_pose))
    if transform is None:
        raise ValueError(f"Snapshot {source}: incomplete world poses for {label}.")
    return transform


def _validated_hinge(
    entry: dict[str, object], name: str, transform: np.ndarray, angle_reference: float, source: str
) -> SimJoint:
    """Axis resolution + identity checks for one measured hinge joint."""
    probe_axis = entry.get("derived_axis")
    axis_source = str(entry.get("axis_source", "unreadable"))
    limits = entry.get("limits")
    limits_tuple = None
    low = limits.get("min_position") if isinstance(limits, dict) else None
    high = limits.get("max_position") if isinstance(limits, dict) else None
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        limits_tuple = (float(low), float(high))
    cross_check_deg = None
    derived_axis: np.ndarray | None = None
    residual: float | None = None
    if probe_axis is not None and axis_source == "posture_probe":
        derived_axis, residual = _recomputed_probe_axis(entry, probe_axis, transform, source)
        readable = entry.get("readable_axis")
        if isinstance(readable, list) and len(readable) == 3:
            cross_check_deg = _axis_angle_deg(readable, derived_axis)
    if derived_axis is None:
        raise ValueError(
            f"Snapshot {source}: joint {name} has no usable axis; re-run the exporter with "
            "posture probing enabled (--probe-axes is the default)."
        )
    return SimJoint(
        name=name,
        kind="hinge",
        axis=derived_axis,
        transform_parent_child=transform,
        angle_reference=angle_reference,
        axis_source=axis_source,
        pose_residual_m=residual,
        limits=limits_tuple,
        axis_cross_check_deg=cross_check_deg,
    )


def _recomputed_probe_axis(
    entry: dict[str, object], stored_axis: Any, transform0: np.ndarray, source: str
) -> tuple[np.ndarray, float]:
    """Recompute the derivation from the recorded second posture and verify.

    The exporter stores ``derived_axis``; the loader re-derives it from the
    recorded probe pose pair so a broken derivation or a truncated file cannot
    silently enter the chain, and returns the fit residual statement required by
    the export (residual beyond the cap is a hard error).
    """
    probe = entry.get("probe")
    if not isinstance(probe, dict) or not isinstance(probe.get("child_pose_probe"), dict):
        raise ValueError(f"Snapshot {source}: probe axis joint lacks its probe child pose.")
    probe_child = _pose_from_payload(probe["child_pose_probe"])
    parent_payload = _complete_pose(entry, "parent", source)
    transform1 = parent_from_child_world(_pose_from_payload(parent_payload), probe_child)
    if transform1 is None:
        raise ValueError(f"Snapshot {source}: probe child pose is incomplete.")
    solved = hinge_axis_residual(transform0, transform1)
    if solved is None:
        raise ValueError(f"Snapshot {source}: probe postures do not form a rigid hinge move.")
    axis, angle, residual = solved
    stored = np.asarray(stored_axis, dtype=float)
    if stored.shape != (3,):
        raise ValueError(f"Snapshot {source}: stored derived_axis is malformed.")
    stored = stored / np.linalg.norm(stored)
    if float(abs(np.dot(stored, axis))) < 1.0 - 1e-9:
        raise ValueError(f"Snapshot {source}: recomputed axis disagrees with the stored axis.")
    if residual > DERIVED_AXIS_RESIDUAL_TOL_M:
        raise ValueError(
            f"Snapshot {source}: hinge axis fit residual {residual * 1e3:.4f} mm exceeds the "
            f"{DERIVED_AXIS_RESIDUAL_TOL_M * 1e3:.1f} mm cap; the two postures are not a "
            "clean rigid hinge move."
        )
    return axis, residual


def _axis_angle_deg(a: Any, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float)
    if np.linalg.norm(a_arr) < 1e-9:
        return float("nan")
    a_arr = a_arr / np.linalg.norm(a_arr)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a_arr, b))), -1.0, 1.0))))


def _tool_offset(
    last_entry: dict[str, object],
    fingers: dict[str, dict[str, object]] | None,
    source: str,
) -> tuple[np.ndarray, str]:
    """arm_7 endPoint frame -> tool frame (finger pad midpoint), from the JSON.

    Tool frame origin = midpoint of the two finger pad centres; orientation =
    arm_7 frame (identity rotation contribution), matching the URDF chain's
    translation-only TOOL_OFFSET semantics. Falls back to the configured
    ``finger_tip_offset_m`` along arm_7's local -z when a finger Solid is not
    readable (marked UNVERIFIED).
    """
    arm7_child = _complete_pose(last_entry, "child", source)
    arm7_pose = _pose_from_payload(arm7_child)
    assert arm7_pose.position is not None and arm7_pose.rotation is not None
    R_arm7 = np.asarray(arm7_pose.rotation, dtype=float).reshape(3, 3)
    p_arm7 = np.asarray(arm7_pose.position, dtype=float)

    if fingers is not None:
        pad_positions: list[np.ndarray] = []
        for pose_payload in fingers.values():
            pose = _pose_from_payload(pose_payload)
            assert pose.position is not None and pose.rotation is not None
            R = np.asarray(pose.rotation, dtype=float).reshape(3, 3)
            pad_world = np.asarray(pose.position, dtype=float) + R @ np.array(
                [0.0, 0.0, -FINGER_TIP_OFFSET_M]
            )
            pad_positions.append(pad_world)
        midpoint_world = np.mean(np.stack(pad_positions), axis=0)
        offset = R_arm7.T @ (midpoint_world - p_arm7)
        return _homogeneous_translation(offset), "fingerEndpointMidpoint"
    LOGGER.warning(
        "Snapshot %s: finger endPoint Solids unreadable; tool frame falls back to the "
        "configured finger_tip_offset_m along arm_7 local -z (UNVERIFIED geometry), "
        "finger-tip-offcan clearance reporting with can_radius_m %.5f stays policy-side.",
        source,
        CAN_RADIUS_M,
    )
    offset = np.array([0.0, 0.0, -FINGER_TIP_OFFSET_M])
    return _homogeneous_translation(offset), "fingerTipOffsetFallback(UNVERIFIED)"


def _finger_payloads(
    payload: dict[str, object], source: str
) -> dict[str, dict[str, object]] | None:
    """Both finger endPoint Solid payloads, or None when either is incomplete.

    A single readable finger is rejected deliberately: a one-sided midpoint
    would silently rotate the tool frame about the CAN axis.
    """
    raw = payload.get("finger_endpoint_solids")
    if not isinstance(raw, dict) or len(raw) != 2:
        LOGGER.warning(
            "Snapshot %s: finger endPoint Solids incomplete (%d recorded).",
            source,
            len(raw) if isinstance(raw, dict) else 0,
        )
        return None
    complete: dict[str, dict[str, object]] = {}
    for name, pose_payload in raw.items():
        if not isinstance(pose_payload, dict):
            return None
        if (
            pose_payload.get("world_xyz_m") is None
            or pose_payload.get("world_orientation_matrix_row_major") is None
        ):
            LOGGER.warning("Snapshot %s: finger %s pose incomplete.", source, name)
            return None
        complete[str(name)] = pose_payload
    return complete if len(complete) == 2 else None


def _homogeneous_translation(offset: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = offset
    return transform


def _validate_arm_parent_chain(model: SimKinematics, source: str) -> None:
    """For arm_i (i >= 2), the recorded parent Solid must be the previous child
    Solid — a wrong ancestor decision cannot silently enter the chain."""
    report = model.metadata.get("chain_report")
    if not isinstance(report, list):
        return
    records = report[1:]  # report[0] is the torso->arm_1 gap (expected nonzero)
    for offset, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if offset + 1 >= len(model.joints):
            continue
        value = record.get("gap_to_previous_child_m")
        if isinstance(value, float) and value > PARENT_CHAIN_MATCH_TOL_M:
            raise ValueError(
                f"Snapshot {source}: joint {model.joints[offset + 1].name}: recorded parent "
                f"differs from the previous child Solid by {value * 1e3:.4f} mm (> "
                f"{PARENT_CHAIN_MATCH_TOL_M * 1e3:.2f} mm); the parent walk chose a "
                "different Solid."
            )


class FixedLink(OriginLink):
    """Constant element carrying one measured parent->child pose.

    ikpy's ``OriginLink`` is hard-bound to the identity matrix; FixedLink keeps
    the same role (fixed element, no variable) but with an arbitrary
    homogeneous transform, resolved as a plain pose each FK call.
    """

    def __init__(self, transform: np.ndarray, name: str) -> None:
        Link.__init__(self, name=name, length=0.0, bounds=(0.0, 0.0), is_final=False)
        self.transform = np.asarray(transform, dtype=float)
        self.has_rotation = False
        self.has_translation = False
        self.joint_type = "fixed"

    def get_link_frame_matrix(self, theta):
        return self.transform


def joint_origin(joint: SimJoint) -> np.ndarray:
    """The fixed post-rotation element ``K`` of one absolute-angle hinge.

    The measured family is ``T(θ) = Rot(â, θ) @ K`` with
    ``K = Rot(â, θ_ref)⁻¹ @ T_pc(θ_ref)``. ikpy composes elements in order, so
    the per-joint element sequence is the **rotation link first** (axis = ``â``
    expressed in the chain's current frame, which is the joint's parent frame)
    followed by the fixed ``K`` element — rotation-before-origin, exactly the
    measured semantics. Placing ``K`` before the rotation would encode a
    *different* hinge family (constant child-origin translation, axis
    conjugated into the post-rotation frame); the loader round-trip test pins
    the correct order.
    """
    delta = np.eye(4)
    delta[:3, :3] = rotation_matrix(joint.axis, joint.angle_reference)
    return np.linalg.inv(delta) @ joint.transform_parent_child


def after_element_name(joint: SimJoint) -> str:
    """Name of the fixed element that lands in the joint's child Solid frame."""
    return f"sim_after_{joint.name}"


def build_chain(model: SimKinematics) -> Chain:
    """Assemble the IKPy chain in the Robot-node base frame.

    Element order: baked torso transform (named ``torso_lift_joint`` so
    ``frames.current_q`` fills the live sensor value there for the drift
    assert), the measured torso_child->arm_1_parent gap, then per arm hinge an
    absolute-angle ``URDFLink`` (axis = ``â`` in its parent frame) followed by
    its fixed ``K`` element, and finally the fixed tool-offset element to the
    finger pad midpoint. Active links are the arm joints only, mirroring the
    URDF chain semantics.
    """
    links: list[Link] = [
        FixedLink(model.torso_transform, name=TORSO_JOINT),
        FixedLink(model.torso_gap, name="sim_gap_torso_to_arm1"),
    ]
    for joint in model.joints:
        links.append(
            URDFLink(
                name=joint.name,
                origin_translation=np.zeros(3),
                origin_orientation=np.zeros(3),
                rotation=np.asarray(joint.axis, dtype=float) / float(np.linalg.norm(joint.axis)),
                bounds=(joint.limits if joint.limits is not None else (-math.pi, +math.pi)),
                use_symbolic_matrix=False,
                joint_type="revolute",
            )
        )
        links.append(FixedLink(joint_origin(joint), name=after_element_name(joint)))
    links.append(FixedLink(model.tool_offset, name="sim_tool_offset"))
    active = [
        link.name is not None and not link.name.startswith("sim_") and link.name != TORSO_JOINT
        for link in links
    ]
    return Chain(links=links, active_links_mask=active, name="tiago_sim")


def verify_export_posture(
    model: SimKinematics, chain: Chain, tolerances: float = EXPORT_POSTURE_FIT_TOL_M
) -> dict[str, float]:
    """FK at the export posture must reproduce the recorded child poses.

    Compares the composed link matrices of every arm ``URDFLink`` (child Solid
    frame in the Robot-node frame) and the final tool element against the
    transforms stored in ``model.expected_child_transforms`` / derived tool
    offset. Raises when the residual exceeds the round-trip tolerance, because
    that indicates a chain-construction defect rather than measurement noise.
    """
    q = np.zeros(len(chain.links))
    for joint in model.joints:
        q[chain_element_index(chain, joint.name)] = joint.angle_reference
    matrices = chain.forward_kinematics([float(value) for value in q], full_kinematics=True)
    residuals: dict[str, float] = {}
    # The child Solid frame is reached after the joint's rotation AND its fixed
    # K element, so compare on the ``sim_after`` element, not the URDFLink.
    for joint, expected in zip(model.joints, model.expected_child_transforms, strict=True):
        recovered = matrices[chain_element_index(chain, after_element_name(joint))]
        residuals[joint.name] = _matrix_residual(recovered, expected)
    tool_index = chain_element_index(chain, "sim_tool_offset")
    tool_expected = model.expected_child_transforms[-1] @ model.tool_offset
    residuals["sim_tool_offset"] = _matrix_residual(matrices[tool_index], tool_expected)
    worst = max(residuals.values())
    if worst > tolerances:
        raise ValueError(
            f"FK at the export posture deviates from the recorded poses by "
            f"{worst * 1e6:.3e} m (cap {tolerances * 1e6:.3e} m): the chain is inconsistent "
            "with the snapshot."
        )
    return residuals


def _matrix_residual(measured: np.ndarray, expected: np.ndarray) -> float:
    translation = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
    rotation = float(np.linalg.norm(measured[:3, :3] - expected[:3, :3]))
    return max(translation, rotation)


def chain_element_index(chain: Chain, element_name: str) -> int:
    for index, link in enumerate(chain.links):
        if link.name == element_name:
            return index
    raise KeyError(f"Element {element_name} is not part of the built chain.")


def validate_torso(model: SimKinematics, live_value: float) -> None:
    """Assert the torso sits at the export posture (sim model precondition)."""
    if abs(float(live_value) - model.torso_reference) > SIM_TORSO_TOLERANCE_M:
        raise ValueError(
            f"sim kinematic model bakes the torso at {model.torso_reference:.4f} m; the live "
            f"torso sensor reads {live_value:.4f} m (tolerance {SIM_TORSO_TOLERANCE_M:.4f} m). "
            "Move the torso back to the export posture or re-export the snapshot."
        )
