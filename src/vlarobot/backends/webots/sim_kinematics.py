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
  ``T(θ) = D(θ-θ_ref) @ T(θ_ref)`` where ``D(d)=Trans(c)@Rot(â,d)@Trans(-c)``
  is rotation about the fitted axis line ``(â,c)`` in the parent frame (â from
  the measured relative rotation, c the point on the axis closest to origin,
  solved from ``(I-R)c=t_perp``). This matches the conjugation
  ``T(θ1)T(θ0)⁻¹ = D(θ1-θ0)`` including the anchor offset (not assumed through
  origin). In IKPy form: ``T(θ) = (Trans(c)@Rot(â,θ)) @ K`` with
  ``K = Rot(â,-θ_ref)@Trans(-c)@T(θ_ref)``, represented as a revolute URDFLink
  (origin=c, axis=â) followed by fixed ``K``; the variable is the absolute
  joint angle and recorded motor limits are IK bounds.
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

import hashlib
import json
import logging
import math
import os
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

# PROTO-derived gripper pad geometry in the arm_7/wrist frame (R2025a TiagoGripper.proto).
# Derivation (q=0, geometric, UNVERIFIED live):
# - Gripper mounted on arm_7 endEffectorSlot: translation [0, 0.016, 0],
#   rotation axis-angle [0.5774, 0.5774, -0.5774, 2.0944] (arm_7 -> gripper).
# - Left/right sliders axis [-1,0,0]/[1,0,0], maxPosition 0.045 m; left finger link Rz(pi).
# - Main pad box in finger link: center [0.004, 0, -0.1741], size [0.0076, 0.05091, 0.08982]
#   => pad bottom center z = -0.1741 - 0.08982/2 = -0.21901 in finger link.
# - Transformed to arm_7 frame (both fingers at q=0): pad-center midpoint ~(0, 0.1901, 0) m,
#   pad-bottom midpoint ~(0, 0.2350, 0) m (differs from legacy fallback [0,0,-finger_tip_offset_m]).
PROTO_PAD_CENTER_ARM7_M = np.array([0.0, 0.1901, 0.0], dtype=float)
PROTO_PAD_BOTTOM_ARM7_M = np.array([0.0, 0.2350, 0.0], dtype=float)


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
    axis_point: np.ndarray  # point on axis line in parent frame (perp to axis)
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
    tool_frame_source: str = "fallback"
    expected_child_transforms: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]


def snapshot_session_hash(session_id: dict[str, object] | None) -> str | None:
    """Stable short hash of a snapshot/session identity (ADR-0020)."""
    if not isinstance(session_id, dict):
        return None
    blob = json.dumps(session_id, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def snapshot_pointer_path(path: Path) -> Path:
    """Stable symlink-free pointer file alongside a snapshot path."""
    path = Path(path)
    return path.with_name(path.stem + ".pointer")


def _read_pointer_target(path: Path) -> Path | None:
    pointer = snapshot_pointer_path(path)
    for candidate in (pointer, path):
        if not candidate.is_file():
            continue
        try:
            raw = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw:
            continue
        # pointer text file contains hashed snapshot filename (or relative path)
        if raw.endswith(".json") and "\n" not in raw and "{" not in raw[:1]:
            target = (candidate.parent / raw).resolve()
            if target.is_file():
                return target
        # also accept JSON pointer object for robustness
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            ref = obj.get("snapshot") or obj.get("pointer")
            if isinstance(ref, str) and ref.endswith(".json"):
                target = (candidate.parent / ref).resolve()
                if target.is_file():
                    return target
    return None


def resolve_snapshot_path(path: Path) -> Path:
    """Resolve a stable pointer/base path to the actual session-hashed snapshot file."""
    path = Path(path)
    target = _read_pointer_target(path)
    if target is not None:
        return target
    return path


def snapshot_content_hash(payload: dict[str, object]) -> str:
    """SHA-256 over canonical JSON of the payload excluding the content hash field."""
    material = {key: value for key, value in payload.items() if key != "content_hash"}
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def finalize_verified_snapshot(
    snapshot: dict[str, object],
    path: Path,
    *,
    session_id: dict[str, object] | None = None,
) -> SimKinematics:
    """Write a verified snapshot atomically; session-hashed file + stable pointer.

    Writes ``configs/generated/sim_kinematics.<session-hash>.json`` (when session_id known)
    and updates the stable symlink-free pointer file and base path for compatibility.
    On failure removes temp/canonical outputs so no loadable stale snapshot remains.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = dict(snapshot)
    if session_id is not None:
        candidate["session_id"] = session_id
    candidate["valid"] = True
    candidate["content_hash"] = snapshot_content_hash(candidate)
    data = json.dumps(candidate, sort_keys=True, allow_nan=False, indent=2).encode("utf-8")
    session = candidate.get("session_id")
    shash = snapshot_session_hash(session if isinstance(session, dict) else None)
    hashed_path = path.with_name(f"{path.stem}.{shash}{path.suffix}") if shash else path
    pointer_path = snapshot_pointer_path(path)
    tmp_path = hashed_path.with_suffix(hashed_path.suffix + ".tmp")
    debug_path = path.with_suffix(path.suffix + ".debug.json")
    try:
        tmp_path.write_bytes(data)
        model = load_snapshot(tmp_path)
        chain = build_chain(model)
        verify_export_posture(model, chain)
        os.replace(tmp_path, hashed_path)
        # stable symlink-free pointer: text filename
        pointer_path.write_text(hashed_path.name, encoding="utf-8")
        # keep base path as a copy for backward compatibility with existing loaders/tools
        try:
            path.write_bytes(data)
        except OSError:
            pass
        try:
            debug_path.unlink(missing_ok=True)
        except OSError:
            pass
        return model
    except Exception as exc:
        try:
            debug_payload = dict(candidate)
            debug_payload["valid"] = False
            debug_payload["verification_error"] = f"{type(exc).__name__}: {exc}"
            debug_payload.pop("content_hash", None)
            debug_path.write_text(
                json.dumps(debug_payload, sort_keys=True, allow_nan=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        for p in (tmp_path, hashed_path, pointer_path, path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_snapshot(path: Path, *, allow_unverified_tool: bool = False) -> SimKinematics:
    """Load and validate a simulator kinematics JSON written by the exporter.

    If ``path`` is a stable base/pointer, resolves to the session-hashed snapshot.
    Refuses fallback/unverified tool sources unless explicitly allowed.
    """
    path = resolve_snapshot_path(Path(path))
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    model = kinematics_from_payload(payload, source=str(path))
    if "UNVERIFIED" in (model.tool_source or "") and not allow_unverified_tool:
        raise ValueError(
            f"Snapshot {path}: tool_source={model.tool_source!r} "
            f"(tool_frame_source={model.tool_frame_source!r}) fallback/unverified; "
            "refusing kinematic_model=sim unless --allow-unverified-tool is passed."
        )
    return model


def kinematics_from_payload(payload: Any, *, source: str = "<memory>") -> SimKinematics:
    """Validate one snapshot payload and return the ready-to-build model."""
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot {source} must be a JSON object.")
    if payload.get("valid") is not True:
        raise ValueError(
            f"Snapshot {source} is not marked valid (valid={payload.get('valid')!r}); "
            "refusing to load an unverified or failed export."
        )
    expected_hash = payload.get("content_hash")
    if expected_hash is not None:
        if not isinstance(expected_hash, str):
            raise ValueError(f"Snapshot {source}: content_hash must be a string.")
        actual_hash = snapshot_content_hash(payload)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Snapshot {source}: content_hash mismatch "
                f"(expected {expected_hash}, got {actual_hash})."
            )
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

    base_payload = _complete_pose({"base_frame": payload.get("base_frame")}, "base_frame", source)
    joints: list[SimJoint] = []
    torso_reference: float | None = None
    torso_transform: np.ndarray | None = None  # Robot/base frame -> torso child Solid
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
            torso_reference = float(value)
            torso_transform = _relative_transform(
                base_payload, child_payload, "base->torso_child", source
            )
            torso_child_payload = child_payload
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
    requested_tool_source = payload.get("tool_frame_source")
    if not isinstance(requested_tool_source, str):
        requested_tool_source = "fallback"
    tool_offset, tool_source, tool_frame_source = _tool_offset(
        entries[-1], fingers, source, tool_frame_source=requested_tool_source
    )
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
        tool_frame_source=tool_frame_source,
        expected_child_transforms=expected_child_transforms,
        metadata={
            "source": source,
            "chain_report": chain_report,
            "webots_version": payload.get("webots_version"),
            "timestamp_utc": payload.get("timestamp_utc"),
            "sim_time_s": payload.get("sim_time_s"),
            "readability_report": payload.get("readability_report"),
            "tool_frame_source": tool_frame_source,
            "proto_pad_center_arm7_m": PROTO_PAD_CENTER_ARM7_M.tolist(),
            "proto_pad_bottom_arm7_m": PROTO_PAD_BOTTOM_ARM7_M.tolist(),
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
    derived_point: np.ndarray | None = None
    residual: float | None = None
    if probe_axis is not None and axis_source == "posture_probe":
        derived_axis, derived_point, residual = _recomputed_probe_axis(
            entry, probe_axis, transform, source
        )
        readable = entry.get("readable_axis")
        if isinstance(readable, list) and len(readable) == 3:
            cross_check_deg = _axis_angle_deg(readable, derived_axis)
    if derived_axis is None:
        # Fallback: readable axis without probe fit -> cannot place axis line; assume origin.
        raw = entry.get("readable_axis")
        if isinstance(raw, list) and len(raw) == 3:
            arr = np.asarray(raw, dtype=float)
            if np.linalg.norm(arr) > 1e-9:
                derived_axis = arr / np.linalg.norm(arr)
                derived_point = np.zeros(3)
                axis_source = str(entry.get("axis_source", "jointParameters_unverified"))
    if derived_axis is None or derived_point is None:
        raise ValueError(
            f"Snapshot {source}: joint {name} has no usable axis; re-run the exporter with "
            "posture probing enabled (--probe-axes is the default)."
        )
    return SimJoint(
        name=name,
        kind="hinge",
        axis=derived_axis,
        axis_point=derived_point,
        transform_parent_child=transform,
        angle_reference=angle_reference,
        axis_source=axis_source,
        pose_residual_m=residual,
        limits=limits_tuple,
        axis_cross_check_deg=cross_check_deg,
    )


def _recomputed_probe_axis(
    entry: dict[str, object], stored_axis: Any, transform0: np.ndarray, source: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recompute the derivation from the recorded second posture and verify.

    Returns (axis, axis_point_c, residual). The exporter stores ``derived_axis``
    (and ``derived_axis_point_m`` for axis-line hinges); the loader re-derives it
    from the recorded probe pose pair so a broken derivation or a truncated file
    cannot silently enter the chain. Residual beyond the cap is a hard error.
    """
    probe = entry.get("probe")
    if not isinstance(probe, dict) or not isinstance(probe.get("child_pose_probe"), dict):
        raise ValueError(f"Snapshot {source}: probe axis joint lacks its probe child pose.")
    probe_child = _pose_from_payload(probe["child_pose_probe"])
    probe_parent_payload = probe.get("parent_pose_probe")
    if isinstance(probe_parent_payload, dict):
        probe_parent = _pose_from_payload(probe_parent_payload)
    else:
        probe_parent = _pose_from_payload(_complete_pose(entry, "parent", source))
    transform1 = parent_from_child_world(probe_parent, probe_child)
    # Prefer the exact poses used at probe-derivation time when present, so
    # chain-consistency adjustments to entry parent/child do not invalidate recomputation.
    if isinstance(probe.get("parent_pose0"), dict) and isinstance(probe.get("child_pose0"), dict):
        t0 = parent_from_child_world(
            _pose_from_payload(probe["parent_pose0"]), _pose_from_payload(probe["child_pose0"])
        )
        if t0 is not None:
            transform0 = t0
    if transform1 is None:
        raise ValueError(f"Snapshot {source}: probe child pose is incomplete.")
    solved = hinge_axis_residual(transform0, transform1)
    if solved is None:
        raise ValueError(f"Snapshot {source}: probe postures do not form a rigid hinge move.")
    axis, angle, c, pitch, residual = solved
    stored = np.asarray(stored_axis, dtype=float)
    if stored.shape != (3,):
        raise ValueError(f"Snapshot {source}: stored derived_axis is malformed.")
    stored = stored / np.linalg.norm(stored)
    if float(abs(np.dot(stored, axis))) < 1.0 - 1e-9:
        raise ValueError(f"Snapshot {source}: recomputed axis disagrees with the stored axis.")
    stored_point = entry.get("derived_axis_point_m")
    if stored_point is not None:
        sp = np.asarray(stored_point, dtype=float)
        if sp.shape == (3,):
            diff = (sp - c) - np.dot(sp - c, axis) * axis
            if float(np.linalg.norm(diff)) > 1e-4:
                raise ValueError(
                    f"Snapshot {source}: recomputed axis point disagrees with stored point."
                )
    if residual > DERIVED_AXIS_RESIDUAL_TOL_M:
        raise ValueError(
            f"Snapshot {source}: hinge axis fit residual {residual * 1e3:.4f} mm exceeds the "
            f"{DERIVED_AXIS_RESIDUAL_TOL_M * 1e3:.1f} mm cap; the two postures are not a "
            "clean rigid hinge move."
        )
    return axis, c, residual


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
    *,
    tool_frame_source: str = "fallback",
) -> tuple[np.ndarray, str, str]:
    """arm_7 endPoint frame -> tool frame.

    Returns (transform, tool_source_label, tool_frame_source).
    If readable finger endpoint solids exist, returns measured pad midpoint.
    Otherwise returns a fallback/PROTO-derived offset (UNVERIFIED).
    """
    arm7_child = _complete_pose(last_entry, "child", source)
    arm7_pose = _pose_from_payload(arm7_child)
    assert arm7_pose.position is not None and arm7_pose.rotation is not None
    R_arm7 = np.asarray(arm7_pose.rotation, dtype=float).reshape(3, 3)
    p_arm7 = np.asarray(arm7_pose.position, dtype=float)

    if fingers is not None:
        pad_positions: list[np.ndarray] = []
        ok = True
        for pose_payload in fingers.values():
            pose = _pose_from_payload(pose_payload)
            if pose.position is None or pose.rotation is None:
                ok = False
                break
            R = np.asarray(pose.rotation, dtype=float).reshape(3, 3)
            pad_world = np.asarray(pose.position, dtype=float) + R @ np.array(
                [0.0, 0.0, -FINGER_TIP_OFFSET_M]
            )
            pad_positions.append(pad_world)
        if ok and len(pad_positions) == 2:
            midpoint_world = np.mean(np.stack(pad_positions), axis=0)
            offset = R_arm7.T @ (midpoint_world - p_arm7)
            return (
                _homogeneous_translation(offset),
                "fingerEndpointMidpoint",
                "finger_endpoint_midpoint",
            )

    LOGGER.warning(
        "Snapshot %s: finger endPoint Solids unreadable; tool frame uses UNVERIFIED "
        "geometry (tool_frame_source=%s), can_radius_m %.5f stays policy-side.",
        source,
        tool_frame_source,
        CAN_RADIUS_M,
    )
    if tool_frame_source == "proto_pad_center":
        offset = PROTO_PAD_CENTER_ARM7_M.copy()
    elif tool_frame_source == "proto_pad_bottom":
        offset = PROTO_PAD_BOTTOM_ARM7_M.copy()
    else:
        tool_frame_source = "fallback"
        offset = np.array([0.0, 0.0, -FINGER_TIP_OFFSET_M], dtype=float)
    return (
        _homogeneous_translation(offset),
        "fingerTipOffsetFallback(UNVERIFIED)",
        tool_frame_source,
    )


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
    """The fixed post-rotation element ``K`` of one absolute-angle hinge line.

    Hinge motion is rotation about axis line ``(â, c)`` in the parent frame:
    ``D(d) = Trans(c) @ Rot(â,d) @ Trans(-c)`` and
    ``T(θ) = D(θ-θ_ref) @ T(θ_ref) = (Trans(c) @ Rot(â,θ)) @ K`` with
    ``K = Rot(â,-θ_ref) @ Trans(-c) @ T(θ_ref)``. IKPy composes the revolute link
    as ``origin @ Rot(axis, θ)`` followed by fixed ``K``, so we set
    ``origin_translation = c`` on the URDFLink and this fixed element to ``K``.
    """
    c = np.asarray(joint.axis_point, dtype=float)
    Tc = np.eye(4)
    Tc[:3, 3] = c
    Tm = np.eye(4)
    Tm[:3, 3] = -c
    Rref = np.eye(4)
    Rref[:3, :3] = rotation_matrix(joint.axis, -float(joint.angle_reference))
    return Rref @ Tm @ joint.transform_parent_child


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
                origin_translation=np.asarray(joint.axis_point, dtype=float),
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
