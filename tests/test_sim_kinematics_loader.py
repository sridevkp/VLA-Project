"""Loader tests for the ADR-0020 sim-derived chain (Webots-free).

A synthetic snapshot is "measured" with an exact forward model: every hinge's
recorded parent->child world poses follow ``T(θ) = Rot(â, θ) @ K`` for known
``â``, ``θ_ref`` and ``K``. The tests pin the loader's contract:

- verification at the export posture is exact (composition identity holds);
- FK at *perturbed* absolute angles matches the hand-built chain, which pins
  the rotation-link-then-``FixedLink(K)`` element order;
- base-tilt invariance: conjugating every recorded world pose by one rigid
  base transform leaves the recovered relative chain unchanged;
- rejection cases: unreadable axis, torso reference mismatch, probe residual
  beyond the fit cap, incomplete fingers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from vlarobot.backends.webots.sim_kinematics import (
    after_element_name,
    build_chain,
    chain_element_index,
    finalize_verified_snapshot,
    kinematics_from_payload,
    load_snapshot,
    snapshot_content_hash,
    validate_torso,
    verify_export_posture,
)

TORSO_REF = 0.35
TORSO_TRANSLATION = np.array([0.02, 0.0, 0.34])
TORSO_TO_ARM1 = np.array([0.0, 0.0, 0.02])
ARM_AXES = {
    "arm_1_joint": np.array([0.0, 0.0, 1.0]),
    "arm_2_joint": np.array([1.0, 0.0, 0.0]),
    "arm_3_joint": np.array([0.0, 1.0, 0.0]),
}
ARM_ORIGINS = {
    "arm_1_joint": np.array([0.10, 0.00, 0.05]),
    "arm_2_joint": np.array([0.15, 0.00, 0.00]),
    "arm_3_joint": np.array([0.20, 0.00, 0.10]),
}
ARM_LIMITS = (-np.pi, np.pi)
EXPORT_ANGLES = {"arm_1_joint": 0.2, "arm_2_joint": -0.4, "arm_3_joint": 0.9}
PROBE_DELTA = 0.05


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def homogeneous(rotation, translation):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def pose(matrix):
    return {
        "world_xyz_m": np.asarray(matrix[:3, 3], dtype=float).tolist(),
        "world_orientation_matrix_row_major": np.asarray(matrix[:3, :3], dtype=float)
        .reshape(-1)
        .tolist(),
        "node_type": "Solid",
        "node_label": "synthetic",
    }


def arm_k(name):
    constant = homogeneous(rot([0.0, 0.0, 1.0], 0.3), ARM_ORIGINS[name])
    if name == "arm_1_joint":
        return constant
    if name == "arm_2_joint":
        return homogeneous(rot([1.0, 0.0, 0.0], 0.2), ARM_ORIGINS[name])
    return homogeneous(rot([0.0, 1.0, 0.0], np.pi / 2), ARM_ORIGINS[name])


def build_payload(
    *, tilt=None, torso_ref=TORSO_REF, delta=PROBE_DELTA, complete_fingers=True
) -> tuple[dict, dict[str, np.ndarray]]:
    """Complete exactly-measured snapshot; returns the payload and world frames.

    ``tilt`` conjugates every recorded world pose (the Robot node frame lives
    there) without touching the relative robot-frame geometry.
    """
    tilt = np.eye(4) if tilt is None else tilt
    world: dict[str, np.ndarray] = {"torso_child": tilt @ homogeneous(None, TORSO_TRANSLATION)}
    previous_child: np.ndarray | None = None
    for name in ARM_AXES:
        if name == "arm_1_joint":
            parent = world["torso_child"] @ homogeneous(None, TORSO_TO_ARM1)
        else:
            assert previous_child is not None
            parent = previous_child
        child = parent @ homogeneous(rot(ARM_AXES[name], EXPORT_ANGLES[name]), [0.0, 0.0, 0.0])
        child = child @ arm_k(name)
        probe_child = (
            parent @ homogeneous(rot(ARM_AXES[name], EXPORT_ANGLES[name] + delta), [0.0, 0.0, 0.0])
        ) @ arm_k(name)
        world[f"parent:{name}"] = parent
        world[f"child:{name}"] = child
        world[f"probe_child:{name}"] = probe_child
        previous_child = child
    arm7 = world["child:arm_3_joint"]
    fingers = {
        "left": pose(arm7 @ homogeneous(None, [0.02, 0.05, 0.0])),
        "right": pose(arm7 @ homogeneous(None, [0.02, -0.05, 0.0])),
    }
    entries = [
        {
            "name": "torso_lift_joint",
            "kind": "slider",
            "axis_source": "unreadable",
            "parent": pose(tilt),
            "child": pose(world["torso_child"]),
            "limits": {"min_position": 0.0, "max_position": 0.6},
            "joint_parameters": {},
            "readable_axis": None,
            "derived_axis": None,
            "probe": None,
        }
    ]
    for name in ARM_AXES:
        entries.append(
            {
                "name": name,
                "kind": "hinge",
                "axis_source": "posture_probe",
                "parent": pose(world[f"parent:{name}"]),
                "child": pose(world[f"child:{name}"]),
                "limits": {"min_position": ARM_LIMITS[0], "max_position": ARM_LIMITS[1]},
                "joint_parameters": {},
                "readable_axis": None,
                "derived_axis": [float(value) for value in ARM_AXES[name]],
                "probe": {
                    "commanded_delta": delta,
                    "theta0": EXPORT_ANGLES[name],
                    "theta1": EXPORT_ANGLES[name] + delta,
                    "theta_delta": delta,
                    "child_pose_probe": pose(world[f"probe_child:{name}"]),
                    "restore": {"restored": True, "restored_pose_error_m": 0.0},
                },
                "probe_residual": {"rotation_angle_rad": delta, "pose_residual_m": 0.0},
            }
        )
    payload = {
        "schema_version": 1,
        "units": {"length": "m", "angle": "rad"},
        "source": "synthetic",
        "webots_version": "R2025a-test",
        "sim_time_s": 12.5,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "session_id": {"pid": 1, "start_time": 0.0, "nonce": "test"},
        "base_frame": pose(tilt),
        "joints": entries,
        "finger_endpoint_solids": fingers if complete_fingers else {"left": {"world_xyz_m": None}},
        "readability_report": {
            "jointParameters": [],
            "posture_probe": list(ARM_AXES),
            "unreadable": ["torso_lift_joint"],
        },
        "posture_reference": {
            "torso_lift_joint": torso_ref,
            "arm_1_joint": EXPORT_ANGLES["arm_1_joint"],
            "arm_2_joint": EXPORT_ANGLES["arm_2_joint"],
            "arm_3_joint": EXPORT_ANGLES["arm_3_joint"],
        },
        "valid": True,
    }
    payload["content_hash"] = snapshot_content_hash(payload)
    return payload, world


def _rehash(payload: dict) -> dict:
    payload["valid"] = True
    payload["content_hash"] = snapshot_content_hash(payload)
    return payload


def test_loader_round_trip_is_exact_at_the_export_posture():
    payload, _ = build_payload()
    model = kinematics_from_payload(payload, source="<synthetic>")
    chain = build_chain(model)
    residuals = verify_export_posture(model, chain)
    assert residuals, "verification ran"
    assert all(value <= 1e-12 for value in residuals.values())
    assert model.tool_source == "fingerEndpointMidpoint"


def test_fk_at_perturbed_absolute_angles_matches_hand_built_chain():
    """The convention test: FK at non-reference angles must follow
    ``T(θ) = Rot(â, θ) @ K`` through the recorded robot-frame chain."""
    payload, _ = build_payload()
    model = kinematics_from_payload(payload)
    chain = build_chain(model)
    perturbed = {"arm_1_joint": 0.7, "arm_2_joint": 0.1, "arm_3_joint": -1.2}
    q = np.zeros(len(chain.links))
    for name, angle in perturbed.items():
        q[chain_element_index(chain, name)] = angle
    matrices = chain.forward_kinematics([float(value) for value in q], full_kinematics=True)
    hand = homogeneous(None, TORSO_TRANSLATION + TORSO_TO_ARM1)
    for name, angle in perturbed.items():
        child = hand @ homogeneous(rot(ARM_AXES[name], angle), [0.0, 0.0, 0.0]) @ arm_k(name)
        joint = next(j for j in model.joints if j.name == name)
        recovered = matrices[chain_element_index(chain, after_element_name(joint))]
        assert np.allclose(recovered, child, atol=1e-10), name
        hand = child
    tool = matrices[chain_element_index(chain, "sim_tool_offset")]
    tool_expected = hand @ model.tool_offset
    assert np.allclose(tool, tool_expected, atol=1e-10)


def test_base_tilt_invariance_of_the_recovered_chain():
    tilt = homogeneous(rot([0.0, 0.0, 1.0], -1.63) @ rot([0.0, 1.0, 0.0], 0.02), [4.1, -3.2, 0.4])
    payload_untilted, _ = build_payload()
    payload_tilted, _ = build_payload(tilt=tilt)
    model_a = kinematics_from_payload(payload_untilted)
    model_b = kinematics_from_payload(payload_tilted)
    for joint_a, joint_b in zip(model_a.joints, model_b.joints, strict=True):
        assert np.allclose(joint_a.axis, joint_b.axis, atol=1e-12)
        assert np.allclose(
            joint_a.transform_parent_child, joint_b.transform_parent_child, atol=1e-12
        )
        assert joint_a.angle_reference == joint_b.angle_reference
    assert np.allclose(model_a.torso_transform, model_b.torso_transform, atol=1e-12)
    assert np.allclose(model_a.torso_gap, model_b.torso_gap, atol=1e-12)
    assert np.allclose(model_a.tool_offset, model_b.tool_offset, atol=1e-12)


def test_tilted_fk_output_is_identical_in_the_robot_frame():
    """FK lives in the Robot node frame: the tilted chain's FK equals the
    untilted chain's FK (the base pose enters only through world_to_base)."""
    tilt = homogeneous(rot([0.0, 0.0, 1.0], -1.63), [0.1, -0.05, 0.02])
    payload, _ = build_payload(tilt=tilt)
    model = kinematics_from_payload(payload)
    chain = build_chain(model)
    payload_0, _ = build_payload()
    model_0 = kinematics_from_payload(payload_0)
    q = [0.0] * len(chain.links)
    for name, angle in EXPORT_ANGLES.items():
        q[chain_element_index(chain, name)] = angle
    tilted_fk = chain.forward_kinematics(q, full_kinematics=True)
    untilted_fk = build_chain(model_0).forward_kinematics(q, full_kinematics=True)
    for a, b in zip(tilted_fk, untilted_fk, strict=True):
        assert np.allclose(a, b, atol=1e-12)


def test_loader_rejects_missing_axis():
    payload, _ = build_payload()
    for entry in payload["joints"]:
        if entry["kind"] == "hinge":
            entry["axis_source"] = "unreadable"
            entry["derived_axis"] = None
    with pytest.raises(ValueError, match="no usable axis"):
        kinematics_from_payload(_rehash(payload), source="<test>")


def test_loader_rejects_torso_reference_mismatch():
    payload, _ = build_payload()
    payload["posture_reference"]["torso_lift_joint"] = 0.30
    with pytest.raises(ValueError, match="export posture torso"):
        kinematics_from_payload(_rehash(payload), source="<test>")


def test_loader_rejects_probe_residual_beyond_cap():
    payload, _ = build_payload()
    for entry in payload["joints"]:
        if entry["kind"] != "hinge":
            continue
        bogus = entry["probe"]["child_pose_probe"]["world_xyz_m"]
        # Shift the probe posture child by 2 cm: the two postures no longer
        # satisfy the rigid hinge model within the fit tolerance.
        entry["probe"]["child_pose_probe"]["world_xyz_m"] = [
            bogus[0] + 0.02,
            bogus[1],
            bogus[2],
        ]
    with pytest.raises(ValueError, match="residual"):
        kinematics_from_payload(_rehash(payload), source="<test>")


def test_loader_flags_tool_fallback_when_fingers_incomplete(caplog):
    caplog.set_level(logging.WARNING)
    payload, _ = build_payload(complete_fingers=False)
    model = kinematics_from_payload(payload, source="<test>")
    assert model.tool_source.startswith("fingerTipOffsetFallback")
    assert "UNVERIFIED" in model.tool_source


def test_validate_torso_flags_drift():
    payload, _ = build_payload()
    model = kinematics_from_payload(payload)
    validate_torso(model, TORSO_REF + 0.0001)
    with pytest.raises(ValueError, match="bakes the torso"):
        validate_torso(model, TORSO_REF + 0.01)


def test_json_round_trip_and_chain_shape(tmp_path: Path):
    payload, _ = build_payload()
    document = json.dumps(payload)
    reloaded = json.loads(document)
    model = kinematics_from_payload(reloaded, source="round-trip")
    chain = build_chain(model)
    assert len(chain.links) == 2 + 2 * 3 + 1


def test_wrong_ancestor_parent_is_rejected(caplog):
    payload, _ = build_payload()
    # Move arm_2's recorded parent 2 cm away from arm_1's child Solid: the
    # derived axis (a parent-relative conjugation quantity) stays valid, but
    # the parent-chain structural check must refuse the inconsistent parent.
    for entry in payload["joints"]:
        if entry["name"] == "arm_2_joint":
            parent = dict(entry["parent"])
            parent["world_xyz_m"] = [
                parent["world_xyz_m"][0] + 0.02,
                parent["world_xyz_m"][1],
                parent["world_xyz_m"][2],
            ]
            entry["parent"] = parent
    with pytest.raises(ValueError, match="previous child Solid"):
        kinematics_from_payload(_rehash(payload), source="<test>")


if __name__ == "__main__":  # pragma: no cover
    __import__("pytest").main([__file__, "-q"])


def test_failed_finalize_removes_temp_and_leaves_preexisting_snapshot_unloadable(tmp_path):
    """ADR-0020 addendum: failed export must not leave a loadable/stale snapshot."""
    path = tmp_path / "sim_kinematics.json"
    # Pre-existing invalid file (not marked valid).
    path.write_text('{"schema_version": 1, "valid": false}', encoding="utf-8")
    assert path.is_file()

    payload, _ = build_payload()
    # Make the payload fail verification: break a probe so hinge fit exceeds cap.
    for entry in payload["joints"]:
        if entry["kind"] != "hinge":
            continue
        bogus = entry["probe"]["child_pose_probe"]["world_xyz_m"]
        entry["probe"]["child_pose_probe"]["world_xyz_m"] = [
            bogus[0] + 0.02,
            bogus[1],
            bogus[2],
        ]
    _rehash(payload)

    with pytest.raises(ValueError, match="residual|Snapshot"):
        finalize_verified_snapshot(payload, path, session_id={"pid": 1, "nonce": "x"})

    # Temp removed and target not loadable (removed or otherwise invalid).
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert not path.exists() or not _is_loadable_snapshot(path)


def _is_loadable_snapshot(path):
    try:
        load_snapshot(path)
    except Exception:
        return False
    return True
