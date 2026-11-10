"""Validation-sweep tests on synthetic telemetry (Webots-free, ADR-0020).

The sweep math is exercised on a fully synthetic "simulated" world: a measured
7-joint arm whose child Solid poses follow ``T(theta) = Rot(axis, theta) @ K``,
with the finger Solids placed so their pad midpoint is exactly the tool frame.
Scoring the sim chain on that telemetry must accept it; injecting a constant or
pose-dependent bias must reject it and produce an honest, pose-cross-validated
refinement proposal.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from vlarobot.backends.webots import kinematics as kinematics_module
from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    TABLE_FOOTPRINT_CENTRE_XY_M,
    TABLE_TOP_HEIGHT_M,
    TORSO_JOINT,
)
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.cli import _drive_pose
from vlarobot.core.types import ApplyResult
from vlarobot.execution.kinematics_validation import (
    SETTLE_TOLERANCE_RAD,
    JointSample,
    diverse_pose_targets,
    evaluate_model,
    evaluate_sweep,
    finger_midpoint_gt,
    format_report,
    pose_limits,
    propose_refinement,
    sample_is_safe,
)

TORSO_REF = 0.35
TORSO_TRANSLATION = np.array([0.02, 0.0, 0.34])
TORSO_TO_ARM1 = np.array([0.0, 0.0, 0.02])
REF_ANGLES = {name: 0.1 * (index + 1) * (-1) ** index for index, name in enumerate(ARM_JOINTS)}
AXES = {
    name: np.asarray([np.cos(index), np.sin(index), 0.3], dtype=float)
    for index, name in enumerate(ARM_JOINTS)
}
K_ORIGINS = {
    name: np.asarray([0.05 + 0.02 * index, 0.01 * ((-1) ** index), 0.06], dtype=float)
    for index, name in enumerate(ARM_JOINTS)
}
FINGER_LOCAL_OFFSETS = (np.array([0.02, 0.05, 0.0]), np.array([0.02, -0.05, 0.0]))
SWEEP_ANGLES = [
    {
        name: REF_ANGLES[name] + 0.15 * ((index + joint + 1) % 5 - 2)
        for joint, name in enumerate(ARM_JOINTS)
    }
    for index in range(6)
]


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def hom(rotation, translation):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def payload_of(matrix):
    return {
        "world_xyz_m": np.asarray(matrix[:3, 3], dtype=float).tolist(),
        "world_orientation_matrix_row_major": np.asarray(matrix[:3, :3], dtype=float)
        .reshape(-1)
        .tolist(),
        "node_type": "Solid",
        "node_label": "synthetic",
    }


def arm_k(name):
    return hom(rot([0.0, 0.0, 1.0], 0.2), K_ORIGINS[name])


def child_frames(angles):
    """World child Solid frames of every joint at ``angles`` (base = identity)."""
    frames = {TORSO_JOINT: hom(None, TORSO_TRANSLATION)}
    previous = None
    for name in ARM_JOINTS:
        parent = hom(None, TORSO_TRANSLATION + TORSO_TO_ARM1) if name == ARM_JOINTS[0] else previous
        child = parent @ hom(rot(AXES[name], angles[name]), [0.0, 0.0, 0.0]) @ arm_k(name)
        frames[name] = child
        previous = child
    return frames


def build_payload() -> dict:
    frames = child_frames(REF_ANGLES)
    entries = [
        {
            "name": TORSO_JOINT,
            "kind": "slider",
            "axis_source": "unreadable",
            "parent": payload_of(np.eye(4)),
            "child": payload_of(frames[TORSO_JOINT]),
            "limits": {"min_position": 0.0, "max_position": 0.6},
            "joint_parameters": {},
            "readable_axis": None,
            "derived_axis": None,
            "probe": None,
        }
    ]
    previous = None
    for name in ARM_JOINTS:
        parent = hom(None, TORSO_TRANSLATION + TORSO_TO_ARM1) if name == ARM_JOINTS[0] else previous
        child = frames[name]
        probe = (
            parent @ hom(rot(AXES[name], REF_ANGLES[name] + 0.05), [0.0, 0.0, 0.0]) @ arm_k(name)
        )
        entries.append(
            {
                "name": name,
                "kind": "hinge",
                "axis_source": "posture_probe",
                "parent": payload_of(parent),
                "child": payload_of(child),
                "limits": {"min_position": -np.pi, "max_position": np.pi},
                "joint_parameters": {},
                "readable_axis": None,
                "derived_axis": [float(value) for value in AXES[name]],
                "probe": {
                    "commanded_delta": 0.05,
                    "theta0": REF_ANGLES[name],
                    "theta1": REF_ANGLES[name] + 0.05,
                    "theta_delta": 0.05,
                    "child_pose_probe": payload_of(probe),
                    "restore": {"restored": True, "restored_pose_error_m": 0.0},
                },
                "probe_residual": {"rotation_angle_rad": 0.05, "pose_residual_m": 0.0},
            }
        )
        previous = child
    arm7 = frames[ARM_JOINTS[-1]]
    return {
        "schema_version": 1,
        "units": {"length": "m", "angle": "rad"},
        "source": "synthetic",
        "webots_version": "R2025a-test",
        "sim_time_s": 12.5,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "base_frame": payload_of(np.eye(4)),
        "joints": entries,
        "finger_endpoint_solids": {
            "left": payload_of(arm7 @ hom(None, FINGER_LOCAL_OFFSETS[0])),
            "right": payload_of(arm7 @ hom(None, FINGER_LOCAL_OFFSETS[1])),
        },
        "readability_report": {
            "jointParameters": [],
            "posture_probe": list(ARM_JOINTS),
            "unreadable": [TORSO_JOINT],
        },
        "posture_reference": {TORSO_JOINT: TORSO_REF, **REF_ANGLES},
    }


def telemetry_for(
    angles,
    *,
    midpoint_bias: np.ndarray | None = None,
    tool_frame_bias: np.ndarray | None = None,
    finger_bias_per_pose: float = 0.0,
    pose_index: int = 0,
) -> dict[str, object]:
    """GT telemetry for one pose; the finger midpoint equals FK unless biased."""
    frames = child_frames(angles)
    shift = np.zeros(3)
    if midpoint_bias is not None:
        shift = shift + np.asarray(midpoint_bias, dtype=float)
    if tool_frame_bias is not None:
        # A constant offset in the arm_7 frame rotates with the pose, so it is
        # only explainable by the tool-frame parameter family.
        shift = shift + frames[ARM_JOINTS[-1]][:3, :3] @ np.asarray(tool_frame_bias, dtype=float)
    if finger_bias_per_pose:
        shift = shift + np.array([0.0, 0.0, finger_bias_per_pose * ((-1) ** pose_index)])
    arm7 = frames[ARM_JOINTS[-1]]
    fingers = {}
    for label, offset in zip(("left", "right"), FINGER_LOCAL_OFFSETS, strict=True):
        posed = arm7 @ hom(None, offset)
        posed[:3, 3] = posed[:3, 3] + shift
        fingers[label] = payload_of(posed)
    return {
        "schema_version": 1,
        "units": {"length": "m", "angle": "rad"},
        "base_frame": payload_of(np.eye(4)),
        "joint_endpoint_solids": {name: payload_of(frame) for name, frame in frames.items()},
        "joint_sensors_rad": {
            TORSO_JOINT: TORSO_REF,
            **{name: angles[name] for name in ARM_JOINTS},
        },
        "finger_endpoint_solids": fingers,
    }


def make_samples(
    *, midpoint_bias=None, tool_frame_bias=None, per_pose_bias=0.0
) -> list[JointSample]:
    return [
        JointSample(
            joints={name: angles[name] for name in ARM_JOINTS},
            telemetry=telemetry_for(
                angles,
                midpoint_bias=midpoint_bias,
                tool_frame_bias=tool_frame_bias,
                finger_bias_per_pose=per_pose_bias,
                pose_index=index,
            ),
        )
        for index, angles in enumerate(SWEEP_ANGLES)
    ]


@pytest.fixture()
def synthetic_sim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the sim chain loader at the synthetic snapshot."""
    path = tmp_path / "sim_kinematics.json"
    path.write_text(json.dumps(build_payload()), encoding="utf-8")
    monkeypatch.setattr(kinematics_module, "SIM_KINEMATICS_PATH", path)
    return path


def test_sim_model_is_accepted_on_exact_telemetry(synthetic_sim: Path):
    samples = make_samples()
    report = evaluate_sweep(samples)
    sim = report.models["sim"]
    assert sim.accepted is True
    assert sim.finger_mean_m is not None and sim.finger_mean_m < 1e-9
    assert sim.finger_max_m is not None and sim.finger_max_m < 1e-9
    assert all(stats.mean_error_m < 1e-9 for stats in sim.per_link)
    assert report.accepted_model == "sim"
    assert report.refinement is None
    # The URDF baseline is reported (bigger geometry mismatch) but never accepted.
    urdf = report.models["urdf"]
    assert urdf.accepted is False
    assert urdf.finger_mean_m is not None and urdf.finger_mean_m > sim.finger_mean_m
    assert [stats.link for stats in urdf.per_link] == [TORSO_JOINT, *ARM_JOINTS]


def test_constant_bias_rejects_and_proposes_a_rigid_offset(synthetic_sim: Path):
    samples = make_samples(midpoint_bias=np.array([0.0, 0.0, 0.004]))
    report = evaluate_sweep(samples)
    sim = report.models["sim"]
    assert sim.accepted is False
    assert sim.finger_mean_m == pytest.approx(0.004, abs=1e-9)
    assert report.accepted_model is None
    refinement = report.refinement
    assert refinement is not None
    assert refinement.model == "sim"
    parameters = [candidate.parameters for candidate in refinement.candidates]
    assert parameters == ["base_frame_offset_xyz_m", "tool_offset_xyz_m"]
    # A constant base-frame bias is exactly the base-frame parameter family:
    # the held-out residual collapses for it (and only for it).
    assert refinement.candidates[0].holdout_rmse_m < 1e-9
    assert refinement.candidates[1].holdout_rmse_m > 1e-3
    assert refinement.residual_variation_m < 1e-9
    notes = " ".join(refinement.notes)
    assert "cross-validated by pose" in notes
    assert "base_frame_offset_xyz_m family reduces" in notes


def test_tool_frame_bias_is_attributed_to_the_tool_parameter_family(synthetic_sim: Path):
    samples = make_samples(tool_frame_bias=np.array([0.0, 0.0, 0.004]))
    refinement = propose_refinement(samples)
    assert refinement.candidates[0].parameters == "base_frame_offset_xyz_m"
    assert refinement.candidates[0].holdout_rmse_m > 1e-3
    assert refinement.candidates[1].holdout_rmse_m < 1e-9
    assert "tool_offset_xyz_m family reduces" in " ".join(refinement.notes)


def test_pose_dependent_bias_is_not_explained_by_a_rigid_offset(synthetic_sim: Path):
    samples = make_samples(per_pose_bias=0.004)
    refinement = propose_refinement(samples)
    assert refinement.residual_variation_m > 0.002
    notes = " ".join(refinement.notes)
    assert "pose-dependent" in notes
    # Even the best rigid fit leaves most of the held-out error in place.
    assert min(candidate.holdout_rmse_m for candidate in refinement.candidates) > 0.002


def test_evaluate_model_requires_samples_and_complete_telemetry(synthetic_sim: Path):
    with pytest.raises(ValueError, match="At least one sample"):
        evaluate_model([], "sim")
    incomplete = JointSample(
        joints={name: REF_ANGLES[name] for name in ARM_JOINTS},
        telemetry={"base_frame": payload_of(np.eye(4)), "joint_endpoint_solids": {}},
    )
    with pytest.raises(ValueError, match="lacks a complete"):
        evaluate_model([incomplete], "sim")


def test_finger_midpoint_gt_matches_the_tool_frame(synthetic_sim: Path):
    angles = SWEEP_ANGLES[0]
    telemetry = telemetry_for(angles)
    midpoint = finger_midpoint_gt(telemetry)
    assert midpoint is not None
    frames = child_frames(angles)
    from vlarobot.backends.webots.sim_kinematics import build_chain, load_snapshot

    model = load_snapshot(synthetic_sim)
    chain = build_chain(model)
    tool_translation = np.asarray(model.tool_offset[:3, 3], dtype=float)
    expected = frames[ARM_JOINTS[-1]][:3, 3] + frames[ARM_JOINTS[-1]][:3, :3] @ tool_translation
    assert np.allclose(midpoint, expected, atol=1e-12)
    assert chain is not None


def test_diverse_pose_targets_are_deterministic_and_bounded():
    limits = {name: (-1.5, 1.5) for name in ARM_JOINTS}
    first = diverse_pose_targets(n=40, seed=0, limits=limits)
    second = diverse_pose_targets(n=40, seed=0, limits=limits)
    other = diverse_pose_targets(n=40, seed=1, limits=limits)
    assert len(first) == 40
    assert first == second
    assert first != other
    for target in first:
        assert set(target) == set(ARM_JOINTS)
        assert TORSO_JOINT not in target
        for name, value in target.items():
            low, high = limits[name]
            assert low + 0.15 - 1e-9 <= value <= high - 0.15 + 1e-9
    # Stratified sampling covers the workspace rather than clustering.
    for name in ARM_JOINTS:
        values = [target[name] for target in first]
        assert max(values) - min(values) > 2.0


def test_pose_limits_reads_the_snapshot_and_falls_back():
    snapshot = build_payload()
    limits = pose_limits(snapshot)
    assert set(limits) == set(ARM_JOINTS)
    assert limits[ARM_JOINTS[0]] == (-np.pi, np.pi)
    fallback = pose_limits(None)
    assert fallback[ARM_JOINTS[0]] == (-2.0, 2.0)


def test_sample_is_safe_checks_the_table_plane_inside_the_footprint():
    centre_x, centre_y = TABLE_FOOTPRINT_CENTRE_XY_M
    angles = SWEEP_ANGLES[0]
    telemetry = telemetry_for(angles)
    safe, reason = sample_is_safe(telemetry)
    assert safe is True, reason
    unsafe = telemetry_for(angles)
    unsafe_solids = unsafe["joint_endpoint_solids"]
    assert isinstance(unsafe_solids, dict)
    unsafe_solids[ARM_JOINTS[3]] = payload_of(
        hom(None, [centre_x, centre_y, TABLE_TOP_HEIGHT_M + 0.01])
    )
    safe, reason = sample_is_safe(unsafe)
    assert safe is False
    assert ARM_JOINTS[3] in reason and "table footprint" in reason
    outside = telemetry_for(angles)
    outside_solids = outside["joint_endpoint_solids"]
    assert isinstance(outside_solids, dict)
    outside_solids[ARM_JOINTS[3]] = payload_of(hom(None, [centre_x + 100.0, centre_y, 0.0]))
    assert sample_is_safe(outside)[0] is True


def test_format_report_states_the_decision(synthetic_sim: Path):
    accepted = format_report(evaluate_sweep(make_samples()))
    assert any(line.startswith("result=ACCEPT model=sim") for line in accepted)
    rejected = format_report(
        evaluate_sweep(make_samples(midpoint_bias=np.array([0.0, 0.0, 0.004])))
    )
    assert any("result=REJECT" in line for line in rejected)
    assert any("note: cross-validated by pose" in line for line in rejected)


# ---------------------------------------------------------------------------
# CLI sweep driver, exercised against a fake lockstep client (no Webots).
# ---------------------------------------------------------------------------


class _FakeSweepClient:
    """Minimal brain-side client: ``apply`` stores the target, ``step`` reports it."""

    def __init__(self, *, accepted: bool = True, settled: bool = True, telemetry=None) -> None:
        self.accepted = accepted
        self.settled = settled
        self.telemetry = telemetry or (lambda angles: telemetry_for(angles))
        self.target: dict[str, float] = {}

    def apply(self, action):
        if not self.accepted:
            return ApplyResult(False, "joint limit")
        self.target = dict(action.q)
        return ApplyResult(True)

    def step(self, n_steps: int = 1):
        q = dict(self.target)
        if not self.settled:
            q = {name: value + 1.0 for name, value in q.items()}
        angles = {name: q[name] for name in ARM_JOINTS}
        return SimpleNamespace(
            t=0.0,
            robot=SimpleNamespace(
                joint_pos=q,
                diagnostics={"kinematics_telemetry": self.telemetry(angles)},
            ),
        )


def _sweep_client(client: _FakeSweepClient):
    return cast("ZmqBackendClient", client)


def test_drive_pose_collects_a_settled_safe_sample():
    client = _FakeSweepClient()
    target = {name: 0.2 * (index + 1) for index, name in enumerate(ARM_JOINTS)}
    outcome = _drive_pose(
        _sweep_client(client),
        target,
        TORSO_REF,
        max_steps=5,
        settle_tolerance=SETTLE_TOLERANCE_RAD,
    )
    assert outcome.sample is not None, outcome.reason
    assert outcome.sample.joints == pytest.approx(target)
    assert client.target[TORSO_JOINT] == TORSO_REF
    assert sample_is_safe(outcome.sample.telemetry)[0] is True


def test_drive_pose_reports_rejections_and_unsettled_poses():
    target = {name: 0.1 for name in ARM_JOINTS}
    rejected = _drive_pose(
        _sweep_client(_FakeSweepClient(accepted=False)),
        target,
        TORSO_REF,
        max_steps=5,
        settle_tolerance=SETTLE_TOLERANCE_RAD,
    )
    assert rejected.sample is None
    assert rejected.reason is not None and "safety rejected" in rejected.reason

    unsettled = _drive_pose(
        _sweep_client(_FakeSweepClient(settled=False)),
        target,
        TORSO_REF,
        max_steps=3,
        settle_tolerance=SETTLE_TOLERANCE_RAD,
    )
    assert unsettled.sample is None
    assert unsettled.reason is not None and "did not settle" in unsettled.reason


def test_drive_pose_aborts_a_pose_that_fails_the_table_pre_check():
    centre_x, centre_y = TABLE_FOOTPRINT_CENTRE_XY_M

    def unsafe_telemetry(angles):
        telemetry = telemetry_for(angles)
        solids = telemetry["joint_endpoint_solids"]
        assert isinstance(solids, dict)
        solids[ARM_JOINTS[3]] = payload_of(
            hom(None, [centre_x, centre_y, TABLE_TOP_HEIGHT_M + 0.01])
        )
        return telemetry

    outcome = _drive_pose(
        _sweep_client(_FakeSweepClient(telemetry=unsafe_telemetry)),
        {name: 0.1 for name in ARM_JOINTS},
        TORSO_REF,
        max_steps=3,
        settle_tolerance=SETTLE_TOLERANCE_RAD,
    )
    assert outcome.sample is None
    assert outcome.reason is not None and "table pre-check" in outcome.reason
