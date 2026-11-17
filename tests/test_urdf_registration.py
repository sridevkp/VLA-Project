"""A static registration must affect FK and IK consistently and never require live GT."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from vlarobot.backends.webots import kinematics
from vlarobot.backends.webots.config import SETTINGS
from vlarobot.backends.webots.urdf_registration import load_registration

CALIBRATION = Path(__file__).parents[1] / "configs/calibration/urdf-base-train-v1.yaml"


def test_registration_predicts_independently_recorded_pad_geometry(monkeypatch):
    legacy = kinematics.Kinematics()
    monkeypatch.setattr(
        kinematics, "SETTINGS", replace(SETTINGS, urdf_base_registration_file=str(CALIBRATION))
    )
    corrected = kinematics.Kinematics()
    fixture = Path(__file__).parent / "fixtures/urdf_registration_holdout.json"
    measurements = json.loads(fixture.read_text(encoding="utf-8"))["samples"]
    for sample in measurements:
        errors = []
        for model in (legacy, corrected):
            q = np.zeros(len(model.chain.links))
            for name, angle in sample["joint_pos"].items():
                q[model.idx[name]] = angle
            errors.append(
                float(np.linalg.norm(model.forward(q)[:3, 3] - sample["measured_pad_base_xyz_m"]))
            )
        assert errors[0] > 0.03
        assert errors[1] < 0.002


def test_registration_transforms_fk_and_inverse_target_consistently(monkeypatch):
    legacy = kinematics.Kinematics()
    monkeypatch.setattr(
        kinematics, "SETTINGS", replace(SETTINGS, urdf_base_registration_file=str(CALIBRATION))
    )
    corrected = kinematics.Kinematics()
    q = np.zeros(len(legacy.chain.links))
    for name in kinematics.ARM_JOINTS:
        lo, hi = legacy.bounds(name)
        q[legacy.idx[name]] = (lo + hi) / 2
    q[legacy.idx["torso_lift_joint"]] = 0.35
    cq = np.zeros(len(corrected.chain.links))
    for name, index in legacy.idx.items():
        if name in corrected.idx:
            cq[corrected.idx[name]] = q[index]
    with CALIBRATION.open(encoding="utf-8") as stream:
        transform = np.asarray(yaml.safe_load(stream)["T_robot_base_urdf_root"])
    expected = transform @ legacy.forward(q)
    np.testing.assert_allclose(corrected.forward(cq), expected, atol=1e-12)
    # IK operates on the registered chain itself, so no target-only offset can be
    # applied a second time. Use an exactly reachable pose with the same orientation.
    solved = corrected.chain.inverse_kinematics(
        expected[:3, 3],
        target_orientation=expected[:3, :3],
        orientation_mode="all",
        initial_position=cq,
    )
    np.testing.assert_allclose(corrected.forward(solved), expected, atol=1e-7)
    for name in kinematics.ARM_JOINTS:
        assert corrected.bounds(name) == legacy.bounds(name)


@pytest.mark.parametrize("bad", ["reflection", "nonfinite", "scale", "wrong_frame"])
def test_registration_rejects_invalid_geometry_or_provenance(tmp_path, bad):
    matrix = np.eye(4)
    if bad == "reflection":
        matrix[0, 0] = -1
    elif bad == "scale":
        matrix[0, 0] = 2
    elif bad == "nonfinite":
        matrix[0, 3] = np.nan
    value = {
        "source_split": "test" if bad == "wrong_frame" else "train",
        "evidence": "recorded",
        "T_robot_base_urdf_root": matrix.tolist(),
    }
    path = tmp_path / "registration.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registration(path)
