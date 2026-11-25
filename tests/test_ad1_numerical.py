"""A-D1 helper regressions: analytic fixtures only, zero real numerical IK calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "ad1_helpers", Path(__file__).parents[1] / "tools/numerical/ad1.py"
)
ad1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad1)


def test_task_null_split_reconstructs_and_is_invariant_to_nonsingular_row_units():
    j = np.c_[np.eye(6), np.zeros(6)]
    delta = np.arange(1.0, 8.0)
    before = ad1.decompose(j, delta)
    scaled = ad1.decompose(np.diag([1000.0] * 3 + [1.0] * 3) @ j, delta)
    assert before["numerical_rank"] == 6
    np.testing.assert_allclose(before["task_component_rad"], [1, 2, 3, 4, 5, 6, 0])
    np.testing.assert_allclose(before["null_component_rad"], [0, 0, 0, 0, 0, 0, 7])
    np.testing.assert_allclose(before["null_component_rad"], scaled["null_component_rad"])
    assert before["reconstruction_error_rad"] == 0
    np.testing.assert_array_equal(before["null_task_residual"], np.zeros(6))


def test_rank_zero_is_reported_without_division_or_invented_rank():
    result = ad1.decompose(np.zeros((6, 7)), np.arange(7.0))
    assert result["numerical_rank"] == 0
    np.testing.assert_array_equal(result["null_component_rad"], np.arange(7.0))


def test_registered_geometric_jacobian_revolute_fixture():
    theta = 0.3
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    joint = np.eye(4)
    joint[:3, :3] = rotation
    end = joint.copy()
    end[:3, 3] = rotation @ np.array([1.0, 0, 0])
    links = [
        SimpleNamespace(),
        SimpleNamespace(joint_type="revolute", rotation=np.array([0, 0, 1])),
    ]
    chain = SimpleNamespace(
        links=links, forward_kinematics=lambda *a, **kw: [np.eye(4), joint, end]
    )
    kin = SimpleNamespace(chain=chain, idx={"arm": 1})
    actual = ad1.geometric_jacobian(kin, np.zeros(2), ("arm",))
    np.testing.assert_allclose(actual[:, 0], [-np.sin(theta), np.cos(theta), 0, 0, 0, 1])


def test_new_evidence_never_overwrites_and_sha_rejects_symlink(tmp_path):
    output = tmp_path / "report.json"
    ad1.write(output, {"value": 1})
    with pytest.raises(FileExistsError):
        ad1.write(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}
    link = tmp_path / "alias.json"
    try:
        link.symlink_to(output)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this test environment")
    with pytest.raises(ValueError, match="symlinked"):
        ad1.sha(link)
    with pytest.raises(ValueError, match="provenance mismatch"):
        ad1.checked(output, "0" * 64)
