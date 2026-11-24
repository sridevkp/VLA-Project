from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vlarobot.core.types import JointTarget
from vlarobot.eval.worldmind_geometric_veto import geometric_margins, joint_limit_margin
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor


@pytest.fixture
def config():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/worldmind_arm5_geometric_v2.json"
        ).read_text()
    )


def test_risk_gate_joint_noise_is_bounded(config):
    limits = {"arm_6_joint": (-1.39, 1.39)}
    noise = config["joint_limit_tolerance_rad"]
    assert joint_limit_margin({"arm_6_joint": -1.390000397}, limits, noise) > 0
    assert joint_limit_margin({"arm_6_joint": 1.3902}, limits, noise) < 0
    assert limits["arm_6_joint"] == (-1.39, 1.39)
    monitor = SafetyMonitor(SafetyLimits(joint_position=limits))
    # Position rejection precedes access to the observed state.
    verdict = monitor.validate(JointTarget({"arm_6_joint": -1.390000397}), None)
    assert not verdict.accepted and "position limits" in verdict.reason
    with pytest.raises(ValueError, match="complete finite"):
        joint_limit_margin({"arm_6_joint": float("nan")}, limits, noise)


def test_geometry_preserves_table_and_box_negative_controls(config):
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    starts = np.array([[0.0, 0.0, -0.1]])
    ends = -starts
    half = np.full(3, 0.05)
    far = (starts + 1.0, ends + 1.0, half)
    boxes = dict.fromkeys(("HAM", "APP"), far)
    phases = ["PREGRASP"]
    safe, _ = geometric_margins(config, points, phases, boxes)
    assert min(safe["table"].values()) > 0 and min(safe["objects"].values()) > 0

    boxes["HAM"] = (starts, ends, half)
    obstacle, _ = geometric_margins(config, points, phases, boxes)
    assert obstacle["objects"]["PREGRASP:HAM"] < 0

    points[-1, 2] = config["table_top_world_z_m"] - 0.01
    below, _ = geometric_margins(config, points, phases, boxes)
    assert below["table"]["PREGRASP"] < 0


def test_contact_endpoint_is_distinct_from_path_floor(config):
    stage = "PLACE_DESCEND"
    z = (
        config["table_top_world_z_m"]
        + config["contact_height_m"][stage]["intended_height_above_top_m"]
    )
    points = np.array([[0.0, 0.0, z + 0.12], [0.0, 0.0, z]])
    boxes = dict.fromkeys(("HAM", "APP"), (points, points[:1], np.full(3, 0.05)))
    boxes = {name: (path[:1] + 1, end + 1, half) for name, (path, end, half) in boxes.items()}
    path, endpoint = geometric_margins(config, points, [stage], boxes)
    assert path["table"][stage] >= 0 and endpoint[stage] > 0
    points[-1, 2] += 0.01
    path, endpoint = geometric_margins(config, points, [stage], boxes)
    assert path["table"][stage] >= 0 and endpoint[stage] < 0
