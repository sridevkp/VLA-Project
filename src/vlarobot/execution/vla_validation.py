"""Read-only integration of a real installed-policy output with registered IK and safety."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.webots.backend import _rotation_to_quaternion
from vlarobot.backends.webots.config import ARM_JOINTS, SAFETY_PROFILE, SETTINGS, URDF_PATH
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.pose_solver import BoundedPoseSolver
from vlarobot.core.types import Pose2D, Pose3D, RobotState, Vec3
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.vla.action_adapter import ActionAdapter, as_quaternion, pose_delta
from vlarobot.vla.settings import load_vla_settings


def validate_recorded_adapter(forward_report: Path, recording: Path, out: Path) -> dict[str, Any]:
    """Check normalized-model physical output and a reachable positive control without apply()."""
    report = json.loads(forward_report.read_text(encoding="utf-8"))
    kin = Kinematics()
    meta = json.loads((recording / "episode-00.meta.json").read_text(encoding="utf-8"))
    if kin.registration_sha256 != meta["outcome"]["geometry_registration_sha256"]:
        raise ValueError("Adapter kinematic registration differs from the recorded policy")
    with np.load(recording / "episode-00.steps.npz", allow_pickle=False) as samples:
        joints = dict(
            zip(samples["joint_names"].tolist(), map(float, samples["joint_pos"][0]), strict=True)
        )
        robot = RobotState(
            joints,
            {},
            Pose2D(*map(float, samples["base_pose"][0])),
            Pose3D(Vec3(*map(float, samples["ee_xyz"][0])), as_quaternion(samples["ee_quat"][0])),
            float(samples["gripper"][0]),
        )
    xml = ET.parse(URDF_PATH)
    velocities = {}
    for name in ARM_JOINTS:
        limit = xml.find(f"./joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"Missing URDF velocity limit for {name}")
        velocities[name] = float(limit.attrib["velocity"])
    safety = SafetyMonitor(
        SafetyLimits(
            joint_position={n: kin.bounds(n) for n in ARM_JOINTS},
            joint_velocity=velocities,
            workspace_min=SAFETY_PROFILE.workspace_min,
            workspace_max=SAFETY_PROFILE.workspace_max,
            gripper_opening=SETTINGS.gripper_opening_limits,
            command_horizon=SETTINGS.safety_command_horizon,
        )
    )
    settings = load_vla_settings()
    adapter = ActionAdapter(
        BoundedPoseSolver(kin, max_joint_step_rad=SETTINGS.max_joint_step_rad),
        safety,
        position_tolerance_m=settings.ik_position_tolerance_m,
        orientation_tolerance_rad=settings.ik_orientation_tolerance_rad,
    )
    q = np.zeros(len(kin.chain.links))
    for name, value in joints.items():
        q[kin.idx[name]] = value
    target_q = q.copy()
    # A deterministic positive control comfortably inside the configured trust region.
    target_q[kin.idx["arm_3_joint"]] += SETTINGS.max_joint_step_rad / 10
    T = kin.forward(target_q)
    target = Pose3D(Vec3(*map(float, T[:3, 3])), _rotation_to_quaternion(T[:3, :3]))
    known_action = np.r_[pose_delta(robot.ee_pose, target), SETTINGS.gripper_opening_limits[0]]
    positive = adapter.decode(known_action, robot)
    raw = np.asarray(report["decoded_action"][0], dtype=float)
    try:
        model = adapter.decode(raw, robot)
        model_result = {
            "accepted": True,
            "position_error_m": model.position_error_m,
            "orientation_error_rad": model.orientation_error_rad,
        }
    except (ValueError, RuntimeError) as error:
        model_result = {"accepted": False, "reason": str(error)}
    result = {
        "status": "MEASURED real pretrained physical output decoded by bounded pose IK/safety",
        "source_forward_report": str(forward_report),
        "model_output_result": model_result,
        "positive_control": {
            "accepted": True,
            "position_error_m": positive.position_error_m,
            "orientation_error_rad": positive.orientation_error_rad,
        },
        "registration_sha256": kin.registration_sha256,
        "robot_commands_sent": 0,
        "orientation_convention": "base-frame left-multiplied rotation-vector delta (radians)",
        "joint_velocity_limit_source": (
            "repository URDF; live motor cross-check required before VLA evaluation"
        ),
    }
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result
