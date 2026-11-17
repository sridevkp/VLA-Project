"""Capture tests for the ADR-0020 Supervisor snapshot (Webots-free).

A fake RobotIO reproduces the exact Supervisor access discipline the exporter
relies on: only Pose-derived nodes (Solid ancestors and endPoint child Solids)
expose ``getPosition``/``getOrientation``; joint nodes raise like the real
``wb_supervisor_node_get_position`` would. The endPoint Solids move with their
joint angle, following the same measured family ``T(θ) = Rot(â, θ) @ K`` as the
loader tests, so the captured snapshot must satisfy the loader contract end to
end.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlarobot.backends.webots import supervisor_kinematics as sk
from vlarobot.backends.webots.backend import WebotsBackend
from vlarobot.backends.webots.sim_kinematics import kinematics_from_payload, snapshot_content_hash

TORSO_REF = 0.35
ARM_ANGLES = {
    "arm_1_joint": 0.2,
    "arm_2_joint": -0.4,
    "arm_3_joint": 0.9,
    "arm_4_joint": 0.1,
    "arm_5_joint": -0.2,
    "arm_6_joint": 0.3,
    "arm_7_joint": -0.5,
}
AXES = {
    "arm_1_joint": np.array([0.0, 0.0, 1.0]),
    "arm_2_joint": np.array([1.0, 0.0, 0.1]),
    "arm_3_joint": np.array([0.0, 1.0, 0.0]),
    "arm_4_joint": np.array([0.0, 1.0, 0.0]),
    "arm_5_joint": np.array([1.0, 0.0, 0.0]),
    "arm_6_joint": np.array([0.0, 0.0, 1.0]),
    "arm_7_joint": np.array([1.0, 0.0, 0.2]),
}


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


# Fixed parent->child transform K per arm hinge (measured at the export posture
# through ``child = parent @ Rot(â, θ) @ K``).
_K_TILTS = {
    "arm_1_joint": ([0.0, 0.1, 0.0], 0.3),
    "arm_4_joint": ([0.0, 0.0, 1.0], 0.4),
    "arm_7_joint": ([0.0, 1.0, 0.0], -0.2),
}
_TARGET_ORIGINS = {
    "arm_1_joint": [0.10, 0.00, 0.05],
    "arm_2_joint": [0.15, 0.00, 0.00],
    "arm_3_joint": [0.20, 0.00, 0.10],
    "arm_4_joint": [0.00, 0.05, 0.18],
    "arm_5_joint": [0.00, 0.00, 0.10],
    "arm_6_joint": [0.00, 0.00, 0.06],
    "arm_7_joint": [0.06, 0.00, -0.02],
}


def _targets() -> dict[str, np.ndarray]:
    return {
        name: hom(rot(*_K_TILTS[name]) if name in _K_TILTS else None, _TARGET_ORIGINS[name])
        for name in AXES
    }


TARGETS = _targets()


def pose_payload(matrix):
    return {
        "world_xyz_m": np.asarray(matrix[:3, 3], dtype=float).tolist(),
        "world_orientation_matrix_row_major": np.asarray(matrix[:3, :3], dtype=float)
        .reshape(-1)
        .tolist(),
        "node_type": "Solid",
        "node_label": "fake",
    }


class _FakeNode:
    """Pose-derived node: pose accessors are legal here."""

    def __init__(self, type_name, pose_matrix, label):
        self._type_name = type_name
        self._pose_matrix = pose_matrix
        self._label = label

    def getTypeName(self):
        return self._type_name

    def getDef(self):
        return self._label

    def getField(self, name):
        return None

    def getPosition(self):
        return np.asarray(self._pose_matrix[:3, 3], dtype=float).tolist()

    def getOrientation(self):
        return np.asarray(self._pose_matrix[:3, :3], dtype=float).reshape(-1).tolist()


class _MovingSolid(_FakeNode):
    """endPoint Solid whose world pose follows the measured joint family."""

    def __init__(self, label, parent_pose, axis, k_matrix, state):
        super().__init__("Solid", None, label)
        self._parent_pose = np.asarray(parent_pose, dtype=float)
        self._axis = axis
        self._k = np.asarray(k_matrix, dtype=float)
        self._state = state

    def _pose(self):
        theta = float(self._state())
        return self._parent_pose @ hom(rot(self._axis, theta), [0.0, 0.0, 0.0]) @ self._k

    def getPosition(self):
        return np.asarray(self._pose()[:3, 3], dtype=float).tolist()

    def getOrientation(self):
        return np.asarray(self._pose()[:3, :3], dtype=float).reshape(-1).tolist()


class _FakeJointNode:
    """HingeJoint/SliderJoint stand-in: poses on the joint itself are illegal."""

    def __init__(self, kind, parent, endpoint, parameters=None):
        self._kind = kind
        self._parent = parent
        self._endpoint = endpoint
        self.parameters = parameters

    def getTypeName(self):
        return self._kind

    def getField(self, name):
        if name == "endPoint":
            return _FakeField(self._endpoint)
        if name == "jointParameters" and self.parameters is not None:
            return _FakeField(self.parameters)
        return None

    def getParentNode(self):
        return self._parent


class _FakeField:
    def __init__(self, node):
        self.node = node

    def getSFNode(self):
        return self.node


class _FakeParametersRead:
    def __init__(self, axis):
        self.axis = axis

    def getField(self, name):
        if name != "axis":
            return None
        return _FakeVec3(self.axis)


class _FakeVec3:
    def __init__(self, values):
        self.values = values

    def getSFVec3f(self):
        return list(self.values)


class _FakeSolid(_FakeNode):
    def getParentNode(self):
        return None


class _FakeMotor:
    def __init__(self, name):
        self.name = name
        self.target = 0.0
        self.low, self.high = (-np.pi, np.pi)
        self.position_calls = []

    def getName(self):
        return self.name

    def setPosition(self, value):
        self.position_calls.append(float(value))
        self.target = float(value)

    def getMinPosition(self):
        return self.low

    def getMaxPosition(self):
        return self.high


class _FakeSensor:
    def __init__(self, value):
        self.value = value

    def getValue(self):
        return self.value


class _FakeRobot:
    def getVersion(self):
        return "R2025a-synthetic"

    def getTime(self):
        return 12.5


def build_world():
    """Static world frames: Robot node + per-joint parent Solids (export pose)."""
    world = {"robot": np.eye(4), "torso_child": hom(None, [0.02, 0.0, 0.34])}
    arm1_parent = world["torso_child"] @ hom(None, [0.0, 0.0, 0.02])
    world["arm_1_joint.parent"] = arm1_parent
    previous = None
    for name, axis in AXES.items():
        parent = world["arm_1_joint.parent"] if name == "arm_1_joint" else previous
        world[f"{name}.parent"] = parent
        previous = parent @ hom(rot(axis, ARM_ANGLES[name]), [0.0, 0.0, 0.0]) @ TARGETS[name]
    return world


def build_fake_io(world, joint_parameters=None):
    """Fake RobotIO with kinematically consistent, moving endPoint Solids.

    ``joint_parameters`` maps an arm joint name to a readable axis (list of 3
    floats) exposed through ``jointParameters``; other joints carry none.
    """

    class _FakeIO:
        def __init__(self):
            self.robot = _FakeRobot()
            self.me = _FakeNode("Robot", world["robot"], "RobotNode")
            self.motors: dict[str, _FakeMotor] = {}
            self.sensors: dict[str, _FakeSensor] = {}
            self._joint_map: dict[str, object] = {}
            self.fingers: list[tuple[_FakeMotor, _FakeSensor]] = []
            self.torso_name = "torso_lift_joint"
            self.arm_names = list(AXES)
            self.step_calls = 0
            self._build("torso_lift_joint", "SliderJoint")
            for name in AXES:
                self._build(name, "HingeJoint")

        def _build(self, name, kind):
            motor = _FakeMotor(name)
            reference = TORSO_REF if kind == "SliderJoint" else ARM_ANGLES[name]
            sensor = _FakeSensor(reference)
            # Unset motors hold the export posture in the real robot; model the
            # same here so probing one joint never disturbs the others.
            motor.target = reference
            if kind == "SliderJoint":
                parent_solid = _FakeSolid("Robot", world["robot"], "torso.parentSolid")
                endpoint = _TorsoEndpoint(world["robot"], lambda: motor.target)
            else:
                parent_solid = _FakeSolid("Solid", world[f"{name}.parent"], f"{name}.parentSolid")
                endpoint = _MovingSolid(
                    f"{name}.endPoint",
                    world[f"{name}.parent"],
                    AXES[name],
                    TARGETS[name],
                    lambda: motor.target,
                )
            axis = (joint_parameters or {}).get(name)
            parameters = _FakeParametersRead(axis) if axis is not None else None
            self.motors[name] = motor
            self.sensors[name] = sensor
            self._joint_map[name] = _FakeJointNode(kind, parent_solid, endpoint, parameters)

        def _joint_node(self, name):
            return self._joint_map.get(name)

        def step_raw(self, count=1):
            self.step_calls += count
            for name, motor in self.motors.items():
                self.sensors[name].value = motor.target
            return True

    return _FakeIO()


class _TorsoEndpoint(_FakeNode):
    """Torso endPoint Solid: child = robot @ Trans(0.02, 0, θ - 0.01)."""

    def __init__(self, robot_pose, state):
        super().__init__("Solid", None, "torso.endPoint")
        self._robot = np.asarray(robot_pose, dtype=float)
        self._state = state

    def _pose(self):
        theta = float(self._state())
        return self._robot @ hom(None, [0.02, 0.0, theta - 0.01])

    def getPosition(self):
        return np.asarray(self._pose()[:3, 3], dtype=float).tolist()

    def getOrientation(self):
        return np.asarray(self._pose()[:3, :3], dtype=float).reshape(-1).tolist()


def capture(io, **kwargs):
    order = (io.torso_name, *io.arm_names)
    return sk.capture_snapshot(io, order, **kwargs)


def test_snapshot_captures_all_joints_and_endpoints():
    io = build_fake_io(build_world())
    snapshot = capture(io)
    names = [entry["name"] for entry in snapshot["joints"]]
    assert names == [
        "torso_lift_joint",
        "arm_1_joint",
        "arm_2_joint",
        "arm_3_joint",
        "arm_4_joint",
        "arm_5_joint",
        "arm_6_joint",
        "arm_7_joint",
    ]
    assert snapshot["base_frame"]["world_xyz_m"] == [0.0, 0.0, 0.0]
    assert snapshot["base_frame"]["node_label"].startswith("RobotNode")
    assert snapshot["webots_version"] == "R2025a-synthetic"
    assert snapshot["posture_reference"]["torso_lift_joint"] == TORSO_REF
    report = snapshot["readability_report"]
    assert report["posture_probe"] == list(AXES)
    assert report["unreadable"] == ["torso_lift_joint"]
    for entry in snapshot["joints"]:
        assert entry["parent"]["world_xyz_m"] is not None
        assert entry["child"]["world_orientation_matrix_row_major"] is not None


def test_snapshot_derives_axes_from_the_probe():
    io = build_fake_io(build_world())
    snapshot = capture(io)
    for entry in snapshot["joints"]:
        if entry["kind"] == "slider":
            # The torso is baked at its export posture: no redundant probe.
            assert entry["derived_axis"] is None
            assert io.motors[entry["name"]].position_calls == []
            continue
        axis = np.asarray(entry["derived_axis"], dtype=float)
        expected = AXES[entry["name"]] / np.linalg.norm(AXES[entry["name"]])
        assert float(abs(np.dot(axis, expected))) > 1.0 - 1e-9, entry["name"]
        assert entry["probe_residual"]["pose_residual_m"] < 1e-6, entry["name"]


def test_probe_moves_within_tolerance_and_restores():
    io = build_fake_io(build_world())
    snapshot = capture(io)
    for name in AXES:
        motor = io.motors[name]
        theta0 = ARM_ANGLES[name]
        commanded = motor.position_calls
        assert len(commanded) >= 2
        assert abs(abs(commanded[0] - theta0) - 0.05) < 1e-9
        assert abs(commanded[-1] - theta0) < 1e-12
        probe = next(e for e in snapshot["joints"] if e["name"] == name)["probe"]
        assert probe["restore"]["restored"] is True
        assert probe["restore"]["restored_pose_error_m"] < 1e-6


def test_probe_disabled_reports_unreadable_without_motion():
    io = build_fake_io(build_world())
    snapshot = capture(io, probe_axes=False)
    report = snapshot["readability_report"]
    assert report["posture_probe"] == []
    assert len(report["unreadable"]) == 8
    for entry in snapshot["joints"]:
        assert entry["derived_axis"] is None
        assert io.motors[entry["name"]].position_calls == []


def test_readable_axis_is_cross_checked_only():
    readable_axes = {
        "arm_1_joint": [0.0, 0.0, 1.0],  # agrees -> 0 deg
        "arm_2_joint": [0.0, 1.0, 0.0],  # disagrees -> caught, unused
    }
    io = build_fake_io(build_world(), joint_parameters=readable_axes)
    snapshot = capture(io)
    for entry in snapshot["joints"]:
        if entry["kind"] != "hinge":
            continue
        assert entry["axis_source"] == "posture_probe"
        if entry["name"] in readable_axes:
            assert entry["readable_axis"] == readable_axes[entry["name"]]
            angle = entry["axis_cross_check"]["angle_between_deg"]
            if entry["name"] == "arm_1_joint":
                assert angle == pytest.approx(0.0, abs=1e-6)
            else:
                assert angle > 1.0
    assert "jointParameters" in snapshot["readability_report"]


def test_capture_snapshot_feeds_the_loader():
    """End to end: the serialized captured JSON must satisfy the loader."""
    io = build_fake_io(build_world())
    snapshot = capture(io)
    reloaded = json.loads(sk.to_json_bytes(snapshot).decode("utf-8"))
    reloaded["valid"] = True
    reloaded["content_hash"] = snapshot_content_hash(reloaded)
    model = kinematics_from_payload(reloaded, source="<capture>")
    axes = {joint.name: joint.axis for joint in model.joints}
    for name, axis in AXES.items():
        expected = axis / np.linalg.norm(axis)
        assert float(abs(np.dot(axes[name], expected))) > 1.0 - 1e-9
    # The torso transform must reproduce the measured Robot->torso_child pose.
    expected_torso = np.asarray([[1.0, 0, 0, 0.02], [0, 1, 0, 0], [0, 0, 1, 0.34], [0, 0, 0, 1]])
    assert np.allclose(model.torso_transform, expected_torso)


def test_telemetry_method_on_bare_backend_maps_joint_order():
    io = build_fake_io(build_world())
    backend = WebotsBackend.__new__(WebotsBackend)
    backend.io = io
    telemetry = backend._kinematics_telemetry()
    endpoint_solids = telemetry["joint_endpoint_solids"]
    sensors = telemetry["joint_sensors_rad"]
    base_frame = telemetry["base_frame"]
    assert isinstance(endpoint_solids, dict)
    assert isinstance(sensors, dict)
    assert isinstance(base_frame, dict)
    assert set(endpoint_solids) == {"torso_lift_joint", *AXES}
    assert sensors["arm_1_joint"] == ARM_ANGLES["arm_1_joint"]
    # node_label composes DEF name + node type: 'RobotNode/Robot'.
    label = base_frame.get("node_label")
    assert isinstance(label, str) and label.startswith("RobotNode")


def test_missing_joint_node_is_reported_unreadable():
    io = build_fake_io(build_world())
    io._joint_map["arm_2_joint"] = None
    snapshot = capture(io)
    entry = next(e for e in snapshot["joints"] if e["name"] == "arm_2_joint")
    assert entry["axis_source"] == "unreadable"
    # The torso slider is baked (not probed), so it lands in 'unreadable' too.
    assert snapshot["readability_report"]["unreadable"] == ["torso_lift_joint", "arm_2_joint"]
