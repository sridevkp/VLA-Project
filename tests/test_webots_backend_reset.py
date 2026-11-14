import math
from typing import Any, cast

import numpy as np
import pytest

from vlarobot.backends.webots.backend import WebotsBackend
from vlarobot.backends.webots.config import ARM_JOINTS, CAN_DEF, TORSO_JOINT
from vlarobot.backends.webots.frames import Frames
from vlarobot.backends.webots.gripper import Gripper
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.robot_io import RobotIO
from vlarobot.core.types import ScenarioSpec


class FakeField:
    def __init__(self, value):
        self.value = list(value)

    def getSFVec3f(self):
        return list(self.value)

    def getSFRotation(self):
        return list(self.value)

    def setSFVec3f(self, value):
        self.value = list(value)

    def setSFRotation(self, value):
        self.value = list(value)


class FakeCan:
    def __init__(self, translation, rotation):
        self.fields = {
            "translation": FakeField(translation),
            "rotation": FakeField(rotation),
        }

    def getField(self, name):
        return self.fields[name]

    def getPosition(self):
        return self.fields["translation"].getSFVec3f()

    def getOrientation(self):
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


class FakeSensor:
    def __init__(self, value):
        self.value = value

    def getValue(self):
        return self.value


class FakeMotor:
    def __init__(self, name, sensor, minimum=0.0, maximum=10.0):
        self.name = name
        self.sensor = sensor
        self.minimum = minimum
        self.maximum = maximum
        self.position_field = FakePositionField()

    def getName(self):
        return self.name

    def setPosition(self, position):
        self.sensor.value = position

    def getMinPosition(self):
        return self.minimum

    def getMaxPosition(self):
        return self.maximum


class FakePositionField:
    def __init__(self):
        self.value = None

    def getSFFloat(self):
        return self.value

    def setSFFloat(self, value):
        self.value = value


class FakePositionRobotIO(RobotIO):
    def __init__(self, motor):
        self.motors = {motor.getName(): motor}
        self.fingers = []
        self.position_field = motor.position_field

    def _joint_position_field(self, motor: Any, name: str) -> Any:
        assert name == motor.getName()
        return self.position_field


def test_reset_baseline_uses_nearest_limit_for_out_of_range_joint_field(caplog):
    name = "torso_lift_joint"
    motor = FakeMotor(name, FakeSensor(-0.000172), minimum=0.0, maximum=0.35)
    motor.position_field.value = -0.000172
    io = FakePositionRobotIO(motor)

    assert io.joint_positions_from_fields() == {name: 0.0}
    assert "using nearest legal limit 0.0" in caplog.text


def test_contact_tracking_reports_external_node_and_world_contact_points():
    class FakeNode:
        def __init__(self, node_id, def_name, type_name, parent=None):
            self.node_id = node_id
            self.def_name = def_name
            self.type_name = type_name
            self.parent = parent
            self.contact_points = []

        def getId(self):
            return self.node_id

        def getDef(self):
            return self.def_name

        def getTypeName(self):
            return self.type_name

        def getField(self, name):
            return None

        def getParentNode(self):
            return self.parent

        def getContactPoints(self, **_kwargs):
            return list(self.contact_points)

    class FakeContact:
        def getNodeId(self):
            return 22

        def getPoint(self):
            return [-1.2, -4.1, 0.73]

    class FakeSupervisor:
        def getFromId(self, node_id):
            return external if node_id == 22 else None

    robot_node = FakeNode(11, "TIAGo", "Robot")
    external = FakeNode(22, "TABLE", "Solid")
    io = RobotIO.__new__(RobotIO)
    io.me = robot_node
    io.robot = FakeSupervisor()
    robot_node.contact_points = [FakeContact()]

    assert io.contact_nodes() == [
        {
            "node_id": 22,
            "node_name": "TABLE",
            "node_type": "Solid",
            "contact_points_world_xyz_m": [[-1.2, -4.1, 0.73]],
        }
    ]


class FakeRobot:
    def __init__(self, can_node, me, objects=None):
        self.can_node = can_node
        self.me = me
        self.time = 10.0
        self.lookup_count = 0
        self.nodes = {CAN_DEF: can_node, **{name: node for name, node in (objects or {}).items()}}

    def getFromDef(self, name):
        self.lookup_count += 1
        return self.nodes.get(name)

    def simulationResetPhysics(self):
        self.can_node = FakeCan(
            self.can_node.getField("translation").getSFVec3f(),
            self.can_node.getField("rotation").getSFRotation(),
        )
        self.nodes[CAN_DEF] = self.can_node

    def step(self, time_step):
        self.time += time_step / 1000.0
        return 0

    def getTime(self):
        return self.time


class FakeCamera:
    def getImageArray(self):
        return np.zeros((2, 3, 3), dtype=np.uint8).tolist()

    def getFov(self):
        return 1.0


class FakeKinematics:
    def forward(self, _q):
        return np.eye(4)


class FakeFrames:
    def current_q(self):
        return np.zeros(7)


class FakeGripper:
    last_close_stalled = True
    last_close_opening = 0.1

    def __init__(self, fingers):
        self.fingers = fingers

    def opening(self):
        return 0.02


class FakeMe:
    def __init__(self):
        self.fields = {
            "translation": FakeField([0.1, 0.2, 0.3]),
            "rotation": FakeField([0.0, 0.0, 1.0, 0.0]),
        }

    def getField(self, name):
        return self.fields[name]

    def getPosition(self):
        return self.fields["translation"].getSFVec3f()

    def getOrientation(self):
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _make_backend(obstacles=None):
    joint_names = ARM_JOINTS + [TORSO_JOINT]
    sensors = {name: FakeSensor(float(index)) for index, name in enumerate(joint_names)}
    motors = {name: FakeMotor(name, sensors[name]) for name in joint_names}
    fingers = []
    for index in range(2):
        sensor = FakeSensor(0.01 * (index + 1))
        name = f"finger_{index}"
        fingers.append((FakeMotor(name, sensor, 0.0, 0.045), sensor))

    can_node = FakeCan([1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 0.0])
    me = FakeMe()
    robot = FakeRobot(can_node, me, objects=obstacles)

    class FakeIO:
        def __init__(self):
            self.robot = robot
            self.ts = 32
            self.sensors = sensors
            self.motors = motors
            self.fingers = fingers
            self.me = me
            self.joint_positions = {**motors, **{motor.getName(): motor for motor, _ in fingers}}

        def set_joint_positions(self, positions):
            for name, position in positions.items():
                motor = self.joint_positions[name]
                motor.position_field.setSFFloat(position)
                motor.setPosition(position)

        def apply_hold_base(self):
            return None

    backend = WebotsBackend.__new__(WebotsBackend)
    backend.io = cast(RobotIO, FakeIO())
    backend.can_node = can_node
    backend.camera = FakeCamera()
    backend.kinematics = cast(Kinematics, FakeKinematics())
    backend.frames = cast(Frames, FakeFrames())
    backend.gripper = cast(Gripper, FakeGripper(fingers))
    backend._initial_joint_pos = {name: sensor.getValue() for name, sensor in sensors.items()}
    backend._initial_finger_positions = [sensor.getValue() for _, sensor in fingers]
    backend._initial_can_translation = [1.0, 2.0, 3.0]
    backend._initial_can_rotation = [0.0, 0.0, 1.0, 0.0]
    backend._initial_base_translation = me.getField("translation").getSFVec3f()
    backend._initial_base_rotation = me.getField("rotation").getSFRotation()
    backend._last_joint_pos = {}
    backend._last_time = 0.0
    backend._episode_start_time = 0.0
    backend._stopped = False
    return backend


def _observation_signature(observation):
    return (
        observation.t,
        observation.rgb.copy(),
        observation.cam_info,
        dict(observation.robot.joint_pos),
        dict(observation.robot.joint_vel),
        observation.robot.base_pose,
        observation.robot.ee_pose,
        observation.robot.gripper_opening,
    )


def test_reset_twice_reacquires_can_and_restores_identical_episode_state():
    backend = _make_backend()
    scenario = ScenarioSpec()

    first_observation = backend.reset(scenario, seed=0)
    first_ground_truth = backend.get_ground_truth()

    for index, name in enumerate(ARM_JOINTS + [TORSO_JOINT]):
        backend.io.sensors[name].value = 20.0 + index
    backend.io.fingers[0][1].value = 0.04
    backend.io.robot.can_node.getField("translation").setSFVec3f([8.0, 9.0, 10.0])
    backend.io.me.getField("translation").setSFVec3f([4.0, 5.0, 6.0])

    second_observation = backend.reset(scenario, seed=1)
    second_ground_truth = backend.get_ground_truth()

    first = _observation_signature(first_observation)
    second = _observation_signature(second_observation)
    assert first[0] == second[0] == 0.0
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    _assert_mapping_close(first[3], second[3], tolerance=1e-3)
    _assert_mapping_close(first[4], second[4], tolerance=1e-3)
    assert abs(first[5].x - second[5].x) <= 1e-3
    assert abs(first[5].y - second[5].y) <= 1e-3
    assert abs(first[5].yaw - second[5].yaw) <= 1e-3
    _assert_pose3d_close(first[6], second[6], tolerance=1e-3)
    assert abs(first[7] - second[7]) <= 1e-3
    _assert_pose3d_close(
        first_ground_truth.object_poses[CAN_DEF],
        second_ground_truth.object_poses[CAN_DEF],
        tolerance=1e-3,
    )
    assert backend.can_node is backend.io.robot.can_node
    assert backend.io.me.getField("translation").getSFVec3f() == [0.1, 0.2, 0.3]
    assert all(
        backend.io.joint_positions[name].position_field.value == position
        for name, position in backend._initial_joint_pos.items()
        if name != TORSO_JOINT
    )
    assert backend.io.joint_positions[TORSO_JOINT].position_field.value == 0.35
    assert backend.io.robot.lookup_count == 2
    assert backend.gripper.last_close_stalled is None
    assert backend.gripper.last_close_opening is None


def test_reset_applies_scenario_can_xy_yaw_offset():
    """ADR-0025: the scenario offset moves/rotates the CAN at reset, composed from the
    world-file pose, and a later default reset restores it."""
    backend = _make_backend()
    backend.reset(ScenarioSpec(name="grid", can_xy_yaw_offset=(0.05, -0.05, math.pi / 2)), seed=0)

    translation = backend.can_node.getField("translation").getSFVec3f()
    rotation = backend.can_node.getField("rotation").getSFRotation()
    np.testing.assert_allclose(translation, [1.05, 1.95, 3.0], atol=1e-9)
    np.testing.assert_allclose(rotation, [0.0, 0.0, 1.0, math.pi / 2], atol=1e-9)
    can_pose = backend.get_ground_truth().object_poses[CAN_DEF]
    np.testing.assert_allclose(can_pose.position.as_array(), [1.05, 1.95, 3.0], atol=1e-9)

    backend.reset(ScenarioSpec(), seed=1)
    translation = backend.can_node.getField("translation").getSFVec3f()
    np.testing.assert_allclose(translation, [1.0, 2.0, 3.0], atol=1e-9)
    rotation = backend.can_node.getField("rotation").getSFRotation()
    np.testing.assert_allclose(rotation, [0.0, 0.0, 1.0, 0.0], atol=1e-9)


def _assert_mapping_close(first, second, tolerance):
    assert first.keys() == second.keys()
    for key in first:
        assert abs(first[key] - second[key]) <= tolerance


def _assert_pose3d_close(first, second, tolerance):
    position_delta = np.linalg.norm(first.position.as_array() - second.position.as_array())
    assert position_delta <= tolerance
    orientation_dot = abs(np.dot(first.orientation, second.orientation))
    orientation_delta = 2.0 * np.arccos(min(1.0, orientation_dot))
    assert orientation_delta <= tolerance


class _FakeEndpointField:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def getSFNode(self):
        return self.endpoint


class _FakeJointNode:
    """HingeJoint stand-in; ``getPosition`` raises like the real Supervisor."""

    def __init__(self, endpoint=None, position_error=None):
        self.endpoint = endpoint
        self.position_error = position_error
        self.position_calls = 0

    def getTypeName(self):
        return "HingeJoint"

    def getPosition(self):
        self.position_calls += 1
        raise self.position_error or RuntimeError(
            "wb_supervisor_node_get_position() can exclusively be used with Pose"
        )

    def getField(self, name):
        return _FakeEndpointField(self.endpoint) if name == "endPoint" else None


class _FakeEndpointNode:
    def __init__(self, position=None, error=None, orientation=None):
        self.position = position
        self.error = error
        self.orientation = orientation
        self.position_calls = 0
        self.orientation_calls = 0

    def getPosition(self):
        self.position_calls += 1
        if self.error is not None:
            raise self.error
        assert self.position is not None
        return list(self.position)

    def getOrientation(self):
        self.orientation_calls += 1
        if self.error is not None:
            raise self.error
        assert self.orientation is not None
        return list(self.orientation)


class _FakeTaggedMotor:
    def __init__(self, tag):
        self._tag = tag


class _FakeDeviceRobot:
    def __init__(self, joint):
        self.joint = joint

    def getFromDevice(self, _tag):
        return self.joint


def _make_joint_io(joint):
    io = RobotIO.__new__(RobotIO)
    io.robot = _FakeDeviceRobot(joint)
    io.motors = {"arm_7_joint": _FakeTaggedMotor(7)}
    io._lookup_failures = {}
    return io


def test_joint_endpoint_uses_the_pose_derived_child_solid(caplog):
    joint = _FakeJointNode(endpoint=_FakeEndpointNode([9.0, 9.0, 9.0]))
    io = _make_joint_io(joint)

    assert io.joint_endpoint_world_position("arm_7_joint") == [9.0, 9.0, 9.0]
    assert joint.position_calls == 0
    assert "getPosition" not in caplog.text


def test_joint_endpoint_returns_none_and_warns_once_for_missing_endpoint(caplog):
    joint = _FakeJointNode(endpoint=None)
    io = _make_joint_io(joint)

    assert io.joint_endpoint_world_position("arm_7_joint") is None
    assert io.joint_endpoint_world_position("arm_7_joint") is None
    assert caplog.text.count("has no endPoint Solid") == 1
    assert joint.position_calls == 0


def test_joint_endpoint_returns_none_and_caches_a_non_pose_position_failure(caplog):
    endpoint = _FakeEndpointNode(error=RuntimeError("not Pose-derived"))
    joint = _FakeJointNode(endpoint=endpoint)
    io = _make_joint_io(joint)

    assert io.joint_endpoint_world_position("arm_7_joint") is None
    assert io.joint_endpoint_world_position("arm_7_joint") is None
    assert endpoint.position_calls == 1
    assert caplog.text.count("endPoint position lookup failed") == 1


def test_joint_endpoint_returns_none_for_unknown_joint(caplog):
    io = _make_joint_io(_FakeJointNode(endpoint=_FakeEndpointNode([0.0, 0.0, 0.0])))

    assert io.joint_endpoint_world_position("missing_joint") is None
    assert caplog.text.count("No HingeJoint/SliderJoint node found") == 1


_IDENTITY_R9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_joint_endpoint_orientation_returns_three_rows_from_the_child_solid(caplog):
    joint = _FakeJointNode(endpoint=_FakeEndpointNode([9.0, 9.0, 9.0], orientation=_IDENTITY_R9))
    io = _make_joint_io(joint)

    rows = io.joint_endpoint_world_orientation("arm_7_joint")

    assert rows == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert joint.position_calls == 0
    assert "getPosition" not in caplog.text


def test_joint_endpoint_orientation_returns_none_and_warns_once_for_missing_endpoint(caplog):
    joint = _FakeJointNode(endpoint=None)
    io = _make_joint_io(joint)

    assert io.joint_endpoint_world_orientation("arm_7_joint") is None
    assert io.joint_endpoint_world_orientation("arm_7_joint") is None
    assert caplog.text.count("has no endPoint Solid") == 1


def test_joint_endpoint_orientation_caches_a_lookup_failure(caplog):
    endpoint = _FakeEndpointNode([0.0, 0.0, 0.0], error=RuntimeError("not Pose-derived"))
    joint = _FakeJointNode(endpoint=endpoint)
    io = _make_joint_io(joint)

    assert io.joint_endpoint_world_orientation("arm_7_joint") is None
    assert io.joint_endpoint_world_orientation("arm_7_joint") is None
    assert endpoint.orientation_calls == 1
    assert caplog.text.count("endPoint orientation lookup failed") == 1


def test_joint_endpoint_orientation_rejects_non_finite_values(caplog):
    bad = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float("nan")]
    joint = _FakeJointNode(endpoint=_FakeEndpointNode([0.0, 0.0, 0.0], orientation=bad))
    io = _make_joint_io(joint)

    assert io.joint_endpoint_world_orientation("arm_7_joint") is None
    assert caplog.text.count("non-finite orientation") == 1


class _StubChain:
    def __init__(self, local_position, rotation=None):
        self.local_position = np.asarray(local_position, dtype=float)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)

    def forward_kinematics(self, _q, full_kinematics=False):
        assert full_kinematics is True
        matrices = [np.eye(4) for _ in range(3)]
        matrices[1][:3, :3] = self.rotation
        matrices[1][:3, 3] = self.local_position
        return matrices


class _StubKinematics:
    idx = {"arm_7_joint": 1}

    def __init__(self, local_position, rotation=None):
        self.chain = _StubChain(local_position, rotation)

    def forward(self, _q):
        return np.eye(4)


def _diagnostics_backend(fk_local, reset_endpoint, current_endpoint):
    backend = _make_backend()
    backend._reset_base_position = np.asarray([0.1, 0.2, 0.3])
    backend._reset_base_yaw = 0.0
    backend.kinematics = cast(Kinematics, _StubKinematics(fk_local))
    backend.io.joint_endpoint_world_position = lambda _name: (
        None if current_endpoint is None else list(current_endpoint)
    )
    fk_world = np.asarray([0.1, 0.2, 0.3]) + np.asarray(fk_local, dtype=float)
    backend._arm7_endpoint_offset_world = (
        None if reset_endpoint is None else np.asarray(reset_endpoint) - fk_world
    )
    backend._arm7_reset_link_rotation = np.eye(3)
    backend._last_joint_pos = backend._joint_positions()
    return backend


def test_fk_ground_truth_diagnostics_report_zero_error_for_a_stable_offset():
    # Base is [0.1, 0.2, 0.3], so FK world = base + local = [0.6, 0.0, 1.2].
    endpoint = [0.6, 0.0, 1.2]
    backend = _diagnostics_backend([0.5, -0.2, 0.9], endpoint, endpoint)

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_world_xyz_m"] == pytest.approx([0.6, 0.0, 1.2])
    assert diagnostics["arm_7_joint_ground_truth_world_xyz_m"] == pytest.approx(endpoint)
    assert diagnostics["arm_7_joint_ground_truth_source"] == "endPointSolid"
    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is False
    assert diagnostics["arm_7_joint_endpoint_fk_validation_available"] is True


def test_fk_ground_truth_diagnostics_flag_a_five_centimetre_offset_change():
    backend = _diagnostics_backend([0.5, -0.2, 0.9], [0.6, 0.0, 1.2], [0.65, 0.0, 1.2])

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(0.05)
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is True
    assert diagnostics["arm_7_joint_endpoint_offset_change_world_xyz_m"] == pytest.approx(
        [0.05, 0.0, 0.0]
    )


def test_fk_ground_truth_diagnostics_mark_unavailable_points_as_none():
    backend = _diagnostics_backend([0.5, -0.2, 0.9], [0.6, 0.0, 1.2], None)

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_ground_truth_world_xyz_m"] is None
    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] is None
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is None
    assert diagnostics["arm_7_joint_endpoint_fk_validation_available"] is False


def test_fk_ground_truth_ignores_a_constant_endpoint_offset():
    # The endPoint Solid sits 6.7 cm beyond FK(arm_7); a constant offset must not be an error.
    fk_local = [0.5, -0.2, 0.9]
    fk_world = np.asarray([0.1, 0.2, 0.3]) + np.asarray(fk_local)
    endpoint = fk_world + np.asarray([0.0, 0.0, 0.067725])
    backend = _diagnostics_backend(fk_local, endpoint, endpoint)

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["arm_7_joint_endpoint_offset_world_xyz_m"] == pytest.approx(
        [0.0, 0.0, 0.067725]
    )


def test_fk_ground_truth_rotates_the_reset_offset_with_the_wrist():
    # The tool offset is constant in the arm_7 link frame, so a wrist rotation must not look like
    # an FK error once the reset-time offset is rotated into the current link frame.
    fk_local = np.asarray([0.5, -0.2, 0.9])
    fk_world = np.asarray([0.1, 0.2, 0.3]) + fk_local
    local_offset = np.asarray([0.0, 0.0, 0.067725])
    reset_rotation = np.eye(3)
    backend = _make_backend()
    backend._reset_base_position = np.asarray([0.1, 0.2, 0.3])
    backend._reset_base_yaw = 0.0
    kinematics = _StubKinematics(fk_local, reset_rotation)
    backend.kinematics = cast(Kinematics, kinematics)
    backend._arm7_endpoint_offset_world = reset_rotation @ local_offset
    backend._arm7_reset_link_rotation = reset_rotation
    backend._last_joint_pos = backend._joint_positions()

    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    kinematics.chain.rotation = rotation
    endpoint = fk_world + rotation @ local_offset
    backend.io.joint_endpoint_world_position = lambda _name: endpoint.tolist()

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is False


def test_fk_ground_truth_flags_an_offset_that_does_not_rotate_with_the_wrist():
    fk_local = np.asarray([0.5, -0.2, 0.9])
    fk_world = np.asarray([0.1, 0.2, 0.3]) + fk_local
    local_offset = np.asarray([0.0, 0.0, 0.067725])
    backend = _make_backend()
    backend._reset_base_position = np.asarray([0.1, 0.2, 0.3])
    backend._reset_base_yaw = 0.0
    kinematics = _StubKinematics(fk_local, np.eye(3))
    backend.kinematics = cast(Kinematics, kinematics)
    backend._arm7_endpoint_offset_world = local_offset
    backend._arm7_reset_link_rotation = np.eye(3)
    backend._last_joint_pos = backend._joint_positions()

    kinematics.chain.rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    endpoint = fk_world + local_offset  # offset did not rotate with the link
    backend.io.joint_endpoint_world_position = lambda _name: endpoint.tolist()

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(
        np.linalg.norm(local_offset - kinematics.chain.rotation @ local_offset)
    )
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is True


def test_reset_fingerprint_is_identical_for_two_same_seed_resets():
    backend = _make_backend()
    backend.io.reset_fingerprint = lambda: {
        "motors": {
            name: {
                "target_position_rad": sensor.getValue(),
                "velocity_rad_s": 0.0,
                "control_mode": "position",
            }
            for name, sensor in backend.io.sensors.items()
        },
        "joint_sensors_rad": {
            name: sensor.getValue() for name, sensor in backend.io.sensors.items()
        },
        "base": {"world_xyz_m": [0.1, 0.2, 0.3], "world_orientation_matrix": [1.0] * 9},
        "can": {"world_xyz_m": [1.0, 2.0, 3.0], "world_orientation_matrix": [1.0] * 9},
    }

    first = backend.reset(ScenarioSpec(), seed=0)
    first_fingerprint = first.robot.diagnostics["reset_fingerprint"]
    second = backend.reset(ScenarioSpec(), seed=0)
    second_fingerprint = second.robot.diagnostics["reset_fingerprint"]

    assert first_fingerprint == second_fingerprint
    assert first_fingerprint["torso_sensor_after_rad"] == pytest.approx(0.35)
    assert first_fingerprint["torso_commanded_rad"] == pytest.approx(0.35)
    assert first.robot.diagnostics["reset_torso_after_rad"] == pytest.approx(0.35)


class _FakeWheelMotor:
    def __init__(self):
        self.position = None
        self.velocity = None

    def setPosition(self, position):
        self.position = position

    def setVelocity(self, velocity):
        self.velocity = velocity


def test_hold_base_none_leaves_the_wheel_proto_hold_untouched(monkeypatch):
    import vlarobot.backends.webots.robot_io as robot_io

    monkeypatch.setattr(robot_io, "HOLD_BASE", "none")
    io = object.__new__(RobotIO)
    motor = _FakeWheelMotor()
    io.wheels = {"wheel_left_joint": (motor, None), "wheel_right_joint": (motor, None)}

    io.apply_hold_base()

    assert motor.position is None
    assert motor.velocity is None


def test_hold_base_wheels_zero_velocity_switches_to_velocity_control(monkeypatch):
    import math

    import vlarobot.backends.webots.robot_io as robot_io

    monkeypatch.setattr(robot_io, "HOLD_BASE", "wheels_zero_velocity")
    io = object.__new__(RobotIO)
    left, right = _FakeWheelMotor(), _FakeWheelMotor()
    io.wheels = {"wheel_left_joint": (left, None), "wheel_right_joint": (right, None)}

    io.apply_hold_base()

    for motor in (left, right):
        assert motor.position is not None
        assert math.isinf(motor.position)
        assert motor.velocity == 0.0


def _attach_settle_fixtures(backend, velocity):
    """Attach the pieces ``_settle_post_reset`` needs that the plain fakes lack.

    ``velocity`` is the 6-vector ``me.getVelocity()`` returns every step; a constant
    value reproduces the MEASURED live behaviour of a base that keeps a residual
    angular jitter without ever decaying (ADR-0022 evidence).
    """
    backend._physics_step_count = 0

    class _FakeVisualization:
        def step(self):
            return backend.io.robot.step(backend.io.ts)

    backend.visualization = _FakeVisualization()
    backend.io.me.getVelocity = lambda: list(velocity)
    return backend


def test_post_reset_settle_persistent_jitter_is_recorded_not_fatal(monkeypatch, caplog):
    """MEASURED live failure mode: angular jitter ~1.6e-4 rad/s > 1e-4 threshold.

    Before ADR-0022 this raised RuntimeError and aborted the reset (error reply ->
    client raise -> runner kill). It must now be recorded with ``fatal: False`` and
    no exception. Thresholds are unchanged (1e-4 rad/s, 20 consecutive steps).
    """
    import vlarobot.backends.webots.backend as backend_module

    monkeypatch.setattr(backend_module, "RESET_POST_SETTLE_ENABLED", True)
    monkeypatch.setattr(backend_module, "RESET_POST_SETTLE_MAX_STEPS", 5)
    jitter = [0.0, 0.0, 0.0, 0.0, 0.0, 1.615e-4]
    backend = _attach_settle_fixtures(_make_backend(), jitter)

    info = backend._settle_post_reset()

    assert info["reason"] == "max_steps"
    assert info["steps_used"] == 5
    assert info["consecutive_stable_steps"] == 0
    assert info["required_consecutive_steps"] == 20
    assert info["fatal"] is False
    assert info["final"]["angular_speed_rad_s"] == pytest.approx(1.615e-4)
    assert info["final"]["linear_speed_m_s"] == pytest.approx(0.0)
    assert any("not settled (recorded, non-fatal)" in message for message in caplog.messages)


def test_post_reset_settle_below_thresholds_settles_normally(monkeypatch):
    import vlarobot.backends.webots.backend as backend_module

    monkeypatch.setattr(backend_module, "RESET_POST_SETTLE_ENABLED", True)
    monkeypatch.setattr(backend_module, "RESET_POST_SETTLE_MAX_STEPS", 500)
    calm = [0.0, 0.0, 0.0, 0.0, 0.0, 9.38e-5]  # MEASURED live ep00 (settled) angular speed
    backend = _attach_settle_fixtures(_make_backend(), calm)

    info = backend._settle_post_reset()

    assert info["reason"] == "settled"
    assert info["steps_used"] == 20
    assert info["consecutive_stable_steps"] == 20
    assert info["fatal"] is False
    assert info["final"]["angular_speed_rad_s"] == pytest.approx(9.38e-5)


def test_reset_with_settle_gate_survives_persistent_jitter(monkeypatch):
    """End to end: reset must return an observation and record the non-fatal outcome."""
    import vlarobot.backends.webots.backend as backend_module

    monkeypatch.setattr(backend_module, "RESET_POST_SETTLE_ENABLED", True)
    monkeypatch.setattr(backend_module, "RESET_POST_SETTLE_MAX_STEPS", 3)
    jitter = [0.0, 0.0, 0.0, 0.0, 0.0, 1.615e-4]
    backend = _attach_settle_fixtures(_make_backend(), jitter)
    backend.io.reset_fingerprint = lambda: {"kinematic_model": "fake"}

    observation = backend.reset(ScenarioSpec(), seed=0)

    fingerprint = observation.robot.diagnostics["reset_fingerprint"]
    settle = fingerprint["post_reset_settle"]
    assert settle["reason"] == "max_steps"
    assert settle["steps_used"] == 3
    assert settle["fatal"] is False
    assert fingerprint["physics_step_count_at_reset"] == 3


# ---------------------------------------------------------------------------
# Scenario objects at reset (ADR-0029 week 1: obstacle positions).
# ---------------------------------------------------------------------------

HAM_HOME = [-0.544929, -4.5081, 0.754298]
APP_HOME = [-0.41, -4.44, 0.83]


def _obstacle_world() -> dict[str, FakeCan]:
    return {
        "HAM": FakeCan(list(HAM_HOME), [0.0, 0.0, 1.0, 0.0]),
        "APP": FakeCan(list(APP_HOME), [0.0, 0.0, 1.0, 0.0]),
    }


def _placed_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="cluttered-test",
        objects=(
            {"def": "HAM", "xyz": [-0.70, -4.50, 0.754298], "yaw_offset": math.pi / 2},
            {"def": "APP", "xyz": [-0.65, -4.40, 0.83], "yaw_offset": 0.0},
        ),
    )


def test_reset_places_scenario_objects_and_restores_them_on_default_reset():
    backend = _make_backend(obstacles=_obstacle_world())
    placed = _placed_scenario()

    backend.reset(placed, seed=0)

    ham = backend.io.robot.nodes["HAM"]
    app = backend.io.robot.nodes["APP"]
    assert ham.getField("translation").getSFVec3f() == [-0.70, -4.50, 0.754298]
    np.testing.assert_allclose(
        ham.getField("rotation").getSFRotation(), [0.0, 0.0, 1.0, math.pi / 2], atol=1e-9
    )
    assert app.getField("translation").getSFVec3f() == [-0.65, -4.40, 0.83]
    assert app.getField("rotation").getSFRotation() == [0.0, 0.0, 1.0, 0.0]

    ground_truth = backend.get_ground_truth()
    assert set(ground_truth.object_poses) == {CAN_DEF, "HAM", "APP"}
    assert ground_truth.object_poses["HAM"].position.x == -0.70

    # A default reset restores every placed node to its cached world-file pose.
    backend.reset(ScenarioSpec(), seed=1)
    assert ham.getField("translation").getSFVec3f() == HAM_HOME
    assert app.getField("translation").getSFVec3f() == APP_HOME
    assert ham.getField("rotation").getSFRotation() == [0.0, 0.0, 1.0, 0.0]

    # Re-placing after a restore starts from the cached pose: no cumulative drift.
    backend.reset(placed, seed=2)
    assert ham.getField("translation").getSFVec3f() == [-0.70, -4.50, 0.754298]
    np.testing.assert_allclose(
        ham.getField("rotation").getSFRotation(), [0.0, 0.0, 1.0, math.pi / 2], atol=1e-9
    )
    assert app.getField("translation").getSFVec3f() == [-0.65, -4.40, 0.83]


def test_reset_reports_unknown_object_def_and_malformed_entries():
    # Unknown DEF: no node to move (the world cannot spawn it).
    backend = _make_backend()
    with pytest.raises(RuntimeError, match="No node with DEF HAM"):
        backend.reset(_placed_scenario(), seed=0)

    backend = _make_backend(obstacles=_obstacle_world())
    with pytest.raises(NotImplementedError, match="spawning new nodes"):
        backend.reset(ScenarioSpec(objects=({"proto": "Hammer"},)), seed=0)
    with pytest.raises(ValueError, match="xyz"):
        backend.reset(ScenarioSpec(objects=({"def": "HAM", "xyz": [1.0, 2.0]},)), seed=0)
    with pytest.raises(ValueError, match="can_xy_yaw_offset"):
        backend.reset(ScenarioSpec(objects=({"def": CAN_DEF, "xyz": [0.0, 0.0, 0.0]},)), seed=0)
