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
    robot_node.getContactPoints = lambda **_: [FakeContact()]

    assert io.contact_nodes() == [
        {
            "node_id": 22,
            "node_name": "TABLE",
            "node_type": "Solid",
            "contact_points_world_xyz_m": [[-1.2, -4.1, 0.73]],
        }
    ]


class FakeRobot:
    def __init__(self, can_node, me):
        self.can_node = can_node
        self.me = me
        self.time = 10.0
        self.lookup_count = 0

    def getFromDef(self, name):
        assert name == CAN_DEF
        self.lookup_count += 1
        return self.can_node

    def simulationResetPhysics(self):
        self.can_node = FakeCan(
            self.can_node.getField("translation").getSFVec3f(),
            self.can_node.getField("rotation").getSFRotation(),
        )

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


def _make_backend():
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
    robot = FakeRobot(can_node, me)

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
    def __init__(self, position, endpoint=None):
        self.position = position
        self.endpoint = endpoint

    def getTypeName(self):
        return "HingeJoint"

    def getPosition(self):
        return None if self.position is None else list(self.position)

    def getField(self, name):
        return _FakeEndpointField(self.endpoint) if name == "endPoint" else None


class _FakeEndpointNode:
    def __init__(self, position):
        self.position = position

    def getPosition(self):
        return list(self.position)


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
    return io


def test_joint_anchor_returns_the_supervisor_joint_position():
    joint = _FakeJointNode([0.5, -0.2, 0.9], endpoint=_FakeEndpointNode([9.0, 9.0, 9.0]))
    io = _make_joint_io(joint)

    assert io.joint_anchor_world_position("arm_7_joint") == [0.5, -0.2, 0.9]


def test_joint_anchor_does_not_substitute_the_endpoint_when_position_is_unavailable():
    joint = _FakeJointNode(None, endpoint=_FakeEndpointNode([9.0, 9.0, 9.0]))
    io = _make_joint_io(joint)

    assert io.joint_anchor_world_position("arm_7_joint") is None
    assert io.joint_endpoint_world_position("arm_7_joint") == [9.0, 9.0, 9.0]


def test_joint_anchor_returns_none_for_unknown_joint():
    io = _make_joint_io(_FakeJointNode([0.0, 0.0, 0.0]))

    assert io.joint_anchor_world_position("missing_joint") is None


class _StubChain:
    def __init__(self, local_position):
        self.local_position = np.asarray(local_position, dtype=float)

    def forward_kinematics(self, _q, full_kinematics=False):
        assert full_kinematics is True
        matrices = [np.eye(4) for _ in range(3)]
        matrices[1][:3, 3] = self.local_position
        return matrices


class _StubKinematics:
    idx = {"arm_7_joint": 1}

    def __init__(self, local_position):
        self.chain = _StubChain(local_position)

    def forward(self, _q):
        return np.eye(4)


def _diagnostics_backend(fk_local, ground_truth_world):
    backend = _make_backend()
    backend._reset_base_position = np.asarray([0.1, 0.2, 0.3])
    backend._reset_base_yaw = 0.0
    backend.kinematics = cast(Kinematics, _StubKinematics(fk_local))
    backend.io.joint_anchor_world_position = lambda _name: list(ground_truth_world)
    backend.io.joint_endpoint_world_position = lambda _name: list(ground_truth_world)
    backend._last_joint_pos = backend._joint_positions()
    return backend


def test_fk_ground_truth_diagnostics_report_zero_error_for_identical_points():
    # Base is [0.1, 0.2, 0.3], so FK world = base + local = [0.6, 0.0, 1.2].
    backend = _diagnostics_backend([0.5, -0.2, 0.9], [0.6, 0.0, 1.2])

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_world_xyz_m"] == pytest.approx([0.6, 0.0, 1.2])
    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is False


def test_fk_ground_truth_diagnostics_flag_a_five_centimetre_offset():
    backend = _diagnostics_backend([0.5, -0.2, 0.9], [0.65, 0.0, 1.2])

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] == pytest.approx(0.05)
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is True


def test_fk_ground_truth_diagnostics_mark_unavailable_points_as_none():
    backend = _diagnostics_backend([0.5, -0.2, 0.9], [0.6, 0.0, 1.2])
    backend.io.joint_anchor_world_position = lambda _name: None

    diagnostics = backend._robot_state().diagnostics

    assert diagnostics["arm_7_joint_ground_truth_world_xyz_m"] is None
    assert diagnostics["arm_7_joint_fk_ground_truth_error_m"] is None
    assert diagnostics["arm_7_joint_fk_ground_truth_mismatch"] is None
