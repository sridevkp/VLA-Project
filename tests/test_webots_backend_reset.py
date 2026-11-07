from typing import Any, cast

import numpy as np

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
        motor.position_field.value == position
        for name, position in backend._initial_joint_pos.items()
        for motor in [backend.io.joint_positions[name]]
    )
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
