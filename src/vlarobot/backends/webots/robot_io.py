"""Webots device discovery and low-level simulation I/O."""

import importlib
import logging
import math
from typing import Any

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    FINGER_JOINTS,
    FINGER_SPEED,
    FINGERS_INVERTED,
    HEAD_JOINTS,
    TIME_STEP,
    TORSO_JOINT,
)

LOGGER = logging.getLogger(__name__)


class RobotIO:
    def __init__(self):
        self.robot = importlib.import_module("controller").Supervisor()
        self.ts = int(self.robot.getBasicTimeStep()) or TIME_STEP
        self.motors, self.sensors, self.fingers = {}, {}, []
        for name in ARM_JOINTS + [TORSO_JOINT] + HEAD_JOINTS:
            motor = self.robot.getDevice(name)
            if motor is None:
                raise RuntimeError(f"Required Webots motor not found: {name}")
            sensor = motor.getPositionSensor()
            if sensor is None:
                raise RuntimeError(f"Position sensor not found for motor: {name}")
            sensor.enable(self.ts)
            self.motors[name], self.sensors[name] = motor, sensor
        for name in ARM_JOINTS:
            self.motors[name].setVelocity(0.5)
        for name in FINGER_JOINTS:
            motor = self.robot.getDevice(name)
            if motor is None:
                print(f"[warn] finger motor {name!r} not found")
                continue
            sensor = motor.getPositionSensor()
            sensor.enable(self.ts)
            motor.setVelocity(FINGER_SPEED)
            self.fingers.append((motor, sensor))
        maxes = [m.getMaxPosition() for m, _ in self.fingers]
        self.full_stroke = min([x for x in maxes if x > 0] + [0.045])
        if FINGERS_INVERTED:
            self.open_pos, self.closed_pos = 0.0, self.full_stroke
        else:
            self.closed_pos, self.open_pos = 0.0, self.full_stroke
        print("Fingers found:", len(self.fingers))
        print(
            "Gripper limits:",
            [(m.getMinPosition(), m.getMaxPosition()) for m, _ in self.fingers],
        )
        self.me = self.robot.getSelf()
        self.can_node = None
        self.step_raw(2)  # allow position sensors to initialize

    def step_raw(self, count=1):
        for _ in range(count):
            if self.robot.step(self.ts) == -1:
                return False
        return True

    def initialize_can(self, def_name):
        self.can_node = self.robot.getFromDef(def_name)
        if self.can_node is None:
            raise RuntimeError(f"No node with DEF {def_name} in the world.")
        self.step_raw(1)
        return self.can_node

    def enable_can_contact_tracking(self):
        if self.me is None or self.can_node is None:
            raise RuntimeError("Robot and CAN nodes must be initialized before contact tracking.")
        self.me.enableContactPointsTracking(self.ts, includeDescendants=True)

    def can_contact_detected(self):
        if self.me is None or self.can_node is None:
            raise RuntimeError("Robot and CAN nodes must be initialized before contact queries.")
        can_id = self.can_node.getId()
        return any(
            contact.getNodeId() == can_id
            for contact in self.me.getContactPoints(includeDescendants=True)
        )

    def current_finger_values(self):
        return [round(s.getValue(), 4) for _, s in self.fingers]

    def camera_noise_value(self, camera: Any) -> float | None:
        node = self.robot.getFromDevice(camera._tag)
        if node is None:
            raise RuntimeError("No Webots node found for the RGB camera.")
        noise_field = node.getField("noise")
        return None if noise_field is None else float(noise_field.getSFFloat())

    def joint_positions_from_fields(self) -> dict[str, float]:
        motors = dict(self.motors)
        motors.update({motor.getName(): motor for motor, _ in self.fingers})
        positions = {}
        for name, motor in motors.items():
            position = self._joint_position_field(motor, name).getSFFloat()
            if not math.isfinite(position):
                raise RuntimeError(f"Non-finite reset baseline for joint {name!r}: {position}.")
            minimum = motor.getMinPosition()
            maximum = motor.getMaxPosition()
            valid_position = min(max(position, minimum), maximum)
            if valid_position != position:
                LOGGER.warning(
                    "Reset baseline for joint %s (%s) is outside [%s, %s]; "
                    "using nearest legal limit %s.",
                    name,
                    position,
                    minimum,
                    maximum,
                    valid_position,
                )
            positions[name] = valid_position
        return positions

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        motors = dict(self.motors)
        motors.update({motor.getName(): motor for motor, _ in self.fingers})
        for name, position in positions.items():
            motor = motors.get(name)
            if motor is None:
                raise ValueError(f"Unknown joint motor {name!r}.")
            if (
                not math.isfinite(position)
                or not motor.getMinPosition() <= position <= motor.getMaxPosition()
            ):
                raise ValueError(f"Invalid reset position for joint {name!r}: {position}.")
            position_field = self._joint_position_field(motor, name)
            position_field.setSFFloat(position)
            motor.setPosition(position)

    def _joint_position_field(self, motor: Any, name: str) -> Any:
        node = self.robot.getFromDevice(motor._tag)
        if node is None:
            raise RuntimeError(f"No Webots node for joint motor {name!r}.")
        node = node.getParentNode()
        while node is not None:
            if node.getTypeName() in {"HingeJoint", "SliderJoint"}:
                parameters = node.getField("jointParameters")
                parameter_node = parameters.getSFNode() if parameters is not None else None
                position_field = (
                    parameter_node.getField("position")
                    if parameter_node is not None
                    else node.getField("position")
                )
                if position_field is None:
                    break
                return position_field
            node = node.getParentNode()
        raise RuntimeError(f"No writable Webots joint position field for {name!r}.")
