"""Webots device discovery and low-level simulation I/O."""

import importlib
import logging
import math
from typing import Any, cast

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    FINGER_JOINTS,
    FINGER_SPEED,
    FINGERS_INVERTED,
    HEAD_JOINTS,
    HOLD_BASE,
    TIME_STEP,
    TORSO_JOINT,
)

LOGGER = logging.getLogger(__name__)


def _finite_position(position: Any) -> list[float] | None:
    """Return a finite world position as floats, or ``None`` when unavailable."""
    if position is None:
        return None
    try:
        values = [float(value) for value in position]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float or ``None`` for infinities, NaNs, and bad values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flatten_matrix(values: Any) -> list[float] | None:
    """Return a finite row-major 3x3 matrix from a Webots orientation field."""
    finite = _finite_position(values)
    if finite is None or len(finite) != 9:
        return None
    return finite


def offset_point_world(
    position: Any, orientation: Any, local_offset: tuple[float, float, float]
) -> list[float] | None:
    """Apply a local offset, rotated by a row-major 3x3 orientation, to a world point."""
    point = _finite_position(position)
    matrix = _flatten_matrix(orientation)
    if point is None or matrix is None:
        return None
    x, y, z = local_offset
    return [
        point[0] + matrix[0] * x + matrix[1] * y + matrix[2] * z,
        point[1] + matrix[3] * x + matrix[4] * y + matrix[5] * z,
        point[2] + matrix[6] * x + matrix[7] * y + matrix[8] * z,
    ]


class RobotIO:
    def __init__(self):
        self.robot = importlib.import_module("controller").Supervisor()
        self.ts = int(self.robot.getBasicTimeStep()) or TIME_STEP
        self.motors, self.sensors, self.fingers = {}, {}, []
        self.wheels = {}
        self._lookup_failures: dict[str, str] = {}
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
        for name in ("wheel_left_joint", "wheel_right_joint"):
            motor = self.robot.getDevice(name)
            if motor is not None:
                sensor = motor.getPositionSensor()
                if sensor is not None:
                    sensor.enable(self.ts)
                self.wheels[name] = (motor, sensor)
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

    def enable_contact_tracking(self):
        if self.me is None:
            raise RuntimeError("Robot node must be initialized before contact tracking.")
        self.me.enableContactPointsTracking(self.ts, includeDescendants=True)

    def contact_nodes(self) -> list[dict[str, object]]:
        """Return deduplicated non-robot contact nodes with stable identifying labels."""
        contacts = self.me.getContactPoints(includeDescendants=True)
        result: dict[int, dict[str, object]] = {}
        robot_id = self.me.getId()
        for contact in contacts:
            node_id = int(contact.getNodeId())
            node = self.robot.getFromId(node_id)
            if node is None or node_id == robot_id:
                continue
            parent = node
            belongs_to_robot = False
            while parent is not None:
                if int(parent.getId()) == robot_id:
                    belongs_to_robot = True
                    break
                parent = parent.getParentNode()
            if belongs_to_robot:
                continue
            def_name = node.getDef()
            type_name = node.getTypeName()
            name_field = node.getField("name")
            name = name_field.getSFString() if name_field is not None else ""
            label = def_name or name or f"{type_name}#{node_id}"
            point_getter = getattr(contact, "getPoint", None)
            point = point_getter() if point_getter is not None else getattr(contact, "point", None)
            item = result.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "node_name": label,
                    "node_type": type_name,
                    "contact_points_world_xyz_m": [],
                },
            )
            if point is not None:
                points = cast(list[list[float]], item["contact_points_world_xyz_m"])
                points.append([float(value) for value in point])
        return list(result.values())

    def wheel_status(self) -> dict[str, dict[str, float | str | None]]:
        status = {}
        for name, (motor, sensor) in self.wheels.items():
            target = float(motor.getTargetPosition())
            status[name] = {
                "control_mode": "velocity" if math.isinf(target) else "position",
                "target_position_rad": target if math.isfinite(target) else None,
                "target_velocity_rad_s": float(motor.getVelocity()),
                "sensor_position_rad": float(sensor.getValue()) if sensor is not None else None,
            }
        return status

    def apply_hold_base(self) -> None:
        """Optionally brake the base wheels; ``none`` leaves the PROTO position hold alone."""
        if HOLD_BASE != "wheels_zero_velocity":
            return
        for motor, _ in self.wheels.values():
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

    def _joint_node(self, name: str) -> Any | None:
        motor = self.motors.get(name)
        if motor is None:
            return None
        return self._device_joint_node(motor)

    def _device_joint_node(self, motor: Any) -> Any | None:
        node = self.robot.getFromDevice(motor._tag)
        while node is not None and node.getTypeName() not in {"HingeJoint", "SliderJoint"}:
            node = node.getParentNode()
        return node

    def _warn_lookup_once(self, key: str, message: str, *args: object) -> None:
        """Log a Supervisor lookup failure once and remember it for later calls."""
        failures = self.__dict__.setdefault("_lookup_failures", {})
        if key in failures:
            return
        failures[key] = message
        LOGGER.warning(message, *args)

    def _lookup_failed(self, key: str) -> bool:
        return key in self.__dict__.setdefault("_lookup_failures", {})

    def joint_endpoint_world_position(self, name: str) -> list[float] | None:
        """World position of the Pose-derived node in a joint's ``endPoint`` field.

        Webots only supports ``getPosition()`` on Pose-derived nodes (``Solid``, ``Robot``,
        ``Pose``, ...). A ``HingeJoint`` or ``SliderJoint`` node is not Pose-derived and calling
        ``getPosition()`` on it emits
        ``wb_supervisor_node_get_position() can exclusively be used with Pose`` for every call.
        The child Solid attached through ``endPoint`` is Pose-derived, so it is the only valid
        Supervisor reference for arm-link telemetry. Failures are cached and warned once per
        distinct lookup.
        """
        key = f"joint_endpoint:{name}"
        if self._lookup_failed(key):
            return None
        node = self._joint_node(name)
        if node is None:
            self._warn_lookup_once(
                key,
                "No HingeJoint/SliderJoint node found for joint %s; link telemetry is None.",
                name,
            )
            return None
        endpoint_field = node.getField("endPoint")
        endpoint = endpoint_field.getSFNode() if endpoint_field is not None else None
        if endpoint is None:
            self._warn_lookup_once(
                key, "Joint %s has no endPoint Solid; link telemetry is None.", name
            )
            return None
        try:
            position = endpoint.getPosition()
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._warn_lookup_once(
                key,
                "Joint %s endPoint position lookup failed (%s); link telemetry is None.",
                name,
                error,
            )
            return None
        finite = _finite_position(position)
        if finite is None:
            self._warn_lookup_once(
                key,
                "Joint %s endPoint returned a non-finite position; link telemetry is None.",
                name,
            )
        return finite

    def joint_endpoint_world_orientation(self, name: str) -> list[list[float]] | None:
        """World rotation (3x3 rows, row-major) of a joint's ``endPoint`` Solid.

        Same Pose-derived restriction as :meth:`joint_endpoint_world_position`: the joint node
        itself is not Pose-derived, but the ``endPoint`` child Solid is, and ``getOrientation()``
        is valid on it. ``world = R @ local`` with R expressed in the world frame. Failures are
        cached and warned once per distinct lookup, mirroring the position lookup.
        """
        key = f"joint_endpoint_orientation:{name}"
        if self._lookup_failed(key):
            return None
        node = self._joint_node(name)
        if node is None:
            self._warn_lookup_once(
                key,
                "No HingeJoint/SliderJoint node found for joint %s; link orientation is None.",
                name,
            )
            return None
        endpoint_field = node.getField("endPoint")
        endpoint = endpoint_field.getSFNode() if endpoint_field is not None else None
        if endpoint is None:
            self._warn_lookup_once(
                key, "Joint %s has no endPoint Solid; link orientation is None.", name
            )
            return None
        try:
            orientation = endpoint.getOrientation()
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._warn_lookup_once(
                key,
                "Joint %s endPoint orientation lookup failed (%s); link orientation is None.",
                name,
                error,
            )
            return None
        if (
            orientation is None
            or len(orientation) != 9
            or not all(math.isfinite(float(value)) for value in orientation)
        ):
            self._warn_lookup_once(
                key,
                "Joint %s endPoint returned a non-finite orientation; link orientation is None.",
                name,
            )
            return None
        return [
            [float(orientation[0]), float(orientation[1]), float(orientation[2])],
            [float(orientation[3]), float(orientation[4]), float(orientation[5])],
            [float(orientation[6]), float(orientation[7]), float(orientation[8])],
        ]

    def finger_tip_world_positions(self, offset_m: float) -> dict[str, list[float]] | None:
        """World position of each finger pad centre, from the Pose-derived endPoint Solid.

        The finger ``SliderJoint`` node is not Pose-derived, but its ``endPoint`` child Solid
        (``gripper_left_finger_link`` / ``gripper_right_finger_link``) is. Its origin sits at the
        slider joint; the pad centre is ``offset_m`` along the Solid's local -z axis. Returns
        ``None`` if either finger is unavailable so a midpoint is never built from one side.
        """
        if not self.fingers:
            return None
        positions: dict[str, list[float]] = {}
        for motor, _ in self.fingers:
            name = motor.getName()
            key = f"finger_tip:{name}"
            if self._lookup_failed(key):
                return None
            node = self._device_joint_node(motor)
            if node is None:
                self._warn_lookup_once(
                    key, "No SliderJoint node found for finger %s; finger telemetry is None.", name
                )
                return None
            endpoint_field = node.getField("endPoint")
            endpoint = endpoint_field.getSFNode() if endpoint_field is not None else None
            if endpoint is None:
                self._warn_lookup_once(
                    key, "Finger %s has no endPoint Solid; finger telemetry is None.", name
                )
                return None
            try:
                position = endpoint.getPosition()
                orientation = endpoint.getOrientation()
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self._warn_lookup_once(
                    key,
                    "Finger %s endPoint lookup failed (%s); finger telemetry is None.",
                    name,
                    error,
                )
                return None
            tip = offset_point_world(position, orientation, (0.0, 0.0, -abs(offset_m)))
            if tip is None:
                self._warn_lookup_once(
                    key, "Finger %s returned a non-finite tip; finger telemetry is None.", name
                )
                return None
            positions[name] = tip
        return positions

    def reset_fingerprint(self) -> dict[str, object]:
        """Capture every Webots state value the soft reset is expected to restore.

        The fingerprint is emitted only on the reset observation. It records motor targets and
        velocities (position and velocity control modes), all joint sensors, base pose and
        Supervisor velocity, and the CAN pose. Non-finite values are serialized as ``None`` so the
        JSONL trace remains valid.
        """
        motors = dict(self.motors)
        motors.update({motor.getName(): motor for motor, _ in self.fingers})
        motors.update({name: motor for name, (motor, _) in self.wheels.items()})
        motor_state: dict[str, object] = {}
        for name, motor in sorted(motors.items()):
            target_raw = self._safe_motor_call(motor, "getTargetPosition", name)
            target = _finite_or_none(target_raw)
            velocity = _finite_or_none(self._safe_motor_call(motor, "getVelocity", name))
            control_mode = (
                "velocity"
                if target_raw is not None and math.isinf(float(target_raw))
                else "position"
            )
            motor_state[name] = {
                "target_position_rad": target,
                "velocity_rad_s": velocity,
                "control_mode": control_mode,
            }

        sensors = dict(self.sensors)
        sensors.update({motor.getName(): sensor for motor, sensor in self.fingers})
        sensors.update(
            {name: sensor for name, (_, sensor) in self.wheels.items() if sensor is not None}
        )
        sensor_state = {
            name: _finite_or_none(sensor.getValue()) for name, sensor in sorted(sensors.items())
        }

        base_pose: dict[str, object] = {
            "world_xyz_m": None,
            "world_orientation_matrix": None,
            "velocity_world_xyz_m_s": None,
            "angular_velocity_world_xyz_rad_s": None,
        }
        if self.me is not None:
            base_pose["world_xyz_m"] = _finite_position(self.me.getPosition())
            base_pose["world_orientation_matrix"] = _flatten_matrix(self.me.getOrientation())
            velocity = self._safe_supervisor_call(self.me, "getVelocity", "base")
            if velocity is not None:
                values = _finite_position(velocity)
                if values is not None and len(values) >= 6:
                    base_pose["velocity_world_xyz_m_s"] = values[:3]
                    base_pose["angular_velocity_world_xyz_rad_s"] = values[3:6]

        can_pose: dict[str, object] = {"world_xyz_m": None, "world_orientation_matrix": None}
        if self.can_node is not None:
            can_pose["world_xyz_m"] = _finite_position(self.can_node.getPosition())
            can_pose["world_orientation_matrix"] = _flatten_matrix(self.can_node.getOrientation())

        return {
            "motors": motor_state,
            "joint_sensors_rad": sensor_state,
            "base": base_pose,
            "can": can_pose,
        }

    def _safe_motor_call(self, motor: Any, attribute: str, name: str) -> Any:
        key = f"motor:{name}:{attribute}"
        if self._lookup_failed(key):
            return None
        try:
            return getattr(motor, attribute)()
        except (AttributeError, TypeError, ValueError) as error:
            self._warn_lookup_once(
                key,
                "Motor %s %s lookup failed (%s); fingerprint value is None.",
                name,
                attribute,
                error,
            )
            return None

    def _safe_supervisor_call(self, node: Any, attribute: str, name: str) -> Any:
        key = f"supervisor:{name}:{attribute}"
        if self._lookup_failed(key):
            return None
        try:
            return getattr(node, attribute)()
        except (AttributeError, TypeError, ValueError) as error:
            self._warn_lookup_once(
                key,
                "Supervisor %s %s lookup failed (%s); fingerprint value is None.",
                name,
                attribute,
                error,
            )
            return None

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
