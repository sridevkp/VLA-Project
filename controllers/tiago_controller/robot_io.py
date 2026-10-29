"""Webots device discovery and low-level simulation I/O."""
from controller import Supervisor
from config import TIME_STEP, ARM_JOINTS, FINGER_JOINTS, TORSO_JOINT, FINGER_SPEED, FINGERS_INVERTED

class RobotIO:
    def __init__(self):
        self.robot = Supervisor()
        self.ts = int(self.robot.getBasicTimeStep()) or TIME_STEP
        self.motors, self.sensors, self.fingers = {}, {}, []
        for name in ARM_JOINTS + [TORSO_JOINT]:
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
        print("Gripper limits:", [(m.getMinPosition(), m.getMaxPosition()) for m, _ in self.fingers])
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

    def current_finger_values(self):
        return [round(s.getValue(), 4) for _, s in self.fingers]
