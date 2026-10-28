from controller import Robot
from tiago_ik import TIAGoIK
import math


class TIAGoArm:
    ARM_NAMES = [f"arm_{i}_joint" for i in range(1, 8)]
    TORSO_NAME = "torso_lift_joint"

    HOME = {
        "torso_lift_joint": 0.09,
        "arm_1_joint": 0.07,
        "arm_2_joint": 0.26,
        "arm_3_joint": -3.16,
        "arm_4_joint": 1.27,
        "arm_5_joint": 1.32,
        "arm_6_joint": 0.0,
        "arm_7_joint": 1.41,
    }

    def __init__(self, robot, speed_fraction=0.20):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.joint_names = [self.TORSO_NAME] + self.ARM_NAMES

        self.motors = {}
        self.sensors = {}

        for name in self.joint_names:
            motor = robot.getDevice(name)
            sensor = robot.getDevice(name + "_sensor")

            if motor is None or sensor is None:
                raise RuntimeError(f"Missing motor/sensor: {name}")

            motor.setVelocity(
                motor.getMaxVelocity() * speed_fraction
            )
            sensor.enable(self.timestep)

            self.motors[name] = motor
            self.sensors[name] = sensor

        self.gripper_motors = {}
        self.gripper_sensors = {}

        gripper_devices = {
            "gripper_left_finger_joint":
                "gripper_left_sensor_finger_joint",
            "gripper_right_finger_joint":
                "gripper_right_sensor_finger_joint",
        }

        for motor_name, sensor_name in gripper_devices.items():
            motor = robot.getDevice(motor_name)
            sensor = robot.getDevice(sensor_name)

            motor.setVelocity(motor.getMaxVelocity() * 0.15)
            sensor.enable(self.timestep)

            self.gripper_motors[motor_name] = motor
            self.gripper_sensors[motor_name] = sensor
            
        # Put this at the end of TIAGoArm.__init__,
        # after all sensors have been enabled.
        for _ in range(5):
            if self.robot.step(self.timestep) == -1:
                raise RuntimeError("Simulation ended during sensor initialization")
        
        # Validate all readings before accepting commands.
        for name, sensor in self.sensors.items():
            value = sensor.getValue()
            if not math.isfinite(value):
                raise RuntimeError(f"Invalid initial reading from {name}: {value}")
        
        for name, sensor in self.gripper_sensors.items():
            value = sensor.getValue()
            if not math.isfinite(value):
                raise RuntimeError(f"Invalid initial reading from {name}: {value}")
        
    def step(self):
        return self.robot.step(self.timestep) != -1
        
    def read_position(self, name):
       value = self.sensors[name].getValue()
       if not math.isfinite(value):
           raise RuntimeError(
               f"{name} returned {value}. "
               "Position sensor feedback is not valid."
            )
       return value

    def joint_positions(self):
       return {
           name: self.read_position(name)
           for name in self.sensors
       }

    def print_status(self):
        print("\nMeasured joint positions:")
        for name, position in self.joint_positions().items():
            print(f"  {name:20s}: {position: .3f}")

        print("\nJoint limits:")
        for name, motor in self.motors.items():
            print(
                f"  {name:20s}: "
                f"{motor.getMinPosition(): .3f} to "
                f"{motor.getMaxPosition(): .3f}"
            )

        print("\nGripper limits:")
        for name, motor in self.gripper_motors.items():
            print(
                f"  {name:30s}: "
                f"{motor.getMinPosition(): .4f} to "
                f"{motor.getMaxPosition(): .4f}; "
                f"current={self.gripper_sensors[name].getValue():.4f}"
            )

    def move_joints(self, targets, timeout_s=12.0, tolerance=0.025):
        # Validate every target before commanding any motor.
        for name, target in targets.items():
            if name not in self.motors:
                raise ValueError(f"Unknown joint: {name}")
            if not math.isfinite(target):
                raise ValueError(f"Non-finite target for {name}")
    
            motor = self.motors[name]
            if not motor.getMinPosition() <= target <= motor.getMaxPosition():
                raise ValueError(f"Target outside limits for {name}")
    
        # Ensure feedback is valid before starting.
        for name in targets:
            self.read_position(name)
    
        for name, target in targets.items():
            self.motors[name].setPosition(target)
    
        max_steps = max(1, int(timeout_s * 1000 / self.timestep))
    
        for _ in range(max_steps):
            if not self.step():
                return False
    
            errors = {
                name: abs(self.read_position(name) - target)
                for name, target in targets.items()
            }
    
            if all(error < tolerance for error in errors.values()):
                return True
    
        print("Motion timed out; final joint errors:")
        for name, target in targets.items():
            actual = self.read_position(name)
            print(
                f"  {name}: target={target:.3f}, "
                f"actual={actual:.3f}, error={abs(actual-target):.3f}"
            )
        return False


    def move_home(self):
        return self.move_joints(self.HOME)

    def move_arm(self, angles):
        """Seven arm angles, ordered arm_1 through arm_7; radians."""
        if len(angles) != 7:
            raise ValueError("Expected exactly 7 arm joint angles.")

        return self.move_joints(dict(zip(self.ARM_NAMES, angles)))

    def set_gripper(self, targets, timeout_s=3.0, tolerance=0.002):
        """
        Targets are finger-joint positions in the model's joint units.
        Inspect print_status() before choosing open/closed targets.
        """
        for name, target in targets.items():
            if name not in self.gripper_motors:
                raise ValueError(f"Unknown gripper motor: {name}")

            motor = self.gripper_motors[name]
            if not motor.getMinPosition() <= target <= motor.getMaxPosition():
                raise ValueError(f"Target outside limits for {name}")

        for name, target in targets.items():
            self.gripper_motors[name].setPosition(target)

        max_steps = int(timeout_s * 1000 / self.timestep)

        for _ in range(max_steps):
            if not self.step():
                return False

            errors = [
                abs(self.gripper_sensors[name].getValue() - target)
                for name, target in targets.items()
            ]

            if errors and max(errors) < tolerance:
                return True

        print("Gripper motion timed out or was blocked.")
        return False
        
    
    def move_tool_to(self, ik_solver, xyz, timeout_s=12.0):
        """
        Move the tool to XYZ in the IK model's root frame.
    
        This version does not constrain tool orientation or check collisions.
        """
        current = [
            self.read_position(f"arm_{i}_joint")
            for i in range(1, 8)
        ]
    
        angles, achieved, error = ik_solver.solve_position(
            xyz,
            initial_angles=current,
        )
    
        targets = {
            f"arm_{i}_joint": float(angles[i - 1])
            for i in range(1, 8)
        }
    
        print("Requested XYZ:", xyz)
        print("Predicted XYZ:", achieved)
        print(f"IK position error: {error * 1000:.1f} mm")
    
        return self.move_joints(targets, timeout_s=timeout_s)



# Webots controller entry point

robot = Robot()
arm = TIAGoArm(robot)
ik = TIAGoIK()

# First: inspect the starting tool position.
print("Starting tool position:", ik.forward_position())

# Keep the simulation running.
while arm.step():
    pass