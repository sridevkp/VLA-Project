
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

JOINT_NAMES = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
]

# Starting pose from the original TIAGo controller.
HOME = [
    0.09,
    0.07,
    0.26,
    -3.16,
    1.27,
    1.32,
    0.0,
    1.41,
]

motors = []
sensors = []

for name in JOINT_NAMES:
    motor = robot.getDevice(name)
    motor.setVelocity(motor.getMaxVelocity() * 0.25)

    sensor = motor.getPositionSensor()
    if sensor is None:
        raise RuntimeError(f"No position sensor for {name}")

    sensor.enable(timestep)

    motors.append(motor)
    sensors.append(sensor)

# Command the starting pose.
for motor, target in zip(motors, HOME):
    motor.setPosition(target)

print("Moving TIAGo arm to the starting pose...")

while robot.step(timestep) != -1:
    errors = [
        abs(sensor.getValue() - target)
        for sensor, target in zip(sensors, HOME)
    ]

    if max(errors) < 0.03:
        print("Starting pose reached.")
        break

# Hold the pose while the simulation continues.
while robot.step(timestep) != -1:
    pass
