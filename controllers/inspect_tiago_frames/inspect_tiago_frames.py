from controller import Supervisor
import math

robot = Supervisor()
dt = int(robot.getBasicTimeStep())

motor = robot.getDevice("arm_1_joint")
sensor = robot.getDevice("arm_1_joint_sensor")
sensor.enable(dt)

# Wait for the sensor to initialize.
for _ in range(5):
    if robot.step(dt) == -1:
        raise RuntimeError("Simulation ended")

start = sensor.getValue()
if not math.isfinite(start):
    raise RuntimeError("Invalid initial joint reading")

target = start + 0.05  # Small movement: 0.05 radians
motor.setPosition(target)
motor.setVelocity(0.15)

print(f"Start:  {start:.5f} rad")
print(f"Target: {target:.5f} rad")

# Allow up to about 3 seconds for the joint to move.
for _ in range(int(3000 / dt)):
    if robot.step(dt) == -1:
        raise RuntimeError("Simulation ended")

    current = sensor.getValue()
    if math.isfinite(current) and abs(current - target) < 0.01:
        print(f"Reached target: {current:.5f} rad")
        break
else:
    print(f"Did not reach target. Final reading: {sensor.getValue():.5f}")

# Return to the starting position.
motor.setPosition(start)

for _ in range(int(3000 / dt)):
    if robot.step(dt) == -1:
        break

print("Return position:", sensor.getValue())