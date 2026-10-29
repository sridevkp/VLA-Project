
from controller import Robot, Keyboard
from robo_server import WebotsHTTPBridge, capture_jpeg

from pathlib import Path
import tempfile
import math

bridge = WebotsHTTPBridge(
    host="127.0.0.1",
    port=8765,
)
bridge.start()

PLACE_BASE      = np.array([-1.07, -3.9, 0.8]) 
# --------------------------------------------------
# Configuration
# --------------------------------------------------
# Cameras
# rgb_camera = robot.getDevice("Astra rgb")
# depth_camera = robot.getDevice("Astra depth")

# rgb_camera.enable(TIME_STEP)
# depth_camera.enable(TIME_STEP)


TIME_STEP = 32

DRIVE_SPEED = 3.0       # Wheel angular velocity, rad/s
HEAD_STEP = 0.04        # Head movement per key event, radians
HEAD_PAN_SPEED = 0.7    # Maximum pan range is model-dependent
HEAD_TILT_SPEED = 0.7   # Maximum tilt range is model-dependent

robot = Robot()
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# --------------------------------------------------
# Get devices
# --------------------------------------------------

def get_device(name):
    device = robot.getDevice(name)
    if device is None:
        raise RuntimeError(
            f"Device '{name}' not found. Check the TIAGo model's device names."
        )
    return device


# Head
head_pan = get_device("head_1_joint")
head_tilt = get_device("head_2_joint")

# Mobile base
left_wheel = get_device("wheel_left_joint")
right_wheel = get_device("wheel_right_joint")

# Stationary torso and arm
fixed_joint_names = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
]

fixed_joints = [get_device(name) for name in fixed_joint_names]

# Cameras
rgb_camera = get_device("Astra rgb")
depth_camera = get_device("Astra depth")

rgb_camera.enable(TIME_STEP)
depth_camera.enable(TIME_STEP)

# --------------------------------------------------
# Initial pose from the original C controller
# --------------------------------------------------

initial_positions = {
    "head_1_joint": 0,
    "head_2_joint": -0.67,
    "torso_lift_joint": 0.09,
    "arm_1_joint": 0.07,
    "arm_2_joint": 0.26,
    "arm_3_joint": -3.16,
    "arm_4_joint": 1.27,
    "arm_5_joint": 1.32,
    "arm_6_joint": 0.0,
    "arm_7_joint": 1.41,
}

# Keep all non-wheel joints stationary at these targets.
for name, position in initial_positions.items():
    motor = robot.getDevice(name)
    motor.setPosition(position)
    motor.setVelocity(motor.getMaxVelocity() * 0.5)

# Continuous-rotation wheel motors.
for wheel in (left_wheel, right_wheel):
    wheel.setPosition(float("inf"))
    wheel.setVelocity(0.0)

# --------------------------------------------------
# Head joint limits
# --------------------------------------------------

pan_min = head_pan.getMinPosition()
pan_max = head_pan.getMaxPosition()
tilt_min = head_tilt.getMinPosition()
tilt_max = head_tilt.getMaxPosition()

pan_target = initial_positions["head_1_joint"]
tilt_target = initial_positions["head_2_joint"]

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def move_head(key):
    global pan_target, tilt_target

    # W/S: tilt up/down.
    # A/D: pan left/right.
    if key == ord('W'):
        tilt_target += HEAD_STEP
    elif key == ord('S'):
        tilt_target -= HEAD_STEP
    elif key == ord('A'):
        pan_target += HEAD_STEP
    elif key == ord('D'):
        pan_target -= HEAD_STEP
    else:
        return

    pan_target = clamp(pan_target, pan_min, pan_max)
    tilt_target = clamp(tilt_target, tilt_min, tilt_max)

    head_pan.setPosition(pan_target)
    head_tilt.setPosition(tilt_target)


# --------------------------------------------------
# Mobile base control
# --------------------------------------------------

def stop_base():
    left_wheel.setVelocity(0.0)
    right_wheel.setVelocity(0.0)


def drive_forward():
    left_wheel.setVelocity(DRIVE_SPEED)
    right_wheel.setVelocity(DRIVE_SPEED)


def drive_backward():
    left_wheel.setVelocity(-DRIVE_SPEED)
    right_wheel.setVelocity(-DRIVE_SPEED)


def rotate_left():
    left_wheel.setVelocity(-DRIVE_SPEED)
    right_wheel.setVelocity(DRIVE_SPEED)


def rotate_right():
    left_wheel.setVelocity(DRIVE_SPEED)
    right_wheel.setVelocity(-DRIVE_SPEED)


# --------------------------------------------------
# Main loop
# --------------------------------------------------

print("""
TIAGo Python controller

Arrow Up/Down : Forward / backward
Arrow Left/Right: Rotate left / right
W / S         : Tilt head up / down
A / D         : Pan head left / right

The torso and arm remain at their initial target positions.
Click the 3D simulation window before using the keyboard.
""")

previous_key = -1


step_count = 0
while robot.step(TIME_STEP) != -1:
    step_count += 1
    key = keyboard.getKey()

    # Drive while an arrow key is active.
    if key == Keyboard.UP:
        drive_forward()
    elif key == Keyboard.DOWN:
        drive_backward()
    elif key == Keyboard.LEFT:
        rotate_left()
    elif key == Keyboard.RIGHT:
        rotate_right()
    else:
        stop_base()

    # Apply head movement once for each new key event.
    if key >= 0 and key != previous_key:
        move_head(key)

    previous_key = key
    
    # telemetry = {
        # "connection": "Webots",
        # "simulation": "Running",
        # "joints": [
            # sensor.getValue()
            # for sensor in arm_sensors
        # ],
        # "gripper": "Open",
    # }

    # bridge.publish(telemetry=telemetry)
    
    if step_count % 1 == 0:
        jpeg = capture_jpeg(
            rgb_camera,
            str(Path(tempfile.gettempdir()) / "tiago_frame.jpg"),
        )
        if jpeg is not None:
            bridge.publish(camera_jpeg=jpeg)