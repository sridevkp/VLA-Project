"""Configuration for the TIAGo Webots pick-and-place controller."""
from pathlib import Path

TIME_STEP = 32
CAN_DEF = "CAN"
PLACE_BASE = [-1.07, -3.9, 0.8]  # target can centre in robot-base coordinates
GRASP_Z = 0.02
APPROACH_H = 0.12
PLACE_CLEAR = 0.01
IK_TOL = 0.015
IK_TOL_GRASP = 0.005
ALIGN_TOL = 0.005
SKIP_ALIGN_CHECK = True
MAX_TILT_DEG = 15
GRIP_FORCE = 16.0
MIN_GRASP_OPEN = 0.004
LIFT_MIN = 0.04
MAX_RETRIES = 2
STOP_AFTER = None
TORSO_HEIGHT = 0.35
FINGER_SPEED = 0.03
CALIBRATE_TOOL = False
FINGERS_INVERTED = False
GRIPPER_TEST = False
HEAD_PAN = 0.0
HEAD_TILT = -0.77
ARM_JOINTS = [f"arm_{i}_joint" for i in range(1, 8)]
FINGER_JOINTS = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
TORSO_JOINT = "torso_lift_joint"
HEAD_JOINTS = ["head_1_joint", "head_2_joint"]
TOOL_OFFSET = [0.0, 0.0, -0.1741]
CONTROLLER_DIR = Path(__file__).resolve().parent
URDF_PATH = CONTROLLER_DIR / "tiago.urdf"
