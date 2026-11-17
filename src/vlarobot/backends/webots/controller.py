"""Original scripted pick-and-place controller retained as the comparison baseline."""

from vlarobot.backends.webots.config import CAN_DEF
from vlarobot.backends.webots.frames import Frames
from vlarobot.backends.webots.gripper import Gripper
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.motion import Motion
from vlarobot.backends.webots.pick_place import PickPlace
from vlarobot.backends.webots.robot_io import RobotIO
from vlarobot.backends.webots.visualization import Visualization


def main() -> None:
    io = RobotIO()
    io.initialize_can(CAN_DEF)
    ik = Kinematics()
    frames = Frames(io, ik)
    visualization = Visualization(io, frames)
    gripper = Gripper(io, visualization)
    motion = Motion(io, frames, ik, visualization)
    task = PickPlace(io, frames, ik, motion, gripper, visualization)
    if task.initialize_posture():
        task.run()


if __name__ == "__main__":
    main()
