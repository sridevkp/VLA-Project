"""Webots controller entry point: dependency wiring only."""
from config import CAN_DEF
from robot_io import RobotIO
from kinematics import Kinematics
from frames import Frames
from visualization import Visualization
from gripper import Gripper
from motion import Motion
from pick_place import PickPlace

def main():
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
