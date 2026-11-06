# VLA Project: Current State Overview

This repository is a robotics and vision-language control project built around a TIAGo robot in Webots, with supporting assets for IK generation and an experimental VLA model test harness.

## 1. Project purpose

The codebase combines:

- a Webots robot controller for TIAGo pick-and-place behavior
- a modular kinematics and motion stack
- calibration and debugging utilities for the Robot arm
- a UI bridge for camera and command streaming
- experimental model-side inference using a Hugging Face SmolVLA policy

The practical goal appears to be: use a high-level VLA policy or symbolic task description to command a robot that can reach, grasp, move, and place objects in a simulated factory environment.

## 2. High-level structure

### Top-level directories

- `controllers/` — robot control, debugging, and motion scripts
- `model/` — VLA / LeRobot model inference prototype
- `tiago_ikfast/` — URDF/DAE assets and IKFast preparation notes
- `worlds/` — Webots world files
- `plugins/` — robot UI / browser plugin assets
- `docs/` — project documentation

### Key files

- `controllers/tiago_controller/tiago_controller.py` — main controller entry point
- `controllers/tiago_controller/config.py` — main tuning/configuration file
- `controllers/tiago_controller/pick_place.py` — high-level task state machine
- `controllers/tiago_controller/kinematics.py` — IKPy-based inverse kinematics and scoring
- `controllers/tiago_controller/motion.py` — joint-space and Cartesian motion
- `controllers/tiago_controller/gripper.py` — gripper open/close and heuristics
- `controllers/tiago_controller/robot_io.py` — Webots device initialization
- `controllers/tiago_controller/frames.py` — coordinate transforms
- `controllers/tiago_controller/visualization.py` — scene markers / UI hooks
- `controllers/tiago_controller/ui_bridge.py` — camera/UI bridge implementation
- `model/main.py` — SmolVLA inference smoke-test entry point

## 3. Controller architecture

The runtime controller is organized into a modular pipeline:

1. `RobotIO` initializes the Webots Supervisor and all required motors/sensors.
2. `Kinematics` builds an IKPy chain from the URDF and allows solving for valid arm poses.
3. `Frames` translates between robot-base and world coordinates.
4. `Motion` executes either joint-space or linear Cartesian moves.
5. `Gripper` handles finger motion and a stall-based grasp heuristic.
6. `PickPlace` runs the state machine for the object manipulation workflow.
7. `Visualization` updates debug markers and optionally streams camera data to the UI.

The main controller entry point wires these together in a clean dependency chain:

- `RobotIO -> Kinematics -> Frames -> Visualization`
- `Gripper -> Motion -> PickPlace`

This is a standard robotics separation of responsibilities: device layer, planning layer, task layer, and visualization layer.

## 4. Current behavior of the TIAGo controller

The active task flow is a pick-and-place loop in simulation. Based on the `PickPlace` state machine, it does the following:

- initialize torso and head pose
- find the can target in the world
- open the gripper
- move to a pregrasp pose
- descend to the grasp point
- close the fingers
- lift the object
- move to a place target
- open the gripper to release
- retreat back to home

States in the workflow include:

- `INIT`
- `PREGRASP`
- `DESCEND`
- `CLOSE`
- `LIFT`
- `TRANSIT`
- `PLACE_DESCEND`
- `RELEASE`
- `RETREAT`
- `HOME`
- `RETRY`
- `DONE` / `FAILED`

This is not just a trivial move script; it is an automation loop with retries, pose evaluation, and alignment checks.

## 5. Kinematics and motion implementation

The kinematic layer uses `ikpy.Chain` with a URDF loaded from `controllers/tiago_controller/tiago.urdf`.

Important implementation notes:

- Only arm joints are enabled for IK solving.
- A `TOOL_OFFSET` is used to model the tool tip relative to the wrist.
- A `score()` function evaluates solution quality using position error and tool tilt.
- `move_linear()` performs interpolation between waypoints and keeps the motion within allowed IK tolerances.

The controller chooses not to blindly trust every IK solution; it rejects solutions that are too tilted or too inaccurate according to configured tolerances.

## 6. Gripper logic and safety heuristics

The gripper module is intentionally simple but practical:

- opens and closes by setting the finger joint positions
- checks whether the motion reached the target position
- uses a stall-based heuristic to decide if an object was grasped
- tracks finger opening and compares against `MIN_GRASP_OPEN`

This makes the controller usable for simulation, but the documentation explicitly notes that the grasp detection is heuristic and should be validated in the 3D view.

## 7. World / simulation setup

The repository includes world files:

- `worlds/factory.wbt`
- `worlds/factory_backup.wbt`

These appear to define the robot environment used by the TIAGo controller, with the target object (`CAN`) expected to exist in the simulator.

The controller also expects the world to contain a `DEF CAN` node and warns if it is missing.

## 8. UI and debug tooling

The project is equipped with a UI bridge for robotics debugging:

- `plugins/robot_windows/ai_interface/ai_interface.html` provides a browser-based interface
- `controllers/tiago_controller/ui_bridge.py` sends camera frames and robot messages through Webots RobotWindow APIs
- `visualization.py` creates markers for target and tool positions in the scene

The UI is useful for:

- checking camera frames
- sending commands to the robot
- observing system status in a lightweight browser UI

## 9. VLA / model side

The `model/` directory is separate from the Webots controller and is configured around Hugging Face LeRobot tooling.

The script in `model/main.py` does the following:

- imports `SmolVLAPolicy`
- loads a pretrained model from `lerobot/smolvla_base`
- checks the model’s expected input/output features
- builds pre/post processors for the model
- creates synthetic observations
- runs a forward pass to generate an output action

This indicates the repo is exploring a future or parallel architecture where a VLA policy could generate high-level task commands, while the Webots controller handles precise simulation execution. In other words, the repository is not purely a classical robotics controller; it is also a VLA experimentation scaffold.

## 10. IKFast and robot model assets

The `tiago_ikfast/` folder contains a simplified TIAGo arm model for IKFast usage. Its README documents how the team prepared a simplified URDF / DAE workflow, including:

- removing Gazebo and transmission tags
- creating a simplified arm URDF
- converting it to a COLLADA file
- rounding values for IKFast compatibility

This is a preparation asset for fast analytic inverse kinematics. It points to a deeper robotics pipeline beyond the main controller.

## 11. Current repo status and main caveats

From the code and comments, the repository is in a functional research/prototype stage. The README notes several important caveats:

- the controller is a structural refactor rather than independent validation of the IK model
- `TOOL_OFFSET` must remain calibrated to the tool geometry
- can placement is interpreted in robot-base coordinates
- the gripper grasp detector is heuristic
- supervisor access to `CAN` is simulation ground truth, not a real perception pipeline
- automatic calibration is still a follow-up task in some places

In short: the project is actively engineered and partially operational, but it still depends on careful tuning and simulation-specific assumptions.

## 12. Likely intended flow for a user

If someone wanted to work with this repo, the expected path is:

1. Load the TIAGo world in Webots.
2. Set the controller to the Webots controller script in `controllers/tiago_controller/`.
3. Tune `config.py` for target poses, tool offset, and tolerances.
4. Run the controller to test pick-and-place in simulation.
5. Use the debug tools and UI bridge to inspect motion and grasp alignment.
6. Optionally experiment with the model-side VLA inference in `model/main.py` as a higher-level policy prototype.

## 13. Summary

The repository is best described as a TIAGo robot simulation project with a modular controller stack, a VLA experimentation path, and supporting IK / calibration assets. The code is organized around solving robot motion problems robustly in Webots, while also preparing a path toward model-based or instruction-driven robot behavior.

The project is an active, research-oriented codebase rather than a final production system.
