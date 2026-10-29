# Modular TIAGo Webots Pick-and-Place Controller

## Modules
- `tiago_controller.py`: entry point and dependency wiring.
- `config.py`: constants, device names, paths and tolerances.
- `robot_io.py`: Webots devices, sensors and low-level setup.
- `kinematics.py`: IKPy chain, forward kinematics, IK and scoring.
- `frames.py`: world/base coordinate conversion and current tool position.
- `motion.py`: joint-space and Cartesian waypoint movement.
- `gripper.py`: opening, closing and stall-based grasp heuristic.
- `visualization.py`: debug markers and optional UI bridge.
- `pick_place.py`: high-level pick-and-place state machine.

## Installation
1. Copy all `.py` files into the same Webots controller directory.
2. Put your exported URDF in that directory as `tiago.urdf`, or change `URDF_PATH` in `config.py`.
3. Keep your existing `ui_bridge.py` beside these files if you want the camera UI.
4. Set the Webots controller name to `tiago_controller`.
5. Ensure the world contains `DEF CAN`, or update `CAN_DEF`.

## Debug sequence
Set `STOP_AFTER` in `config.py` to `"INIT"`, then `"PREGRASP"`, `"DESCEND"` and `"CLOSE"` before trying a full run. Set it to `None` for a complete run.

## Notes
- This is a structural refactor of the supplied code, not independent validation of the IK model or grasp physics.
- `TOOL_OFFSET` is retained as supplied and must remain calibrated to the simulated tool/finger geometry.
- `PLACE_BASE` is retained as supplied and is interpreted in robot-base coordinates.
- The finger-stall grasp detector is heuristic; verify that the can actually follows the gripper during lift.
- Supervisor access to `CAN` is simulator ground truth, not a perception method for a deployed VLA system.
- The automatic calibration helper from the original script has been left as a follow-up task; `CALIBRATE_TOOL` currently prints a note rather than changing the tool offset.
