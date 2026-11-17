# TIAGo Webots Backend

The existing controller modules live in this package without changing the scripted
pick-and-place state machine. `controller.py` retains the original script as a comparison
baseline. `server.py` is the default Webots-side entry point and `backend.py` adapts Webots
devices to the simulator-independent `RobotBackend` interface. Set `VLAROBOT_WEBOTS_MODE=legacy`
before launching Webots to run that original script directly instead of starting the server.

## Running

Install the project with `uv sync --extra sim --extra dev`, configure the Webots controller
environment as described in `docs/SETUP.md`, and run a world whose TIAGo controller is named
`tiago_controller`. The controller shim starts the ZeroMQ REP server. In a separate terminal,
run `uv run python -m vlarobot.cli brain` to connect and advance the simulator in lockstep.

## Configuration and limitations

- Controller defaults are stored in `configs/robot.yaml`; `config.py` exposes compatibility aliases.
- Camera extrinsics are a placeholder until the M3 calibration task.
- Webots scenario spawning is not implemented before M2. `reset` restores the original CAN and
  arm configuration from the loaded world; it does not create randomized scenes.
- Supervisor access to `CAN` is ground truth for metrics/debugging only and must not feed a
  perception-driven task policy.
