# WorldMind recorded demonstration

Use the successful arm-5 equal VALIDATION episode validation-clear-i00-s0,
seed 0. The recorded manipulation was attempt 2 after a missing-definitions
startup failure. The archive is from revision 150b05a and uses the executor
profile later selected by the freeze.
[Frozen manifest](../configs/worldmind_frozen_execution.json)

1. Open the [architecture figure](figures/architecture.pdf). Identify the
   scripted proposer, analytic prediction, predictive veto and motor check.
2. Play [demo-arm5-equal.mp4](figures/demo-arm5-equal.mp4).
   To regenerate it from the repository root, use FFmpeg and the project
   environment with a fresh output path. One frame represents one confirmed
   32 ms simulator tick.

   ```text
   uv run --offline python tools/render_worldmind_demo.py runs/worldmind/nov24-matched-arm5-equal/validation-clear-i00-s0/attempt-2/episode runs/worldmind/nov25-validation-report/demo-arm5-equal.mp4
   ```

3. Use the [phase montage](figures/webots-execution.png) to pause at PREGRASP,
   CLOSE, LIFT, TRANSIT, RELEASE and HOME. The [pool plot](figures/candidate-pool.pdf)
   shows vetoed draws excluded ahead of ranking.
4. Show the [terminal verification](../runs/worldmind/nov24-matched-arm5-equal/validation-clear-i00-s0/attempt-2/episode/verification-0.json).
   Verification completed at tick 635 with 15.24 mm final CAN error and a
   maximum HOME error of 0.01529 rad.
5. Close with the [18 paired outcomes](figures/paired-outcomes.pdf). This is a
   successful development example. The frozen arm achieved 14/18 overall
   versus the baseline's 15/18.

Playback reads existing development evidence and launches no simulator.
The generated sidecar records the ordered aggregate digest of source observations.
