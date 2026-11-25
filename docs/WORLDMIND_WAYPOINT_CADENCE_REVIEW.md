# WorldMind scripted waypoint cadence review (2026-11-17)

This is an inert review of the frozen `ScriptedCanPolicy` baseline against the production
`ScriptedChunkAdapter` path. `src/vlarobot/execution/scripted_can.py` was **not modified**. No
Webots run, real IK solve, inference, training, actuation, scored evaluation, TEST/TRAIN access or
pilot execution was performed.

## What was corrected safely

`corrected_scripted_waypoints()` now emits the baseline station order:

```text
INIT(open) -> PREGRASP -> DESCEND -> CLOSE(closed) -> LIFT -> TRANSIT ->
PLACE_DESCEND -> RELEASE(open) -> RETREAT -> HOME(joint target)
```

The previous adapter-only order was:

```text
PREGRASP -> GRASP_APPROACH -> LIFT -> TRANSIT -> PLACE_DESCEND ->
PLACE_RETREAT -> RELEASE
```

The corrected station list still sums to the frozen H=50 control ticks. `INIT`, `CLOSE`, `RELEASE`
and `HOME` are explicit station types; `HOME` is a reviewed joint target and is still safety-validated
before it enters a chunk. `ScriptedWaypoint` accepts per-station IK tolerances, populated from the
existing reviewed `IK_TOL`, `IK_TOL_GRASP` and `IK_TOL_TRANSIT` settings. `worldmind_pipeline.py`
re-derives waypoints from the latest oracle state at every decision instead of once per episode,
matching the baseline's world-anchored re-derivation pattern.

These are inert, deterministic corrections only. They do not claim settle-equivalence, contact
equivalence, IK equivalence or physical success.

## Dimension-by-dimension comparison

| Dimension | Frozen baseline (`scripted_can.py`) | Adapter before correction | After safe correction | Status |
|---|---|---|---|---|
| Station order | INIT(open), PREGRASP, DESCEND, CLOSE, LIFT, TRANSIT, PLACE_DESCEND, RELEASE, RETREAT, HOME | PREGRASP, GRASP_APPROACH, LIFT, TRANSIT, PLACE_DESCEND, PLACE_RETREAT, RELEASE | baseline order emitted explicitly | Inert order aligned; timing not equivalent |
| Gripper timing | INIT opens until measured aperture; CLOSE issues `GripperCmd(0.0, force=50)` then steps until fingers stall/stop; RELEASE steps until open threshold | one fixed opening command at the end | explicit OPEN/CLOSE/OPEN stations at the correct sequence points, fixed hold steps | PHYSICAL VALIDATION REQUIRED: stall/settle detection is not reproduced |
| Orientation | Position-only IK with a measured tool tilt limit (`MAX_TILT_DEG`); no full-pose target asserted | identity full-pose quaternion supplied to IK | unchanged identity quaternion; per-station tolerances added | PHYSICAL VALIDATION REQUIRED: vertical-only constraint and identity tool pose are not the same claim |
| Tolerances | `IK_TOL`, `IK_TOL_GRASP`, `IK_TOL_TRANSIT`, `MAX_TILT_DEG`, align/close-align checks | one `(position, orientation)` tolerance pair for every station | reviewed state-specific position tolerances; orientation tolerance still from config | PARTIAL: tilt and alignment gates remain physical-validation items |
| Cadence | settle-driven joint interpolation and closed-loop waypoint re-targeting | fixed `hold_steps` summing to H=50; waypoints computed once per episode | fixed cadence summing to H=50; waypoints recomputed each decision | PARTIAL: fixed holds are a scripted proxy, not measured settling |
| Verification | final frozen criteria plus measured settle/rejection accounting | `PlacementVerifier` plus frozen final criteria in `worldmind_pipeline.py` | unchanged | Unchanged; physical validation required |
| Recovery | settle/IK failure, base-drift and close-alignment branches | `RecoveryState` (reobserve / alternate family / abort), verifier retry | unchanged; alternate family still unbound in production | Unchanged; alternate family is unavailable and fails closed |
| Failure labelling | explicit `failure_state` strings from the policy | adapter errors become `EXCEPTION` in the pipeline | unchanged | Documented; no unvalidated reclassification made |
| Execution boundary | baseline drives the sim directly through many steps | one chunk's first action crosses `execute_first` per decision | unchanged | Deliberate architecture limit; unexecuted horizon is never claimed executed |

## Why no further corrections were made

The remaining differences require real geometry, measured settling, contact feedback or an owner-
approved IK/orientation contract. Changing them inertly would invent physical facts. In particular:

- a fixed-gripper hold cannot be converted to stall detection without live aperture feedback;
- an identity tool quaternion cannot be declared equivalent to baseline tilt-only IK;
- `MAX_TILT_DEG`, `ALIGN_TOL`, `CLOSE_ALIGN_TOL` and settle tolerances are measured control gates,
  not values to re-label as Eq. 3.8 weights;
- production binding an alternate waypoint family or a full-arm sweep certificate needs an
  owner-reviewed certificate, not a test fixture;
- reclassifying adapter errors as recoverable `NO_SAFE_SET` would change evidence semantics without
  measured validation.

## Physical validation required before pilot authorization

1. Run the reviewed waypoint sequence on the real Webots geometry and compare achieved tool poses,
   tool tilt and settling time against the frozen baseline trace.
2. Confirm that the explicit CLOSE and RELEASE hold durations produce the same attachment and
   release outcome as stall-driven gripper control on the CAN and destination surfaces.
3. Confirm that the reviewed position tolerances are sufficient with the corrected registration and
   that identity tool orientation does not cause grasp or placement tilt failures.
4. Decide whether the production composition may bind an alternate waypoint family and whether a
   certified swept-arm check can be registered. Until then the `unverified-arm-sweep` veto remains
   hard and the pilot remains blocked.
