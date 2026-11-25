# WorldMind equations and frozen implementation

The controlling Phase I report, *WorldMind: World Model-Based Robot Simulation*, has SHA-256
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`.
The frozen execution [manifest](../configs/worldmind_frozen_execution.json), committed as
`e6cc0bb`, has SHA-256
`d6e5e7e7cc767d8784b208be9ba2f49d292935d8285f2954ae55c15baf027176`.
The submitted PDF numbers equations 3.1–3.8. The additional numbers mentioned in the
steering notes have no counterpart in that PDF.

| Eq. | Mathematical role | Evaluated implementation |
|---|---|---|
| 3.1 | Predict future states from a current state and an action sequence | [AnalyticSurrogate](../src/vlarobot/world_model/rollout.py) emits initial state plus future states for each complete rollout |
| 3.2 | VLA policy conditioned on imagery, language, context and prediction | ADR-0029 replaces policy-internal prediction with external propose-and-evaluate. After the 0/24 VLA-only TRAIN gate, the actual proposer in arms 3–5 is the scripted chunk adapter |
| 3.3 | Turn selected actions into controls | The [stateful caller](../src/vlarobot/execution/worldmind_stateful_chunk.py) follows expert phase and waypoint feedback, and independently validates a command before apply |
| 3.4 | Compare prediction to observation | [Prediction errors](../src/vlarobot/world_model/evaluation.py) use timestamp-aligned confirmed prefixes and matched state identities |
| 3.5 | Construct state from observations | [Oracle binding](../src/vlarobot/execution/worldmind_pipeline.py) builds the task state with the full observed base transform |
| 3.6–3.7 | Generate candidates and evaluate predicted trajectories | [Seeded generator](../src/vlarobot/planning/candidates.py) preserves draw order. The [phase-aware scorer](../src/vlarobot/execution/worldmind_stateful_proxy.py) evaluates held-waypoint chunks |
| 3.8 | Combine goal, path, risk and uncertainty costs subject to hard constraints | [RiskGate](../src/vlarobot/risk/gate.py) applies hard checks before minimizing the applicable weighted components. Ties preserve draw order |

## Analytic prediction and execution

Each pool has 32 ordered candidates. A complete candidate has 50 actions at 32 ms
(1.60 s), with 51 predicted states including the initial state. The commands in a
chunk hold the same waypoint target. For each joint, the model computes

```text
Δqⱼ = clip(q*ⱼ − qⱼ, −vⱼ Δt, +vⱼ Δt)
qⱼ′ = qⱼ + Δqⱼ,     q̇ⱼ′ = Δqⱼ / Δt
T_base,tool′ = FK(q′)
```

The tracking velocities come from registered URDF bindings. The gripper uses a
bounded approach to its commanded opening, clipping the increment to
±0.03 m/s × Δt. For objects with a configured support, the ballistic update uses
`p′ = p + v Δt − (g Δt² / 2) ẑ` with `g = 9.81 m/s²`, then applies an
inelastic support clamp. The binding supplies support for CAN. Unknown-support
objects retain their poses and emit a reason. The model can propagate an explicitly
supplied rigid attachment until release, but the evaluated phase-ranking call supplies
no attachment and never infers a new grasp. It does not simulate contacts, friction,
attachment acquisition or released angular dynamics. [Bindings](../src/vlarobot/backends/worldmind_bindings.py)

The current expert phase persists across chunks. The scorer uses the executable
prefix ending at predicted waypoint completion or tick 50. Execution can finish
the waypoint earlier or replan at prefix expiry from a fresh observation. The 50-action
horizon and the 1,059-tick episode ceiling serve different purposes. Terminal HOME
uses the configured verification joint vector and a finite inclusive maximum error
of 0.02 rad. Final verification uses CAN position error ≤0.10 m, absolute height error
≤0.02 m, tilt <15°, and the same HOME limit. [Frozen settings](../configs/worldmind_diagnostic.approved.yaml)

## Frozen objective

```text
J = 1 Jg + 1 Jl + 1 Jr + 0 Ju   (minimize)
Jg = ||p_tool(k_place) − p_tool,goal||₂
Jl = Σ ||p_tool(k) − p_tool(k−1)||₂ over the active prefix
Jr = mean(clip((0.03 m − d_k) / 0.03 m, 0, 1))
```

- **Jg** uses metres without further normalization. The goal is the frozen placement
  destination expressed in base axes with the scripted tool-height offset. The
  binding is the first predicted joint settle at terminal `PLACE_DESCEND`. Earlier
  chunks omit the component. Arm 5 vetoes missing required placement evidence or
  predicted tool error above 0.10 m. This is separate from final observed CAN verification.
- **Jl** uses metres without further normalization. The time coefficient is zero
  under the approved binding. A common nominal time term at fixed H could not
  reorder the candidate pool.
- **Jr** is dimensionless. The implemented soft envelope samples distances to
  scene boxes in the base-frame proxy used by
  [worldmind_objectives.py](../src/vlarobot/eval/worldmind_objectives.py).
  This soft path includes the target CAN. The separate hard geometry excludes CAN
  as the declared grasp target. The 0.03 m divisor is the soft deficit scale.
- **Ju** has weight zero because no independently justified uncertainty-scale bundle
  was approved. The analytic rollout is deterministic, and its measured errors
  are reported separately. A zero contribution does not establish zero uncertainty.

Calibration used observed success across all 18 frozen VALIDATION trials per
induced policy. `equal` and `goal_x2` each scored 14/18. The approved deterministic
policy-ID tie-break selected `equal`. These same-set rates are development evidence.
Arm 4 minimizes the phase objective without the predictive veto. Arm 3 uses
`finite_chunk_v1` and first-command admissibility, then selects the first eligible
draw without prediction or Eq. 3.8 ranking. Its comparison with arm 4 changes both
prediction availability and selection rule.

## Hard veto, geometry and LIFT

For arm 5, a complete rollout must satisfy these applicable checks before ranking.

| Check | Frozen implementation |
|---|---|
| Joint limit | Minimum distance to lower/upper limits over predicted states, plus 0.0001 rad numerical allowance in RiskGate only |
| IK | Existing phase-specific feasibility, position residual and tilt limit. The expert may reject an unreachable waypoint before a pool exists |
| Table | World-frame tool-tip height above the 0.75 m table plane. PREGRASP, LIFT, TRANSIT and RETREAT use their phase floors. DESCEND and PLACE_DESCEND use path floors and, at applicable terminal states, contact-height windows |
| Non-target boxes | Tool segments transformed into each oriented HAM/APP box frame. Clearance of the 3.8 mm tip envelope must meet the 22.4484 mm operating margin |
| Placement | Applicable at terminal PLACE_DESCEND. Required evidence must exist and meet the predictive 0.10 m tolerance |

The [clearance profile](../configs/worldmind_arm5_matched_clearance.json) contains the
exact phase floors and windows. It grounds the table frame in the verified table
geometry. The tool-envelope scope excludes other arm links, background world geometry
and certified collision coverage. The independent motor SafetyMonitor remains in force.

Only the first LIFT waypoint after CLOSE can use the
[LIFT exception](../configs/worldmind_stateful_lift_recovery.yaml). Its initial height
must be strictly above the geometric plane. A below-floor start may proceed only
when predicted tick one and every remaining prefix endpoint reach the unchanged
80.1658 mm LIFT floor and all other checks pass. The confirmed first observation
must reach that floor with no reported contact. Failure terminates the episode.
There is no second exception for that waypoint. Three live VALIDATION attempts
met this observed criterion. This establishes the reported simulation behavior
within the geometric proxy scope.

## Measured errors and costs

Per-quantity RMSE is `sqrt(mean(e²))` and MAE is `mean(|e|)`. Units are metres
for tool and object translation and radians for joint and object angle. Object
translation uses the RMS across matched object identities at each step, and angular
error uses the shortest quaternion angle. The VALIDATION horizon tables only retain
confirmed, aligned, complete executable prefixes. Executed tool path is the sum of
consecutive base-frame TCP distances. Simulated duration is final minus initial
simulator time. Wall time and decision latency are reported separately.
[Results and interpretation](RESULTS.md)
