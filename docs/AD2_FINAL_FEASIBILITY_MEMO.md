# A-D2 final feasibility and prioritization decision memo

**Recommendation: stop A-D2 and preserve completed A-D1 findings (option3).**
Physical A-D2 remains **NO-GO**. This is a prioritization recommendation, not a new safety
waiver, experiment authorization or change to the frozen result. Preserve the isolated CPU
infrastructure and its60-test evidence; do not extend it into a live harness under this task.
The final **0/24, FAILED gate, scripted-primary** decision under ADR-0029 is unaffected.

## Evidence and collision-preflight feasibility

**Observed:** [A-D1](AD1_NUMERICAL_REPORT.md) established local numerical sensitivity:
A=B exactly; C differs by0.212248rad joint L2 for a0.670555mm/0.005184768rad requested pose
change. It did not measure physical response. The
[offline infrastructure report](AD2_CPU_INFRASTRUCTURE_REPORT.md) and
[static inventory](evidence/ad2-static-geometry-v1.json) enumerate59 URDF links,
30 with collision elements, and34 cached PROTO sources. Source enumeration is not a
complete conservative preflight. The remaining obligations differ:

| Area | Feasible offline work from existing sources | Remaining proof / measurement boundary |
|---|---|---|
| Nested table solids | Table lines59–88 plus SolidBox line137 define a box top and four feet; expand formulas using the world's4.5×1.5×0.75m size and compose all local transforms. This gap is tractable, not missing geometry in principle. | Need verified expansion/containment and collision-query implementation. Existing inventory only identifies instances; no clearance certificate exists. |
| Floor/support | Installed Plane schema explicitly describes infinite collision extent; rendering size cannot justify finite-floor culling. Nominal floor transforms are available. | Resolve actual plane normal/frame, distinguish permitted wheel/support contact from forbidden body contact, and bound tilt/base motion. Current logger cannot certify those contact pairs. |
| Self-collision/exclusions | URDF topology and primitive origins allow candidate adjacent/fixed-link pairs to be listed and reviewed. |58 adjacency records are **not** approved exclusions. Tiago defaults selfCollision FALSE; external-contact logging filters robot descendants. Neither establishes physical noninterference or justifies excluding every adjacent pair. |
| Moving controllers/objects | Resolve world overrides, PROTO controller propagation, nominal joint limits and conservative enclosing geometry. Far objects could be excluded only with a separation/reachability proof. | Runtime position, velocity, contact and control mode are not all known. Valve-controller name alone does not prove forced rotation: source lines23–37 treat argv1 as an angular stop and use zero torque in the interior. It is not a speed bound. Do not silently freeze other actors. |
| Robot/world transforms | Compose nominal URDF/PROTO transforms and the saved robot/object poses; compare existing frame-audit evidence. | Tool registration is not all-link registration. Head/passive-joint state, compliant/contact-dependent motion and effective expanded scene identity are incompletely measured. A live reader could verify current transforms, not reconstruct missing historical state. |
| Swept volumes | In principle construct interval reachable envelopes including motor/passive/base/obstacle motion, or an even broader all-reachable envelope, then prove separation. | Neither proof exists. Requested motor speed and SafetyMonitor's1s implied-velocity check are not a certified32ms displacement bound. Endpoint or straight-joint interpolation checks omit possible dynamics. Broad envelopes may simply reject the pulse. |

**Assessment:** existing sources support substantial offline geometry reconstruction, but
**do not presently establish complete conservative acceptance** for this reset and pulse.
This is not a proof such a certificate is impossible. It would require verified containment,
transform uncertainty and motion bounds; measurements can constrain current state but do
not themselves prove bounds between observations. Some historical controller/physics state
is unrecoverable from these recordings. No collision margin is supplied by this memo.
Startup/reset itself commands motors and advances physics: any future authorization must
cover that preparation, not just the final tick. “Observe-only after reset” is not a
command-free simulator experiment.

## Interpretability and incremental scientific value

**Established limitation:** historical object twists, full frame0 base twist and effective
PID/internal state are incomplete. Earlier reset fingerprints are not frame0 measurements.
Exact historical dynamic-state replay cannot be certified. New measurements cannot recover
those missing historical values.

A **weaker prospective matched-reset experiment** could be interpretable if explicitly
approved as such: measure all accessible current start states/settings, apply preregistered
reference and pairwise admission criteria, preserve missing-history labels, and restrict
claims to initial response under those starts. Unobserved PID/contact state remains a
confound unless its initialization or conservative influence is established. Missing object
twists might be irrelevant to arm tracking only after separation/contact independence is
justified; their irrelevance cannot be assumed for safety.

Because A=B, the physical bench has **two distinct commands**, not three independent
control-path treatments. Its saved-target replay bypasses online IK, so A/B does not test
an additional physical IK path. A/B disagreement would expose session/state/measurement
inconsistency; agreement would not establish a reproducibility distribution or bound the
third session's variability. One C response can document command-versus-one-tick motion,
but cannot identify motor tracking as a cause rather than reset/dynamics variation.
Ordinary acceleration/lag is not necessarily a tracking fault. A longer hold, replication
or changed design would require a separate protocol and is not approved here.

**Judgment:** A-D2 could add a descriptive initial-motion observation beyond A-D1, but the
proposed three pulses cannot establish convergence, manipulation success or causation of
the failed HOME episodes. It cannot distinguish learned-policy error from IK sensitivity:
C's changed joint command already incorporates both. Its incremental causal value is low
relative to its unresolved safety and matching obligations.

## Minimum evidence needed if the owner later reopens physical work

1. Confirm the candidate submitted Phase I revision and explicitly choose prospective
   partial observability versus keeping exact-replay requirements (currently infeasible).
2. Provide justified, approved reset/measurement criteria and admissible uncertainties;
   no exploratory tolerance is activated. Obtain applicable repeatability/resolution
   evidence where absent under a separately authorized measurement scope.
3. Establish scene/geometry closure, all relevant transforms, approved self/support-contact
   treatment and conservative reset/pulse swept bounds with a justified clearance criterion.
   Do not substitute an endpoint pass or a physical trial for this preflight.
4. Verify non-mutating, same-instant live instrumentation and effective motor/controller
   settings, tick/ack accounting and cleanup. CPU fixtures do not verify those APIs. Any
   live verification needs its own authorization, including preparation steps/commands.
5. Approve a narrowly stated scientific question, confound treatment and stopping rules;
   separately authorize the physical protocol only after the preceding gates are satisfied.

These are evidence/authorization gates, not instructions to collect more data now.

## Deadline and stop/go rationale

[ADR-0029](DECISIONS.md:1305) fixes December12 and selects scripted-primary after the failed
gate. A-D2 is supplemental; neither it nor a different diagnostic can revise the accepted
campaign. Remaining work is not merely executing three ticks: geometry expansion and
containment, exclusion review, dynamics bounds, applicable reset criteria, live-reader
verification and a physical harness remain. No measured engineering-duration estimate
exists; assigning completion hours or guaranteeing deadline feasibility would be invented.

**Priority judgment:** those open-ended obligations compete with the contracted world-model,
RiskGate, integration and reporting deliverables without changing proposer selection.
Continue those only under their applicable authorizations; this memo does not authorize
new arms, validation/TEST access or other experiments. A smaller one-command pulse would
retain the safety obligations and lose even the limited A/B control; redesign is not the
preferred deadline tradeoff. Stop A-D2, retain A-D1's explicitly local numerical conclusion,
and report the unmeasured physical mechanism as a limitation. Owner may decide otherwise,
but no further A-D2 work is inferred from this review.

Read-only provenance: candidate PDF SHA-256 remains
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`;
PDF pp25–26/48–49 support checked execution, repeatability and honest outcome logging,
not invented tolerances. Its status remains candidate authoritative revision, not
independently authenticated submission. Static assets are pinned in the existing inventory.
Additional inspected installed sources: `projects/objects/factory/valves/controllers/valve_turner/valve_turner.py`
SHA-256 `ca46815b5c59c9910674eb13f41cf0ef93a6ffbaab6e41fc15ee421ab5040ec5`;
`resources/nodes/Plane.wrl` SHA-256
`0b6f97c3e02a91bf0b3d54cd83a35d984e3891633a78c2ebc3e007a8aab53782`, both under
`/home/tj330/.local/opt/webots/`. These sources were read, never imported/executed.
Only documentation/whitespace/link checks accompany this memo; no tests, simulations,
IK, inference, actuators or evaluations ran. Original code and evidence remain untouched.
