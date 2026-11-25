# WorldMind V2 recovery

Parent and immutable V1 checkpoint: `b2d67b0109eabfbc9e70a35aeec061ab50d41967`.
Integration dependency: [V1 PR #2](https://github.com/sridevkp/VLA-Project/pull/2)
remains open against main. V2 changes stay on `work/vla-recovery`.
The original PDF is the authoritative Phase I specification. Its SHA-256 is
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`.

## Proposal gaps

| Phase I requirement | V1 disposition and evidence | Recovery or completion criterion | Estimated work |
|---|---|---|---|
| §§1.3–1.4, 3.4.1, 4.2 simulation, goals, obstacles, repeatable control | Fully implemented in the pinned TIAGo factory world with eligible paired runs | Reuse Webots, scenario generation, guarded control, frozen provenance | P0, reuse |
| §§3.4.2, 4.3, Eq. 3.5 observation and scene state | Prototype implemented with oracle and RGB inputs. §4.3 allows simulator state for the initial prototype. Camera-derived task state remains missing | Preserve oracle as reference. Establish scope of camera-based state estimation required for the final claim and compare it to oracle if prioritized | P1 after proposer, 1–3 days for a narrow simulated detector and validation |
| §§3.3.2, 3.4.3, Eq. 3.2 VLA accepts imagery, language, context and prediction | VLA-only implementation experimentally unsuccessful at 0/24. V1 substituted scripted proposals and external ranking. SmolVLA has no native predicted-state input | Recover competency first. Then test whether a prediction-conditioned second decision can satisfy Eq. 3.2, with explicit evidence that changing prediction changes the decision. External ranking alone retains the deviation. A material interface/model substitution needs owner approval | P0, 2–4 days for admission investigation, 2–4 more for integration/conditioning if admission passes |
| §§3.3.1, 3.4.3–3.4.4, Eqs. 3.1, 3.6–3.7 feasible alternatives and prediction | Analytic prediction and seeded scripted alternatives measured. VLA-conditioned candidate pools missing in eligible V1 runs. Phase I leaves model selection open | Preserve the analytic model. Verify distinct executable VLA-derived alternatives and their exact predicted/observed correspondence. No learned residual required by the Phase I equations | P0, 1–2 days after admission |
| §§3.4.5, 4.6, Eq. 3.8 predictive constraints and ranking | Partial. V1 tool-envelope proxy, Eq. 3.8 ranking and independent motor veto were exercised. Full-body collision coverage and calibrated uncertainty remain absent. Ju has zero weight | Reuse supported proxy scope and report its boundaries. Quantify coverage and check controls under VLA inputs. Estimate uncertainty only if independent evidence supports it | P0 integration 1–2 days; broader geometry or calibration may exceed the deadline |
| §§3.3.3, 3.3.6, 3.4.6, 4.7–4.9, Eqs. 3.3–3.4 control, verification, feedback | Fully exercised for V1 scripted closed loop, with measured prediction/observation mismatch. VLA episode completion experimentally unsuccessful | Separate indication of return-to-HOME from task completion in a versioned VLA protocol. Verify independent command admission and a real closed-loop task before changing scientific arms | P0, about 1 day offline plus simulator evidence after proposal admissibility |
| §§3.1.2, 3.4.7, 5.3–5.9, Tables 5.1–5.2 traceability and comparison | V1 complete for its declared scope. No measured collision reduction. Timing and prediction pair coverage are limited. V1 TEST is exposed for V2 | Reserve new identities before outcome access, record missingness, and freeze V2 before independent execution. Keep V1 as regression evidence and preserve failed attempts | P0, 1 day protocol plus run time; use V1 measured 17–21 eligible trials/hour only as planning reference |
| §§1.4, 4.1, 6.2 reproducible simulation deliverables | V1 tracked docs, figures and demonstration complete within its scope. No mandatory app or physical-robot deployment in Phase I | Reuse report and demo infrastructure. Report deviations if competency, conditioning, or evidence cannot be recovered in time | P0 reporting, 2–3 days |

Large-scale AI training, physical-robot deployment, autonomous general-purpose
operation and a client application are outside the initial proposal scope.
Predictive integration remains a requirement. The sign of the observed effect
is an experimental result.

## Reproduced VLA failure causes

The new read-only audit verified all 225 historical artifacts. It reconciled 21
SAFETY exits and three failed placements following HOME detection. All 24 chunks
contain at least one out-of-range target. Of 1,200 proposed gripper values, 761
exceeded 0.045 m, with a maximum of 0.045792781 m. The existing limit worked.
The three HOME exits occurred after 4–7 accepted actions and under 0.3 μm
of CAN rise. The archived joint excursions cross the original 0.02 rad
departure and return predicate exactly. All episodes stopped in the first
chunk. Later grasp, lift and release capability remains unknown.
[Archived calculation and method](VLA_TRAIN_POSTMORTEM.md)

The archived 80 demonstrations contain 63,081 valid gripper labels at the
endpoints. The 4,000 initial labels are all open, so the absence of close in
historical first chunks is consistent with training. The saved normalizer and
inverse use matching affine statistics with a measured exact round trip.
The first-frame input records do not show a gross RGB, state scale or
pose-convention error. JPEG training versus RGB inference is a potential
conditional difference. Saved processor and checkpoint provenance establish
loading, not training convergence or language-conditioned competence.

The AD1 controlled replay gives a second distinction. The equivalent recorded
expert hold returns the exact original joints after IK. In the matched example,
the learned 0.67 mm residual produces a 0.15 rad accepted joint change.
That demonstrates sensitivity for this input, not a global IK defect.
[AD1 experiment](AD1_NUMERICAL_REPORT.md)

Run the independent reconciliation from this checkout while keeping the archive
read-only.

```text
uv run --offline python tools/audit_vla_train_admission.py ../VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1 docs/evidence/vla-train-step20000-files.json
```

## Critical path to December 12

1. **Nov 25–26.** Lock V1 and verify raw-archive checksums. Reserve V2 held-out
   identities with an independent custodian or sealed manifest before recovery.
   Use TRAIN-only matched frames from approach, grasp, lift and release to measure
   the existing checkpoint's predicted gripper endpoint, action residuals and
   sensitivity to RGB/instruction changes. Check loss and checkpoint statistics
   before spending on training.
2. **Nov 26–29.** Review a versioned admissible gripper representation and
   task-aware completion binding. Keep motor limits and independent safety veto.
   Test exact expert-command parity as a negative control for adapter mismatch.
   Run a separately identified TRAIN admission comparison after offline checks.
   Fine-tune on the admitted data only if phase-specific diagnostics justify
   the change. No V1 outcome is rescored. If the proposed change alters the
   research contract, obtain approval before activation.
3. **Nov 30–Dec 2.** If VLA clears admission, compose VLA-conditioned distinct
   proposals and existing prediction/ranking. Exercise the actual selected
   path and feedback. If admission fails, report this plainly and compare
   feasible alternatives for owner review before substituting architecture.
4. **Dec 3–6.** Prioritize Eq. 3.2 conditioning evidence and remaining mandatory
   gaps, then prepare the frozen comparison. Keep camera perception and
   uncertainty claims within what is measured.
5. **Dec 7–9.** Freeze V2 parameters and code before obtaining new held-out
   outcomes. Pair identical scene/seed identities. Preserve V1 TEST as exposed
   regression evidence only. At 17–21 eligible trials/hour, 120 V1-like trials
   would need about six to seven hours of elapsed campaign time at
   comparable total concurrency. Live inference and retries can increase that
   estimate. A V2 allocation and actual throughput have yet to be established.
6. **Dec 10–12.** Integrate measured V2 outcomes, disagreements, limitations and
   actual demonstration if available. Keep the V1 package independently
   reproducible if V2 cannot meet the original proposal.

Parallel dependencies: reserve the split and back up V1 while investigating
the checkpoint. Prepare perception evidence and final report crosswalk without
touching held-out outcomes. The gate for scientific VLA-primary integration is
competent learned control under the approved protocol.

## November 25 V2 execution evidence

The held-out commitment reserves 30 fresh scene/seed identities. It records
the digest while the private plan is withheld. No held-out outcomes were
obtained. The full TRAIN gate remains 10 successes out of the original 24
paired identities, from the existing 40% criterion and 23/24 baseline.
[Sealed plan commitment](evidence/worldmind-v2-holdout-commitment.json)

The same step-20000 checkpoint was probed at 21 archived observations from
successful TRAIN demonstrations. The two scenes were clear and cluttered.
Each sampled phase used the original JPEG pixels, state, and expert labels
in the original base-frame residual representation. The independent probe
also sampled the first actual open label during RELEASE. These are
teacher-forced comparisons. Later actions in each predicted chunk are
compared to the expert tape, without claiming the model would reach those
observations autonomously. [Phase evidence](../runs/worldmind/v2-phase-probe/)

- The checkpoint produced mostly near-closed targets in LIFT and TRANSIT
  and an open-like release for 49/50 proposals at the actual opening boundary.
  During two CLOSE observations with 50 closed expert labels each, it emitted
  1 and 6 open-like proposals. The 22.5 mm midpoint is descriptive only.
  Learned translation error against the expert was 12.28–13.37 mm in LIFT
  and 10.79–13.33 mm in TRANSIT. Command consistency remains a learned gap.
- The V2 projection maps all 761 historical out-of-range targets into
  the motor domain, holding pose components fixed. Some newly measured
  chunks remain outside the distinct projection envelope and fail closed.
- The first three closed-loop trials passed 44 actions each. V1 had
  accepted 1, 0 and 0 actions on those identities. All three V2 trials
  stopped on an IK translation residual. An exact observed witness reduced
  the residual from 9.22 mm to 1.54 mm, with 0.01985 rad orientation error,
  inside the same 0.15 rad joint trust region and independent admission.
  This demonstrates a mixed-objective solver trade-off on that input.
- The second three-scene run used the bounded fallback and reached a
  second model call in all three cases. Each accepted 50 actions and moved
  the TCP 72.6–74.3 mm net. Then all three failed closed on second-chunk
  gripper proposals, with maxima of 46.08–46.76 mm. Maximum observed CAN
  climb was about 2.1 μm. There was no observed grasp, lift, transport,
  placement, release or successful HOME in either diagnostic.
  Both results are **0/3 diagnostic**, with no full TRAIN gate run.

The next targeted comparison is an endpoint-native learned gripper output
on TRAIN-only phase windows, with independent motor limits and malformed-output
checks retained. Measure timing, reopening and residual behavior before
deciding whether a bounded output head or more extensive fine-tuning is
needed. One successful second query does not establish learned competence.
Expanding the decoder envelope to cover these particular failures does not
address the observed reopening during CLOSE. Eq. 3.2 integration remains
downstream of the admission gate.

## Integration and PR access

PR #2 remains bound to evaluated V1. Both official CI jobs failed at
`uv run ty check src`. The separate `work/v1-integration-fixes` branch
has a first behavior-preserving narrowing patch and nine passing focused
tests. The same local type-check command still reports 93 diagnostics,
down from 98. It is not ready to advance PR #2. The notebook-helper SHA
mismatch is separate.

From a machine with GitHub CLI and repository permissions, authenticate
without pasting a token into a shell command, then edit the existing PR.

```text
gh auth login --hostname github.com --web --git-protocol ssh
gh auth status --hostname github.com
gh pr edit 2 --repo sridevkp/VLA-Project --title "WorldMind V1: completed implementation, frozen evaluation and research findings" --body-file runs/worldmind/v2-train-admission-audit/v1-pr-description.md
```

Alternatively, sign in to GitHub, open PR #2, use the title and description
edit controls, and save the draft text from that file. No API credential
was available in this session. SSH branch pushes succeeded.
