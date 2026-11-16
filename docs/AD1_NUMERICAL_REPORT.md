# A-D1: one matched TRAIN snapshot, exactly two CPU IK solves

**Completed within the explicit A-D1 authorization.** Cases A/B/C use approved
TRAIN episode0/frame0, source attempt0000, scene `train-clear-i00-s100` and the
previously saved step20000 action0. Exactly **two** new IK solves occurred: B and C.
No neural inference, simulator, actuator command, scored evaluation, projection,
training, dataset/checkpoint mutation or withheld-split access occurred. No A-D2
proposal or execution follows this report. **0/24, gate FAILED, scripted-primary
remain unchanged.** The missing Phase I report remains a conformance blocker.

## Provenance, snapshot and solver checks

The native Parquet file matches its approved integrity hash. Its first row, raw
recording and saved model input agree exactly after the declared float32 state
serialization. Encoded JPEG SHA and decoded RGB SHA match the prior local inference
record. The saved action chunk matches its earlier hash and checkpoint-manifest
identity. Source commands, numeric steps, INIT trace and metadata match original
provenance, with the authorized scene/seed/frame. No substitute observation, policy
sample, seed or scene was used.

The recorded INIT `commanded_joint_target_rad` reference is q-star=q0. There is **no
explicit JointTarget apply event at frame0**; the first accepted event opens the
gripper, while the exporter initializes the held joint-target reference from q0.
Thus A is the recorded controller hold reference, corroborated by command/label
replay—not independent evidence of a new actuator command at that instant. This
matches the approved initial-hold design and limits inference about motor behavior.

Recorded torso position **0.3497446130613471m** was preserved in the passive seed,
not replaced with configured0.35m. Registered FK agrees with the recorded starting
pose to float64 roundoff; the harness uses128·eps·matrix-scale solely for this
numerical identity check, not as a safety tolerance.

Both solves used the actual frozen brain environment, Python3.11.18,
**IKPy4.4.4 / NumPy2.4.6 / SciPy1.17.1 / SymPy1.14.0**, matching the approved
source archive's lock. The ML environment's IKPy4.5.0 was **not used** for either
solve; that interpreter only extracted the Parquet row. Frozen solver/wrapper,
URDF, registration, configuration and relevant source hashes were verified. The
harness checked the actual solver-entry initial vector, arm-only active mask,
orientation_mode=all and per-joint trust bounds before each call. No multi-start,
restart, alternate seed or regularization setting was introduced.

Registration SHA remains
dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0.
The real ActionAdapter's decode and SafetyMonitor checks were used, without apply.
Trust region0.15rad, residual limits0.005m /0.0872664626rad, gripper bounds0–0.045m
and command horizon1s are unchanged. All source/export fingerprints were rechecked;
all225 original campaign files also match their sealed inventory after the audit.

## Observed A/B/C results

A is the original hold target with no IK. B reconstructs its pose using registered
FK, composes the equivalent expert residual through the unchanged adapter, and
solves from q0. C composes saved learned action0 with the recorded starting pose
and uses the same adapter/IK from q0. B's reconstructed solver target equals
FK(q-star) to roundoff. The original saved C values were not modified.

Saved action0:
`[-0.00040359731, -0.00053247676, 0.00005675969, -0.00076539547, 0.00143330730, 0.00492357789, 0.04490866512]`.

The learned request differs from the expert hold by **0.670555mm translation** and
**0.005184768rad rotation** (about0.2971deg). Its gripper target is legal. Native
expert action0 is six zero pose residuals plus the stored float32 open endpoint.

| Case | Result | Max joint change from A | Joint L2 change from A | Translation residual to its target | Orientation residual, frozen solver metric |
|---|---|---:|---:|---:|---:|
| A: recorded hold |Safety checks accepted; no solve |0 |0 |0 by FK reference definition |0 by reference definition |
| B: expert FK target→IK |Accepted |**0 exactly** |**0 exactly** |**0m** |6.989264e-8rad |
| C: saved learned residual→IK |Accepted |**0.15rad** |**0.2122484414rad** |**0.0009579472m** |**0.0001036976rad** |

B returns exactly q-star, so no spurious hold-branch displacement was observed in
B. Its nonzero acos-based orientation residual is a tiny floating-point metric
floor; the joint vector and FK position are unchanged. It is preserved, not
rounded away in the numeric evidence.

| Joint | A=B=q0, rad | C, rad | C−B, rad |
|---|---:|---:|---:|
| arm1 |0.0710397561 |0.0700000000 |−0.0010397561 |
| arm2 |−0.0000563301 |−0.0015776230 |−0.0015212929 |
| arm3 |−0.0000228965 |−0.1500228965 |−0.1500000000 |
| arm4 |0.0000219037 |−0.0023296580 |−0.0023515618 |
| arm5 |−0.0000055012 |−0.1500054831 |−0.1499999819 |
| arm6 |0.0000018481 |0.0061974754 |0.0061956273 |
| arm7 |−0.0000016805 |−0.0014491774 |−0.0014474968 |

C reaches arm1's lower position bound0.07rad and arm3's lower trust bound to
roundoff; arm5 is1.81045e-8rad above its lower trust bound. All remain within the
existing limits. This is an accepted bounded solution, not an acceptance-check
failure. Full margins, target/achieved matrices and residual vectors are retained.

### Observed objective trade-off, not a tested correction

Against C's target, the original A hold already has errors below the existing
5mm/5deg residual limits. Nevertheless, the current wrapper always invokes IK;
it does not implement a hold-if-within-tolerance shortcut. No such shortcut was
added or tried here.

The installed IK objective combines position error and rotation-matrix error
with existing orientation coefficient1; the wrapper supplies no regularization.
Arithmetic on the saved matrices gives:

| Evaluation of the existing objective for C target | Position squared term | Orientation-matrix squared term | Total squared |
|---|---:|---:|---:|
| At A's held joints |4.49644e-7 |5.37635e-5 |5.42132e-5 |
| At solved C joints |9.17663e-7 |2.25219e-8 |9.40185e-7 |

The combined objective decreases while position residual worsens from0.670555mm
to0.957947mm and orientation error shrinks. This is consistent with the installed
objective's trade-off, **not** evidence that its optimizer violated its objective.
These are the solver's mixed-unit residual terms, not a calibrated risk score or
the Phase I report's J utility. No weight or threshold was changed.

## Jacobian spectrum and local decomposition

An analytic base-frame spatial Jacobian was evaluated at q0 using the registered
chain's revolute axes. A centered-FK derivative cross-check with step
cbrt(float64 eps)=6.055454e-6rad agrees to maximum1.39872e-11. Those FK evaluations
are derivative arithmetic, not additional IK, action cases, simulation or rollouts.

| Jacobian | Singular values |
|---|---|
| Position3×7, m/rad |1.0476197864,1.0368669709,0.0174600853 |
| Rotation3×7, rad/rad |1.7320508169,1.4142143190,1.4142127997 |
| Stacked6×7, declared SI row convention |1.7511074483,1.7320508365,1.6975577291,0.4397912407,0.1772264858,4.6916295919e-7 |

Numerical rank is6 under cutoff `eps·max(shape)·s_max =2.72177e-15`—a roundoff
rank rule, **not** a calibrated singularity or safety threshold. The stacked
spectrum depends on translation/rotation row units; it must not be turned into
an unqualified physical condition-number claim. Separate spectra are included
for that reason. There is no new veto/classification from these values.

Using the orthogonal joint-space row/null projections at q0:

- A−q0 and B−A are exactly zero in both components.
- C−B task component norm: **0.2062364772rad**.
- C−B local null component norm: **0.0501589111rad**.
- Null squared-norm fraction: **5.5848%**, under the stated numerical rank and
  Euclidean joint metric. This is not a causal percentage of physical error.

The weak stacked direction remains in the numerical task subspace. A large
"task" joint component does not imply large Cartesian displacement. This C
response is therefore **not purely null-space motion**. Finite FK displacement
from A is0.366864mm and0.005095221rad. The q0 linearization differs from that
finite change by0.199928mm /0.000351636rad; the local decomposition cannot certify
a0.212rad finite joint displacement as task-invariant.

## Conclusions with evidence status

**Directly observed:** exact expert hold→current IK returns the same hold (B=A).
A small saved learned target perturbation→the same IK returns a different,
limit-near joint vector (C), accepted by the existing checks. Both learned target
error and the current decoder response are present in this matched comparison.
The pose tolerance accepts C even though a held pose would also be within limits.

**Supported interpretation:** at this one configuration, the current full-pose
mapping can turn a small learned initial residual into substantial bounded joint
changes. The objective trade-off and local stacked spectrum are consistent with
that response. The audit does not support blaming an unsolicited branch change
for an exactly zero expert hold, nor describing C as solely null-space drift.

**Hypotheses only:** similar geometric sensitivity may contribute to the physical
HOME excursions; learned initial holding error and decoder realization could
interact. D1 uses the saved local-validation prediction on TRAIN seed100, not a
physical replay of the three seed1 HOME failures, so it does not establish that
causal chain for those episodes or generalize to other poses.

**Unmeasured:** motor tracking, command dwell, contact, physical response and
controller performance. No joint target was sent anywhere. The audit cannot say
which behavior correction would improve success. No correction is proposed here,
and no A-D2 design/authorization is requested by this report.

## Evidence, implementation and reproduction record

Isolated harness: [tools/numerical/ad1.py](/home/tj330/programs/VLA-Project-analytic/tools/numerical/ad1.py).
It uses the frozen ActionAdapter only for decode, with a process-local counter
around the original IK entry restored afterward. The first run writes a marker;
existing evidence cannot be overwritten by another run. Process/network launch
and neural/simulator runtime imports are blocked by a Python audit hook.

Output identity/directory:
`/home/tj330/programs/VLA-Project-analytic/runs/ad1-train-clear-i00-s100-frame0-v1/`.
`native-snapshot.json`, `provenance.json`, `solve-1-started.json`,
`solve-2-started.json`, `solve-ledger.json`, `case-B.json`, `case-C.json`,
`result.json`, `derived-metrics.json`, `objective-analysis.json`, `run.log`,
`files.sha256.json`. The ledger records **B:1,C:1**, no retries. Source/package-file
hashes, seed, targets, bounds, versions and all input fingerprints are preserved.
[Committed numeric evidence](evidence/ad1-numerical-v1.json).

Executed commands, working directory `/home/tj330/programs/VLA-Project-analytic`:

```text
/home/tj330/programs/VLA-Project/.venv-vla/bin/python tools/numerical/ad1.py prepare --project /home/tj330/programs/VLA-Project --evidence /home/tj330/programs/VLA-Project-analytic --out /home/tj330/programs/VLA-Project-analytic/runs/ad1-train-clear-i00-s100-frame0-v1
/home/tj330/programs/VLA-Project/.venv/bin/python tools/numerical/ad1.py run --project /home/tj330/programs/VLA-Project --evidence /home/tj330/programs/VLA-Project-analytic --out /home/tj330/programs/VLA-Project-analytic/runs/ad1-train-clear-i00-s100-frame0-v1
```

Both used `PYTHONDONTWRITEBYTECODE=1`; run additionally used
`PYTHONPATH=/home/tj330/programs/VLA-Project/src`. The recorded invocation completed
successfully. These commands document reproduction; **do not rerun them under the
exhausted two-solve authorization**. A reproduction requires a new authorized
identity/directory and explicit solve budget, not deleting this run's marker.

Four analytic-fixture helper tests pass with **zero real IK calls**. Scoped Ruff
and global formatting pass. The broad CPU suite was deliberately not run: it
contains unrelated real IK tests that could exceed the owner's exact two-solve
budget. No Windows runtime claim is made; the harness uses pathlib/arg lists and
no shell launch. No dependencies were installed or upgraded. No core/public
interface, frozen execution source, model, processor or dataset was changed.

**Stopped after A-D1.** The original225-file campaign, failed gate, repaired23/24
baseline, checkpoint and processor remain intact. Phase I conformance remains
blocked for behavioral corrections and physical diagnostic authorization.
