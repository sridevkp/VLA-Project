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

## Endpoint decoder and closed-loop decision

The approved native TRAIN dataset has 63,081 action labels across 80 demos.
The gripper component is exactly one of two stored endpoints: 40,106 open
and 22,975 closed. Every demo contains one close and one release transition.
The decoder selects the endpoint from the model's finite scalar score using
the actuator midpoint. It rejects malformed, nonfinite and exactly ambiguous
scores. It takes no task-phase or oracle input. The pose commands, SafetyMonitor,
HOME lifecycle and bounded IK fallback are unchanged.

On 1,050 aligned commands in 21 teacher-forced windows, 485 raw targets were
outside the motor interval and zero decoded targets were outside. Endpoint
intent disagreed with the expert at 14/100 DESCEND, 7/100 CLOSE and 1/50
first-open RELEASE actions. All 100 LIFT and all 100 TRANSIT commands agreed
on the gripper endpoint. The seven CLOSE reopenings are learned decisions
and remain visible after decoding.

| Seed-0 TRAIN scene | Model queries | Accepted ticks | Closest TCP–CAN | Maximum CAN rise | Terminal |
|---|---:|---:|---:|---:|---|
| clear-i00 | 3 | 108 | 791 mm | 4.7 μm | Workspace z violation |
| cluttered-i01 | 2 | 92 | 886 mm | 4.0 μm | IK residual exceeds tolerance |
| reach-limited-i02 | 2 | 92 | 880 mm | 4.0 μm | IK residual exceeds tolerance |

The three live diagnostics each commanded only the 45 mm open endpoint and
failed independent task verification. Observed final half-apertures
were 45.013, 45.003 and 45.006 mm, respectively. No closed-loop grasp, lift, transport, release
or HOME completion was observed. Raw proposed targets ranged from 44.10 to
46.70 mm. Of 350 proposed targets, 252 were outside the motor interval. These
are diagnostic results under the corrected execution path. Full TRAIN admission
remains unrun. The TCP–CAN distance uses the confirmed TCP pose transformed by
the full base pose and CAN ground truth from the observation recorder. Ground
truth is used only for this post-run calculation.
[Recorded diagnostic](../runs/worldmind/v2-train-endpoint-1/)

Targeted TRAIN fine-tuning is warranted. The intervention keeps the validated
action labels and pose objective, and allocates explicit sample coverage to
grasp and release transitions. It is a hypothesis to test for improved timing
and pose accuracy. The closed-loop evidence also shows that learning to
approach the target is necessary. Progress on the grip score alone is
insufficient. The original optimizer telemetry confirms updates and throughput.
It does not establish convergence.
The original processor audit confirms exact RGB and state-normalization parity,
matching dataset statistics, and zero error on a postprocessor round-trip
of recorded physical actions. It does not cover the original normalized
intermediate predictions or establish sensitivity to the language instruction.

For Eq. 3.2, the minimum recovery is a learned decision head that jointly
consumes the VLA observation/language representation, candidate actions and the
corresponding predicted state sequences. Once learned admission is established,
train and test this head on TRAIN-only evidence. Hold images, state, language
and candidate payloads fixed, and test whether changing only the predicted
context can change the learned decision. Report this conditioned decision
separately from external objective ranking.

## Integration and PR access

PR #2 remains bound to evaluated V1. Both official CI jobs failed at
`uv run ty check src`. The separate `work/v1-integration-fixes` branch
contains behavior-preserving typing patches. The local type-check reports
54 diagnostics, down from 98. The two latest patches passed 33 production
regressions and five arm-sweep regressions. It is not ready to advance PR #2. The notebook-helper SHA
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

## TRAIN transition warm-start on Colab

Use the original 80-demonstration TRAIN archive and the same step-20,000 export
as the closed-loop endpoint experiment. The source export manifest SHA-256 is
`5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1`.
The policy receives the original camera, state, language and action representations.
The sampler uses TRAIN label transitions to select windows. No phase label enters
training inputs or the runtime decoder.

1. Follow [Colab setup](COLAB_EXECUTION.md) in a T4 runtime through the first
   three code cells, including TRAIN dataset verification. Skip the V1 training cells.
2. Upload the original `acct2-durable-v1-drive-v1-portable-020000-attempt-1.zip`
   and its receipt from Downloads into `MyDrive/vlarobot_smolvla/`.
3. Add a Colab Python cell and run the following. Set `REVISION` to the full recovery
   commit containing this section. If the repo requires access, upload the recovery
   source archive for that commit and extract it at `/content/VLA-Project-v2`.
   To make the archive, run `git archive --format=zip -o vla-recovery.zip REVISION`
   from the recovery worktree, substituting the full commit for `REVISION`.
   Upload it to the same Drive folder and run
   `!unzip -q /content/drive/MyDrive/vlarobot_smolvla/vla-recovery.zip -d /content/VLA-Project-v2`
   in Colab before the Python cell below.

```python
import hashlib
import os
import subprocess
import zipfile
from pathlib import Path

REVISION = "replace-with-full-recovery-commit"
drive = Path("/content/drive/MyDrive/vlarobot_smolvla")
repo = Path("/content/VLA-Project-v2")
export = Path("/content/vla-source-step-020000")
python = "/content/vla-env/bin/python"
if not repo.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "work/vla-recovery",
            "https://github.com/sridevkp/VLA-Project.git",
            str(repo),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", REVISION], check=True)
if (repo / ".git").exists():
    assert (
        subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        == REVISION
    )
if not export.exists():
    with zipfile.ZipFile(drive / "acct2-durable-v1-drive-v1-portable-020000-attempt-1.zip") as z:
        z.extractall(export)
assert hashlib.sha256((export / "export-manifest.json").read_bytes()).hexdigest() == (
    "5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1"
)
subprocess.run([python, "-c", "import torch; assert torch.cuda.is_available()"], check=True)
subprocess.run(
    [
        python,
        "-u",
        str(repo / "tools/finetune_vla_transitions.py"),
        str(export),
        "/content/vlarobot-data",
        str(drive / "v2-transition-warmstart-01"),
        "--steps",
        "1000",
        "--batch",
        "8",
        "--save-every",
        "250",
        "--device",
        "cuda",
    ],
    env={**os.environ, "PYTHONPATH": str(repo / "src")},
    check=True,
)
```

The earlier T4 run recorded about 1.18 seconds per update over its final 3,000
steps. That gives about 20 minutes for 1,000 updates at the same speed.
Allow 30–60 minutes with setup and checkpoint writes. This is an estimate.
After an interruption, use `--resume` if the run has a complete checkpoint.
If it stopped before the first save, keep the attempted run and use a fresh
output directory. Download the `step-021000` directory and retain the
optimizer log. Before attempting the full TRAIN gate, rerun the same three
TRAIN diagnostics with the resulting export and check for an observed grasp
and lift. The diagnostic cannot satisfy the 10/24 admission gate.
