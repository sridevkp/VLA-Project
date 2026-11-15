# Durable recovery readiness — 2026-11-16

**Recommendation: NO-GO for GPU training and evaluation. GO for CPU preparation.**
No training, learned-policy inference, physical evaluation, TEST access, key
creation, SSH listener startup or public ingress change occurred in this task.

| Area | Status | Evidence / remaining stop gate |
|---|---|---|
| Linux original017000 | PASS | All12 COMPLETE entries rehashed; supplied model/optimizer hashes match |
| Frozen configuration/source | PASS | steps20000, batch8, seed0, save1000; production files equal approved source.zip |
| Approved dataset | PASS | All native file hashes revalidated; 80 successful TRAIN episodes; checkpoint statistics/provenance match |
| Local full-size receiver | PASS | 1,610,503,770 bytes; SHA, file/directory fsync, receipt and fresh-process read-back verified |
| Authenticated Colab→Linux canary | BLOCKED | No authorized Colab endpoint/session supplied; network overhead unmeasured |
| Private route candidate | PASS (discovery only) | Existing Tailscale is Running/online at100.86.97.108; no SSH listener |
| Restricted SSH receiver activation | BLOCKED | Config/dispatcher prepared; host/client keys require owner approval; no host key exists |
| Drive quota/ownership/persistence | BLOCKED | No connected Drive tool or local ADC credentials; read-only Colab diagnostic provided |
| CUDA restoration/T4 | BLOCKED | T4 temporarily exhausted; actual restore gate must pass before any update |
| Fresh17→20 recovery | PASS (CPU implementation) | Unchanged update/safety/storage gates; actual GPU execution not tested here |
| Resume after durable18/19 | BLOCKED (execution) | Current gate deliberately requires17 and one3000-update invocation; no reset/replay is authorized |
| Local import/TRAIN commands | PASS (prepared) | Listed below; not executed |
| CPU tests | PASS |338 isolated tests;24 primary VLA infrastructure tests; TEST-generating legacy unit deliberately excluded |
| Global lint | BLOCKED (pre-existing) | One E402 in untouched old Colab notebook; scoped source/tests/tools lint passes |

## Measurements and storage

Actual local subprocess handoff, not SSH/network:

- Payload: 1,610,503,770 bytes; SHA-256
  `e34c9596d0fdaef579b734829d45a60c257370d9de4e17348055de6734366109`.
- Sender-to-receiver receipt:23.035 s (66.68 MiB/s locally).
- Additional fresh-process read-back:3.286 s; identical SHA.
- Receiver free space after write:275,100,401,664 bytes.
- file_fsync=true, directory_fsync=true, receiver_readback_verified=true.

Do NOT use these local timings to estimate Colab network bandwidth. Authenticated
full-size network timing remains unavailable. The owner must run Cell3 in
DURABLE_RECOVERY_PLAN.md before allocating scarce GPU time; its receipt supplies
measured handoff overhead and current receiver capacity. Keep partial attempts.

Evidence: `docs/evidence/fullsize-local-receiver.json`,
`readiness-frozen-audit.json`, `private-route-readiness.json`,
`receiver-config-check.json` and final `readiness-verification.json`.

## Owner actions: private endpoint and credentials

These actions are not executed. The user explicitly prohibited credential
creation without owner approval; the key-creation commands below are owner-only.
The commands provision a dedicated restricted client key and server host key;
neither has been created by the agent.

The prepared directory is
`/home/tj330/programs/VLA-Project-analytic/runs/receiver-private-v1/`.
It contains `sshd_config` and `receiver-server.json`, neither is a credential.
The config listens ONLY on Tailscale100.86.97.108:2222, disables passwords,
forwarding and TTYs, and forces `receiver_dispatch.py`. That dispatcher permits
only the exact fixed golden-input download and fixed-root artifact receiver.
Shells, SFTP/SCP commands, alternative source files and receiver roots are denied.

1. Approve credential provisioning and use a CPU Colab session as a temporary
   member of the existing tailnet. Follow the official Tailscale Linux setup,
   approve the device interactively and permit only that device→receiver TCP2222
   in the existing tailnet policy. No public SSH, router port-forward, Funnel,
   exit-node or subnet-route change is needed. This agent has not installed or
   enrolled a Colab node. If Colab lacks a working private network interface,
   stop and supply an authorized route; no unsupported proxy fallback is claimed.
2. On the Linux console, only after approval, run Python with these commands:

```python
import subprocess
from pathlib import Path

SERVER = Path("/home/tj330/programs/VLA-Project-analytic/runs/receiver-private-v1")
for name in ("host_ed25519", "client_ed25519"):
    assert not (SERVER / name).exists()
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(SERVER / name)], check=True)
public = (SERVER / "client_ed25519.pub").read_text().strip()
with (SERVER / "authorized_keys").open("x") as stream:
    stream.write("restrict " + public + "\n")
(SERVER / "authorized_keys").chmod(0o600)
host = (SERVER / "host_ed25519.pub").read_text().split()
with (SERVER / "verified_known_hosts").open("x") as stream:
    stream.write("[100.86.97.108]:2222 " + host[0] + " " + host[1] + "\n")
subprocess.run(["ssh-keygen", "-lf", str(SERVER / "host_ed25519.pub")], check=True)
subprocess.run(["/usr/bin/sshd", "-t", "-f", str(SERVER / "sshd_config")], check=True)
```

The current read-only `sshd -T` check stops with `no hostkeys available`; this is
expected before approval, not proof of a working server. If syntax, privilege,
account or SELinux checks fail after provisioning, preserve the error and stop.
Do not disable system security or bind publicly to work around it.

3. Copy `verified_known_hosts` through a trusted channel and provision the approved
   client private key into Colab as `/content/authorized_recovery_key`, outside
   all model/archive directories. Place the host file at
   `/content/verified_known_hosts`. On Colab run `ssh-keygen -lf` on that host file
   and compare its SHA-256 fingerprint to the Linux-console result. Do not trust
   an unverified ssh-keyscan result. Never paste private keys into chat/notebooks.
4. In a dedicated Linux terminal, after config validation and approval, start:

```python
import subprocess

subprocess.run(
    [
        "/usr/bin/sshd",
        "-D",
        "-e",
        "-f",
        "/home/tj330/programs/VLA-Project-analytic/runs/receiver-private-v1/sshd_config",
    ],
    check=True,
)
```

Verify its listener with `ss -ltn`: only100.86.97.108:2222 is acceptable for this
service. Keep it running during transfers. Arbitrary `ssh ... echo` probes will
be denied by design; use the permitted golden-input/canary operations instead.
Set v4 transfer config destination=`tj330@100.86.97.108`, port=2222; retain the
prepared receiver/helper/golden paths. Run the full-size canary, not training.

The Tailscale setup and daemon activation are still owner actions; an online
Linux node alone does not prove Colab reachability. Source references:
[Tailscale Linux](https://tailscale.com/docs/install/linux),
[OpenSSH configuration](https://man.openbsd.org/sshd_config).

## Drive investigation: CPU-only and read-only

Run Cell2 in DURABLE_RECOVERY_PLAN.md in authenticated CPU Colab and transfer the
small diagnostic JSON to Linux. It reports principal/project ownership IDs,
quota/usage/trash bytes and tensor file metadata, including trashed candidates.
Compare against the owner of the shortcut target and any shared/pooled storage
policy. Check owner-side file activity/revisions/trash/retention. API access errors
and missing metadata remain explicit unknowns. No delete/restore/API mutation is
part of this audit. No local ADC file or connected Drive session was found here.
Missing model files and mounted-filesystem reads alone do not establish a cause.

## Retry/interruption review and stop gates

| Event | Permitted next action; no extra optimizer updates |
|---|---|
| Transfer fails before receipt | Sender tries three unique artifact IDs; exhausted retries abort checkpoint callback; preserve local/partial files |
| Receiver has file but ACK lost | Verify Linux receipt and fresh hash; a new artifact ID may retry transfer only; do not rerun training to resend |
|18/19 receipt succeeded, runtime lost | Preserve highest verified full checkpoint plus bundled provenance/logs; do not reset to17 automatically |
|20 portable receipt succeeded, full-state transfer failed | Preserve portable model; retry full-state transfer only while runtime exists |
| Runtime has20 files but no receipt | No durability claim; independently transfer/read-back before accepting success |

Important current implementation limit: `restore_gate.py` fixes checkpoint017000,
and the terminal gate requires calls3000 restored17000 in one invocation.
`latest()` can discover a valid18/19 but the subsequent restore gate rejects it.
This is fail-closed, not a working partial-resume path. The existing v4 launch
must not be used blindly after interruption. A separately reviewed lineage audit
and remaining-update plan is required before enabling an18/19 continuation.
The private receiver key also permits downloading only the fixed golden017000
archive. A later18/19 restart needs an explicitly verified new download allowlist
entry; broad arbitrary-file access must not bypass that restriction.

Logical committed steps17→20 total3000, but physical updates can exceed3000 if an
unrecorded tail is replayed. The every100-step logger cannot prove there were
zero unrecorded calls after the last durable checkpoint. Do not silently equate
committed steps with actual calls, rewrite old logs or change frozen definitions.
If an actual interruption occurs, report this distinction before another launch.

## Single future GPU procedure

Use only DURABLE_RECOVERY_PLAN.md (v4), not old Drive-output training cells:

1. CPU: approved setup/data validation, metadata diagnostic, private SSH setup,
   full-size endpoint canary and capacity check. STOP on any failure.
2. Stage the verified golden017000 input into fresh local `/content` directories.
   New checkpoint writes must remain under `/content/durable-recovery-v1`, outside
   `/content/drive`; pre-copy canonical/disjoint checks and sink guard enforce this.
3. When T4 AND new training authorization exist: actual CUDA restore-only gate.
   Require step17000, all model/AdamW/scheduler/RNG values, dataset identity and
   zero optimizer updates. STOP before training if any item is unavailable.
4. Only after authorization, run the single v4 training command with the verified
   durability config. Require first17001/calls1 and final20000/calls3000; each frozen
   save hands off synchronously. Transfer failure is a STOP, not a fallback to Drive.
5. Require portable/full checkpoint receipts, Linux fresh hashes and strict local
   import preflight. No evaluation is currently authorized.

The approved seed0, batch8, total20000, cadence1000, model/normalization/data and
December12 deadline remain fixed. This plan does not promise a Colab session
duration. Network overhead must be measured on the actual route.

## Downstream commands prepared — DO NOT EXECUTE now

From `/home/tj330/programs/VLA-Project`, after independently verified final export
is safely extracted to NEW `runs/trained-export-durable-v1/` and fresh authorization:

```text
.venv-vla/bin/python /home/tj330/programs/VLA-Project-analytic/tools/recovery/local_import_gate.py --project /home/tj330/programs/VLA-Project --export runs/trained-export-durable-v1 --out runs/import-trained-durable-v1
.venv/bin/python -m vlarobot.cli eval-vla --export runs/trained-export-durable-v1 --out runs/eval/vla-train-s0-durable-v1 --model-python .venv-vla/bin/python --baseline runs/eval/baseline-train-s0-fresh-v2-registered --webots /home/tj330/.local/opt/webots/webots
```

Before the first command, require portable `checkpoint_step=20000`, every required
model/processor/normalization file, complete hashes, recovery lineage and matching
approved80 dataset identity. Compare all final weights to pinned initialization.
Installed strict loading and real TRAIN RGB/state8→finite50×7 actions must pass;
record cold/warm latency. No17-only weights or synthetic fixtures qualify.
The prepared `local_import_gate.py` enforces final20000, exact80 episodes,
state8/action7/camera1/50-action/normalization and saved frozen settings before
delegating to the existing installed-policy inference validator. It was tested
only with CPU fixtures; no policy was loaded by this task.
The VLA-only campaign retains motor safety, with zero scripted actions, heuristic
selection or world-model assistance. Preserve all24 paired TRAIN scene/seed logs.
Gate remains min(0.40,0.60×23/24)=0.40, hence10/24 successes. No TEST access.

## Analytic/RiskGate blockers independent of model weights

- Collision coverage remains uncertified: live probing reached17/30 collision
  links, with incomplete base/passive geometry, unsupported planes, missing
  validated self-exclusions, swept-volume bounds and carried-object coverage.
  Unknown geometry stays vetoed. Nominal32×50 kernel latency0.468 s excludes
  learned inference and full certified collision processing.
- Four physical candidate diagnostics expose closure/contact and release errors;
  they do not establish grasp/placement prediction or calibrated probabilities.
  No new physical experiment ran under the current no-evaluation instruction.
- Authoritative PhaseI J/Table5.1 definitions remain unavailable in repository
  docs; parameterized interfaces reject missing definitions. No weights guessed.
- Five-arm distinct-selection and safety/fallback fixtures pass CPU tests; real
  learned-policy comparison still needs usable final weights and the TRAIN gate.
- Canonical physical VALIDATION15/18 is historical scripted evidence, not a VLA
  rate. The five frozen arms, splits and paired seeds are unchanged.

Safety completion therefore remains blocked by geometry/report authority and
physical validation requirements in addition to the trained-policy artifact.

## Verification outcome

- 338 CPU-only isolated-worktree tests passed in134.41 s; one legacy TEST scene
  generation unit deliberately deselected. No regressions observed.
- 52 targeted storage/receiver/import-gate regressions passed.
- 24 CPU-only primary-worktree VLA infrastructure tests passed in2.62 s.
- Scoped Ruff lint passes. Global formatting passes169 files after formatting
  existing documentation code blocks. Global lint retains one pre-existing E402
  in the untouched older notebook; no approved notebook or source bundle changed.
- No actual private SSH canary, authenticated Drive API call or CUDA restoration
  has been performed. These remain genuine external stop gates, not implied PASS.
