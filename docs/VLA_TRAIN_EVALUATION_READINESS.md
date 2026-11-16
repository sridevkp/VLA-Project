# Unchanged VLA-only TRAIN evaluation readiness

**Technical GO; launch NO-GO pending explicit owner authorization.** No episodes,
Webots process, model server or inference were started by this preparation.
No behavior/configuration/model/dataset changes are needed to run the current arm.
Expected safety failures remain failures; readiness is not a claim of likely task success.

## Frozen contract and verified inputs

- Exactly **8 TRAIN slots × seeds0,1,2 =24 episodes**, seed-major order. Generated
  only the TRAIN plan and compared every scene/offset/object/seed entry against
  `runs/eval/baseline-train-s0-fresh-v2-registered/plan.json`: exact match.
- Repaired baseline **23/24**. All12 baseline provenance file hashes match current
  source/calibration. Literal threshold `min(0.40,0.60×23/24)=0.40`; pass requires
  **at least10/24**, fail at9/24. Existing boundary tests pass.
- Step20000 export manifest and all required assets verified again. Dataset approval
  identity matches approved80 successes; recovered lineage17000+3000 is retained.
  Weights SHA:7531991e27719e6fee402344aadbe43b2fc4ac858a87ae5ec77809c8efe385d8.
  Manifest SHA:5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1.
- Existing loader remains strict=True, local processor/statistics, state8/action7,
  chunk50; model server defaults to **CPU**. Prior real CPU inference passed;
  no repeat inference was needed. No available RTX3050 is assumed.
- Static registered ActionAdapter instantiated without a simulator; gripper bounds
  remain `[0,0.045]m`, position tolerance0.005m, orientation tolerance0.0872664626rad.
- Webots wrapper exists and is executable at
  `/home/tj330/.local/opt/webots/webots`; rendering remains ON. Display environment
  exists, but live startup/rendering was deliberately not tested. About274GB disk
  is free; no full recording-size forecast or live simulator guarantee is claimed.

Prepared TRAIN plan, exact argv/env, source seals and logs are under
`/home/tj330/programs/VLA-Project-analytic/runs/vla-train-readiness-v1/`.
[Machine-readable readiness evidence](evidence/vla-train-readiness-v1.json).

## Rejection, reset, lifecycle and provenance audit

[ActionAdapter](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:105)
keeps every finite/shape, gripper, workspace, IK residual, joint and velocity check.
Only the pre-existing exact float32 endpoint representation is canonicalized.
The [SafetyMonitor predicate](/home/tj330/programs/VLA-Project/src/vlarobot/execution/safety_monitor.py:185)
is unchanged. No clipping, larger tolerance, resampling, skipping, policy substitution,
scripted fallback, heuristic veto or world-model assistance is enabled.

On rejection, [VLAEpisodePolicy](/home/tj330/programs/VLA-Project/src/vlarobot/execution/vla_episode.py:119)
sets `SAFETY`, breaks the action loop and ends the episode. It makes no subsequent
prediction/retry. HOME remains false and the physical scorer reports success=false.
The [metrics code](/home/tj330/programs/VLA-Project/src/vlarobot/eval/metrics.py:476)
uses that false outcome, and final aggregation keeps all24 denominator entries.
A completed SAFETY failure is reusable by
[`_completed_recording`](/home/tj330/programs/VLA-Project/src/vlarobot/eval/runner.py:336),
not rerun to seek success. Adapter-level vetoes happen before backend.apply, so their
values appear in full `latency.jsonl` chunks and metadata (`adapter_vetoes=1`), while
`rejections` counts backend vetoes and may remain0. Both are meaningful;0 does not
mean the adapter bypassed safety.

CPU checks: **17 existing targeted tests pass**, plus a dedicated fixture injecting
the actual saved maximum at the second action: one fixture prediction, one accepted
simulation-stub step, two accepted backend commands, then false/SAFETY, no further
commands/retry. A completed false/SAFETY record is reused on resume. These are unit
fixtures, not physical episodes or substitute-policy evaluation evidence.

[Campaign lifecycle](/home/tj330/programs/VLA-Project/src/vlarobot/execution/vla_campaign.py:198):
one isolated local model process serves sequential episodes; startup handshake checks
manifest hash, camera key and state/action semantics. Each episode gets a fresh Webots
process/port, then explicit backend scene/seed reset and model reset(seed). Torch RNG
is seeded and policy action queues cleared. Backend reset restores joints/base/object
poses, clears velocities and episode/stall state. Scene geometry comes from the frozen
seeded plan. The reset code does not newly reseed Webots' physics RNG; this is the
unchanged baseline lifecycle, not a newly proved bitwise physical reproducibility claim.
Model preprocessing has no recurrent state; the frozen RTC configuration is null.

World shutdown requests simulation quit and terminates the process tree if needed.
The model receives shutdown, then terminate if alive, with a30s wait. There is no
stronger model kill escalation in this implementation; a hung shutdown requires
inspection rather than launching another overlapping campaign. No model service
was started here. Startup probes retry connection only, not policy actions.

Campaign provenance binds the checkpoint manifest, full plan, horizon, baseline
summary, controller hashes, action interval and HOME tolerance. The readiness ledger
additionally seals SafetyMonitor and runner sources. Export manifest covers weights,
processors, statistics and training provenance. `FakeBackend` appears only in final
**record aggregation**, after all24 physical recordings are required; its factory
refuses to replace a missing physical recording. It does not generate VLA actions
or synthetic successes. Final gate verifies measured TRAIN arms and paired scenes/seeds.

## CPU runtime estimate

Measured one-chunk latency:14.086286406s (~14.09s), not a latency distribution.
Existing episode budget is `max(approved episode_frame_counts)-1 =1140` actions,
50 per chunk, at most23 chunks/episode; action cadence remains32ms lockstep.

| Scenario (all24 episodes) | Model calls | Inference-only time |
|---|---:|---:|
| One chunk each, e.g. early rejection |24 |5.63min |
| Five chunks each |120 |28.17min |
| All reach the1140-action budget |552 |129.59min /2.16h |

Add cold model load,24 world startups/resets, IK, simulator steps/rendering,
recording, RPC and shutdown. These estimates are neither whole-campaign bounds nor
predictions that every scene repeats the previously seen gripper failure. No horizon,
chunk size, seed, action cadence or timeout was changed to fit a runtime target.

## Exact launch command — NOT executed; authorization required

Working directory: `/home/tj330/programs/VLA-Project`.
Process environment: `PYTHONPATH=/home/tj330/programs/VLA-Project/src`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. All other environment is inherited.

```text
/home/tj330/programs/VLA-Project/.venv/bin/python -m vlarobot.cli eval-vla --export /home/tj330/programs/VLA-Project-analytic/runs/local-model-validation-v1/export-step-020000 --out /home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1 --baseline /home/tj330/programs/VLA-Project/runs/eval/baseline-train-s0-fresh-v2-registered --model-python /home/tj330/programs/VLA-Project/.venv-vla/bin/python --webots /home/tj330/.local/opt/webots/webots
```

For exact environment/working-directory execution, **only after explicit authorization**,
the following Python snippet uses the prepared command file. Preparation did not run it:

```python
import json
import os
import subprocess
from pathlib import Path

command = json.loads(
    Path(
        "/home/tj330/programs/VLA-Project-analytic/runs/vla-train-readiness-v1/command.json"
    ).read_text(encoding="utf-8")
)
env = dict(os.environ)
env.update(command["env"])
subprocess.run(command["argv"], cwd=command["cwd"], env=env, check=True)
```

The new output directory is currently absent. Do not precreate it with unrelated
files or reuse any previous experiment directory. No validation/test split flag,
resume flag, fallback option or GPU option is present. Both CLI `--help` paths were
checked without model loading or simulator startup.

## Expected artifacts, completion and stop conditions

Output: `/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/`.

- Run: `provenance.json`, `model-server.log`, `model-metadata.json`, `plan.json`.
- Each episode00–23: `.meta.json` (outcome/failure), `.steps.npz`, `.actions.jsonl`,
  `.contacts.jsonl`, `.predicted.jsonl`, frames; `.latency.jsonl` preserves every
  original50-action output and `.latency.inputs/` the learned-input PNGs.
- Fresh Webots logs: `vla-ep00.log` etc; temporary PID files are cleaned at shutdown.
- Completed campaign: `episodes.csv` with24 rows, `summary.json` with Wilson intervals,
  per-scene rates and failure categories; `proposer-role.json`; `vla-gate.json` with
  actual gate and mean/p50/p95 inference latency. CLI prints the completion result.
  `timings.json` is rewritten by final resume aggregation, so it is **aggregation
  timing**, not physical campaign wall time. Use `vla-gate.json:campaign_wall_s`.

**Current stop:** no launch until owner explicitly authorizes the24 TRAIN episodes.
After authorization, hash/plan/registration mismatch, existing conflicting output,
model startup/identity failure, world startup failure or partial-trace collision
must stop rather than bypass a guard. Ordinary SAFETY failures stop that episode
and remain failed rows while the campaign proceeds to the next fixed pair.
Model-output failures also remain failures; no replacement policy is allowed.
In-episode unexpected exceptions become recorded EXCEPTION outcomes; the current
runner can continue subsequent attempts, but EXCEPTION/partial recordings are not
eligible for a final gate. All24 usable physical outcomes are required.

If interrupted, preserve all files. `--resume` is allowed only for identical campaign
identity and reuses completed failures as well as successes. Partial latency traces
block blind reruns. Do not delete/move failed evidence or repeat SAFETY episodes to
improve the result. Infrastructure recovery needs its own evidence-preserving audit.
No gate conclusion is valid before the complete24-recording aggregation.
