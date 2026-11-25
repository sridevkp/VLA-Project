# Recovered step20000 VLA-only TRAIN campaign — completed, gate FAILED

Executed the exact command/env prepared in0037915 following explicit owner
launch authorization. **All24 physical Webots episodes completed**: eight frozen
TRAIN slots × seeds0,1,2, matched to the repaired23/24 baseline. No early stop,
repeat attempts, clipping, threshold changes, resampling, skipping, scripted
fallback, heuristic veto or world-model assistance. No validation/TEST access,
retraining, model modification or further experiment arm was launched.

| Measurement | Result |
|---|---:|
| Physical pick/place successes | **0/24 (0%)** |
| Wilson95% interval | **0–13.7976%** |
| Frozen pass requirement | **10/24** |
| Literal gate | **FAILED**, threshold min(0.40,0.60×23/24)=0.40 |
| SAFETY failures | **21**, all upper gripper-bound violations |
| PLACEMENT failures | **3**, HOME returned but frozen physical criteria failed |
| Adapter veto count | **21** |
| Backend rejection count | **0**, because these commands were rejected before backend submission |
| Model calls / original action chunks | **24 /24**, each50×7 |
| Original predicted actions preserved | **1,200** |
| Accepted/executed actions | **85** |
| CPU inference latency: mean / median / p95 | **15.8514 /15.4391 /18.2790s** per chunk |
| Campaign wall time | **970.1816s (16min10.18s)** |
| Launcher wall time |974.7158s (16min14.72s), including entry/exit overhead |
| Command return code / infrastructure failures | **0 /0** |

The small floating-point Wilson lower endpoint in the untouched runner JSON is
1.3877787807814457e-17, displayed as0 above. Wilson intervals are descriptive for
24 paired episodes; repeated seeds across eight scene clusters are not24 independent
scenes. Every scene is0/3; each seed is0/8. Family totals are clear0/9,
cluttered0/9 and reach-limited0/6. No subgroup was excluded.

## Physical outcomes and veto evidence

All21 SAFETY failures have reason `gripper opening is outside configured limits`.
The offending original targets range0.04500041529536247–0.04524412006139755m.
All exceed the existing exact float32 endpoint accommodation. No rejected target
was submitted to the backend. `summary.json:rejections_total=0` is therefore the
**backend** counter, not an absence of safety vetoes; per-episode metadata records
`adapter_vetoes=1` for each of these21 episodes. Each is a genuine failed row.

PLACEMENT failures:

| Pair | Accepted actions | Final target distance |
|---|---:|---:|
| train-clear-i00-s1 |7 |0.19426153m |
| train-clear-i03-s1 |4 |0.17170853m |
| train-clear-i06-s1 |7 |0.21001296m |

These three returned HOME quickly; physical success remained false. Numeric
recordings show maximum can rise across all24 episodes only
**6.5453e-7m**. There is no recorded physical lift, transport or placement success.
This is an additional diagnostic read of recorded TRAIN ground truth, not a new
success criterion or an input to the model.

The unchanged summary reports13 displacement/tilt **collision-proxy** hits,
zero recorded actual contact events and no FK-clearance measurement. Proxy hits
must not be relabeled actual collisions; missing collision/FK evidence is not a
claim of collision safety. Original output includes these limitations.

## Integrity, lifecycle and evidence

Independently checked the24 finished metadata files, numeric records, learned
input images, one full prediction chunk per episode and24 distinct world-session
nonces. Numeric observation counts equal executed_actions+1. All recorded pairs
match the prepared full plan and TRAIN split. No EXCEPTION/partial outcome, model
restart, policy retry or repeated completed failure was needed. Model server,
Webots and campaign processes exited; no campaign PID files remain.

Rechecked all12 repaired-baseline source hashes, readiness source seals and all21
export member hashes after completion. They are unchanged. The model server
reported the expected manifest identity. Verified weights SHA remains
7531991e27719e6fee402344aadbe43b2fc4ac858a87ae5ec77809c8efe385d8;
manifest SHA remains
5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1.
All earlier recovery, local-validation and root-cause evidence is preserved.

Primary unchanged runner artifacts:

`/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/`

- [summary.json](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/summary.json)
- [vla-gate.json](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/vla-gate.json)
- [episodes.csv](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/episodes.csv)
- [proposer-role.json](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/proposer-role.json)
- Full per-episode original `.latency.jsonl` predictions, input PNGs, recordings,
  frames, outcomes, Webots logs, model-server log and checkpoint provenance.

Separate verification preserves rather than rewrites the runner outputs:
`/home/tj330/programs/VLA-Project-analytic/runs/vla-train-result-v1/`.
It holds verification.json, the exact authorized command and launch log/return code,
source simulator-status sidecar copies and SHA-256 inventories. The campaign
inventory covers225 files /9,458,347bytes and hashes to
016a5160e37639f4e95c2aa4f72d08d2df677df8a5a8876b316cc5029e6dde07.
[Committed per-file inventory](evidence/vla-train-step20000-files.json) and
[per-episode numeric verification](evidence/vla-train-step20000-results.json).
`timings.json` measures the runner's final aggregation; the actual campaign time
reported above comes from `vla-gate.json:campaign_wall_s`.

## Frozen gate outcome and stop

The literal gate fails:0/24 is below10/24. Existing `proposer-role.json` selects
**scripted-primary / VLA MEASURED partial result**, as required by ADR-0029. This
role decision does not retroactively add scripted actions to VLA-only. The repaired
23/24 baseline is retained separately; failed fresh-v1 evidence is not merged.

**Stopped after completing and verifying the authorized campaign.** No policy
correction, new training or additional experiment arm was launched. Any subsequent
work must preserve this complete failed VLA-only result and frozen acceptance rules.
