# Step-20000 local model validation — FAIL / stopped

The recovered portable model loads and performs real CPU inference. Its physical
action output fails the existing gripper limits. No further model calls,
retraining, Webots steps, evaluation or TEST access followed that discovery.
No weights, statistics, processors, dataset, limits or frozen criteria were changed.

| Check | Result | Evidence |
|---|---|---|
| Model integrity | PASS | Receipt size/SHA, ZIP CRC, all 21 manifest hashes, step20000 and 3000 restored updates |
| Model loading | PASS | Actual local SmolVLA weights loaded with the existing `strict=True` loader |
| Processor compatibility | PASS | Local tokenizer/assets load; approved mean/std match; exact state/RGB transform checks |
| Action inference | PASS | Real approved TRAIN episode0/frame0; finite float32 action chunk `(50,7)` |
| Project adapter integration | **FAIL** | 35/50 absolute gripper targets exceed the frozen 45mm bound |
| RTX3050 4GB inference | NOT VERIFIED | No NVIDIA PCI/device nodes, nvidia-smi or CUDA-enabled PyTorch here |

This is a one-observation compatibility check, not an evaluation or success-rate claim.
Kinematic/IK execution has not been tested. The five frozen arms, paired seeds,
TRAIN/validation/TEST splits and literal 10/24 VLA gate are unchanged.

## Verified artifacts and provenance

Source ZIP:
`/home/tj330/Downloads/acct2-durable-v1-drive-v1-portable-020000-attempt-1.zip`

Receipt:
`/home/tj330/Downloads/acct2-durable-v1-drive-v1-portable-020000-attempt-1.receipt.json`

ZIP size: **1,202,740,976 bytes**. SHA-256:
`3de511c1728855a682e87025bd8d30b93d639e7cd3527b3a0c99955382d027d0`.
CRC and all manifest-member SHA-256 checks passed before extraction; all member
hashes passed again after validation. Extraction rejected unsafe paths, symlinks,
extra/unlisted members and existing output. The new export is read-only (files0444,
directories0555) and is not overwritten by the validation procedure:

`/home/tj330/programs/VLA-Project-analytic/runs/local-model-validation-v1/export-step-020000/`

`model.safetensors` SHA-256:
`7531991e27719e6fee402344aadbe43b2fc4ac858a87ae5ec77809c8efe385d8`.
This differs from the independently preserved step17000 tensor file. Manifest
records step20000, restored17000, actual_calls3000, actual post-step hooks and
sampled_parameters_changed=true. Approval matches the local 80-success TRAIN
identity, native statistics and source provenance. The recovery provenance still
discloses `dataloader_cursor_restored=false`; validation does not erase that caveat.

## Installed architecture and observation/action interface

The actual loader is the primary project's
`src/vlarobot/vla/checkpoint_export.py:load_portable_policy`, called by
`src/vlarobot/vla/model_service.py:PortableChunkModel`. Installed LeRobot calls
`safetensors.torch.load_model(..., strict=True)`; a missing/unexpected weight causes
an error. Construction uses local architecture metadata with load_vlm_weights=false,
then restores the exported model file. No random-initialized policy is returned as
a substitute, and there is no dummy policy or scripted fallback.

SmolVLAPolicy uses a SmolVLMForConditionalGeneration backbone reduced to the frozen
16 VLM layers, a flow action expert with10 denoising steps, and50-action chunks.
The saved file contains500 tensors /450,046,176 elements:148,811,856 F32 and
301,234,320 BF16. No precision conversion or quantization was introduced by this task.
The metadata's full backbone template has32 text layers; the policy's configured
16-layer reduction is expected and logged during load.

State: seven arm joint angles in radians plus measured gripper half-aperture in
metres (`float32[8]`). Camera: `observation.images.camera1`, decoded RGB uint8
480x640x3, converted to float32 CHW/255. The model internally pads/resizes to512x512.
Task tokens/mask are each `[1,48]`; preprocessed state `[1,8]`, image `[1,3,480,640]`.
Internal state/action padding is32; the physical action output is unpadded to7.

Action names: `(dx_m,dy_m,dz_m,drx_rad,dry_rad,drz_rad,gripper_target_m)`.
The first six are base-frame pose deltas; the last is an absolute per-finger opening.
The adapter remains `src/vlarobot/vla/action_adapter.py:ActionAdapter`, retaining
SafetyMonitor vetoes before motor commands. No adapter decode/apply or robot step
was performed in this task; its actual gripper safety predicate was checked directly
against the offending numeric output and rejected it.

Both saved processor state files are identical and match the approved dataset's
state/action mean/std. VISUAL=IDENTITY, STATE/ACTION=MEAN_STD. State normalization
was checked exactly against `(state-mean)/(std+1e-8)`; the RGB transformation matches
exactly. Denormalization through the installed postprocessor reconstructs the real
recorded physical chunk with maximum error0.0. This is a round-trip audit: the first
inference did not record its original normalized intermediate, so the report does
not claim a captured raw/intermediate comparison.

## Blocking physical output

Measured CPU inference: **14.086286406s** for one50-action chunk, seed0, using two
PyTorch CPU threads. Input: native TRAIN episode0/frame0, task0, approved successful
source episode. JPEG decoding and RGB/state hashes are preserved in the audit.

Gripper range:0.04416652–0.0454760753m. **35/50** values exceed0.045m. The first
invalid value is chunk index1 (second action):0.0451897532m. Maximum excess is
**0.0004760753m =0.4760753mm**, beyond the adapter's exact float32-endpoint
representation accommodation. The real SafetyMonitor returns
`accepted=false`, reason `gripper opening is outside configured limits`.
No clipping, widened bounds, fallback, resampling or further policy calls were used.
This is not evidence of corrupt weights or failed optimization; it is a measured
physical-interface violation. Resolving it requires a separate investigation under
the frozen contract. Evaluation is blocked at this checkpoint.

## Runtime and reproducibility

Actual ML interpreter: `/home/tj330/programs/VLA-Project/.venv-vla/bin/python`,
Python3.11.18; torch2.7.1+cpu, torchvision0.22.1+cpu, LeRobot0.4.3,
transformers4.57.1, safetensors0.8.0, numpy2.2.6. Full installed versions,
source fingerprints and execution logs are retained in:

`/home/tj330/programs/VLA-Project-analytic/runs/local-model-validation-v1/`

Key artifacts: `integrity.json`, `existing-import.log`,
`existing-import/import-validation.json`, `existing-import/physical-action-chunk.npy`,
`processor-action-audit.json`, `processor-action-audit.log`,
`actual-safety-veto.json`, `installed-versions.txt`, `source-hashes.json`,
`hardware.json`, `validation-summary.json`, and `reproduce-command.json`.
The read-only processor audit script is saved there as `audit-readonly.py`; it
exits2 to preserve the measured integration failure. It makes no policy call.

`reproduce-command.json` contains the exact interpreter/argv/working directory and
explicit PYTHONPATH/offline environment for a future isolated rerun to an unused
output path. The original measured import used the same argv/export/adapter with
its recorded output path; it loaded weights/assets locally and did not explicitly
set HuggingFace offline flags. No reproduction was run after detecting the failure.

Reference inference command (do not run until the stopped validation is deliberately
reopened; this command is not an evaluation): from the analytic worktree, use the
recorded Python environment and arguments:

```text
/home/tj330/programs/VLA-Project/.venv-vla/bin/python tools/recovery/local_import_gate.py --project /home/tj330/programs/VLA-Project --export /home/tj330/programs/VLA-Project-analytic/runs/local-model-validation-v1/export-step-020000 --out /home/tj330/programs/VLA-Project-analytic/runs/local-model-validation-v1/reproduction-new-output
```

Set `PYTHONPATH=/home/tj330/programs/VLA-Project/src`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` in that process's environment as recorded. Never reuse an
existing output directory. CPU is the proven inference route here. A4GB CUDA fit
cannot be inferred from file size or Colab training measurements: actual accelerator
availability and peak allocated/reserved inference memory remain unmeasured locally.
