# Independent training readiness — 2026-11-15

The accepted v3 source/data archives and bundle-status.json remain unchanged. Verification:
`runs/week2-progress-v1/final-bundle-verification.json`. Both archives match approved
SHA-256 values and pass full ZIP CRC checks. Manifest SHA-256:
`b335fa8a4825688fcfeb539dd7c9539c73dccc35bcf98892c0a94482d79939ff`.
Notebook checks each copied archive against the manifest before extracting that archive,
requires approved status, then validates native dataset approval before training. This is
integrity against the accepted manifest, not independent authentication of a replaced manifest.

Checkpoint requirements include model/config, saved processors/normalization, optimizer
tensors/parameter groups, scheduler, Python/NumPy/Torch RNG and global step. Completion
markers SHA-hash every required file; partial/corrupt newer saves are skipped. Installed
LeRobot save/restore was repeated with copied regular-file checkpoints in a second directory.
Model, optimizer, scheduler, all CPU RNGs and step restored; the next optimizer update was
bitwise equal. Evidence: `runs/lerobot-drive-restore-v1/restore-report.json`. This was a toy
serialization test with zero SmolVLA optimizer steps. Actual Drive disconnect behaviour and
CUDA RNG restoration remain untested without the authenticated T4.

Frozen training configuration remains batch 8, 20,000 steps, seed 0, checkpoint every 1,000.
The installed compatibility model has 450,046,176 total / 99,880,992 trainable parameters.
Measured parameter+gradient+two Adam moment storage lower bound is 1,816,643,328 bytes
(about 1.69 GiB) in the loaded mixed storage dtypes. This excludes activations, temporary
optimizer workspaces, CUDA context/allocator and data transfers; it is NOT a peak estimate.
Vision freezing/expert-only flags remain enabled, one camera, 50-action chunks, use_amp=false.
T4 peak memory and dtype/kernel compatibility cannot be confirmed on this CUDA-free host.
Record actual GPU peak allocated/reserved memory during the initial training batch; stop
and report OOM rather than silently changing batch size, precision, model or data.

Local import preflight is prepared at `runs/week2-progress-v1/validate-import.py`. It
requires actual trained evidence, verifies manifest and accepted dataset identities, strictly
loads model/normalization and runs one real TRAIN observation through installed inference.
An untrained compatibility export was rejected before model loading or robot commands.
When a trained portable export is supplied, from the project root run:

```text
.venv-vla/bin/python runs/week2-progress-v1/validate-import.py --export runs/trained-export --out runs/import-trained-v1
uv run python -m vlarobot.cli eval-vla --export runs/trained-export --out runs/eval/vla-train-s0-v1 --model-python .venv-vla/bin/python --baseline runs/eval/baseline-train-s0-fresh-v2-registered
```

The evaluation command runs frozen TRAIN 8×3 pairs only, local inference, safety vetoes,
no scripted fallback. HOME alone cannot satisfy physical placement success. The paired
literal gate remains min(40%, 60%×23/24)=40%; 10/24 successes passes. No TEST evaluation
has occurred. No trained checkpoint has been received and no learned task gate is measured.

Analytic follow-up is isolated in sibling worktree `VLA-Project-analytic`, branch
`work/analytic-risk-followup`; its numeric input/identity guards do not enter the training
artifacts. November 27 Week 2 gate and December 12 deadline remain unchanged.
