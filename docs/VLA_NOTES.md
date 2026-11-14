# VLA Notes — SmolVLA / LeRobot integration (ADR-0029, M8)

**Status:** draft (Week 1/2 prep). **MEASURED** only where evidence is cited; otherwise **UNVERIFIED** until verified against the installed `lerobot`/`transformers`/`torch` stack and `policy.config`.

## Current repo state (MEASURED)
- Prototype smoke script: `model/main.py` loads `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`, prints `policy.config.input_features` / `output_features`, builds dummy inputs, runs `preprocess -> policy.select_action -> postprocess`.
- Prototype inputs shown in `model/main.py`: 3 cameras `observation.images.camera{1,2,3}` as `(1,3,480,640)` float tensors, `observation.state` as `(1,6)` zeros, `task` string. **This reflects SO-100/SO-101-style assumptions in the prototype, not TIAGo.**
- Environment/setup: `model/readme.md` suggests a separate `lerobot` conda env (`python=3.12`, `ffmpeg`, editable `lerobot[smolvla]`). Main project venv (`uv sync --extra sim --extra dev`) does **not** include `lerobot` (MEASURED: `import lerobot` unavailable in project env; `pyproject.toml` does not pin it).
- `collect-demos` CLI: **not implemented yet** in `src/vlarobot/cli.py` (target: ADR-0029/ARCHITECTURE M8/M2 — scripted expert over randomized scenes → LeRobot-compatible-convertible dataset).

## TIAGo embodiment vs SmolVLA base (UNVERIFIED — must be reconciled)
- **Do not assume SO-100/SO-101 observation/action dimensions are compatible with TIAGo** (owner directive; ARCHITECTURE M8 notes embodiment/action-space mismatch).
- TIAGo control in this project: 7 arm joints + gripper (+ torso/base handled separately by policy/scripted stack); scripted expert acts through existing IK/safety layer. ADR-0029 demo spec: **RGB, state, instruction, 7-D EE-delta + gripper actions** through the existing IK/safety layer (~80 episodes, perturbed can poses in reachable envelope).
- Required to record in this file once verified on the actual installed stack/checkpoint:
  - `policy.config.input_features` / `output_features` (exact keys, shapes, dtypes).
  - Expected camera keys/names, resolution, and normalization.
  - State vector definition/dim and normalization.
  - Action chunk shape, action dim/semantics, normalization, and how to map **7-D EE-delta + gripper** to the model space (or document/implement the adapter).
  - Language/instruction field name and tokenization path (`preprocess` behavior).
  - Control/chunk rate and execution integration with `ActionAdapter` (EE→IK, limits, chunk execution, invalid-output rejection).

## Verification plan (to make MEASURED)
1. Install compatible `torch`/`lerobot`/`transformers` in an isolated env (Linux/WSL2/Colab recommended for training per ARCHITECTURE/DECISIONS). Record exact versions.
2. Run `from_pretrained("lerobot/smolvla_base")` and dump `policy.config` (input/output features, camera keys, state/action dims, normalization stats/paths) into this file with command/output evidence.
3. Validate a forward pass with TIAGo-shaped (adapted) observations or document the required adapter; confirm action decoding/un-normalization units.
4. Verify `collect-demos` output schema against LeRobot dataset tooling (loader/visualizer) and record dataset structure + demo-count guidance from LeRobot docs (UNVERIFIED until checked against current docs/version).

## Demo collection (planned, ADR-0029/ARCHITECTURE)
- Source: scripted expert (`ScriptedCanPolicy` / `execution.scripted_baseline`) over seeded/randomized scenes (`eval.scenarios.generate_scenes`, reachable/campaign envelope as specified) with `EpisodeRecorder` evidence.
- Recorded signals (target): RGB frames, robot/state, instruction text, 7-D EE-delta + gripper actions, safety verdicts/outcomes, provenance — stored in a LeRobot-convertible layout (existing npz/jsonl/jpg + metadata; converter to ship with `collect-demos`).
- **Rendering/observation optimizations (`observe_images=false`, no-rendering, etc.) are forbidden for equivalence-critical/demo paths until required ≥10-episode outcome-equivalence proofs are recorded** (ADR-0029). Process hygiene and per-episode isolation preserved.

## Training (Colab, owner-directed constraints)
- GPU: Google Colab (T4-class, disconnect-prone). All training **resumable from Google Drive checkpoints**; exact commands and runtime estimate to be provided.
- **Inference runs locally** from exported checkpoint; **no Colab tunnel dependency**.
- Week-2 gate drives role (VLA primary proposer only if `>= min(40%, 60% × scripted baseline)` on 8 train scenes × 3 seeds, same seeds; never tune on val/test). If Google auth/interactive Colab is unavoidable, provide a minimal one-time HUMAN STEP and continue other work.

## Open items / blockers
- LeRobot package + compatible torch/transformers not installed in project env (UNVERIFIED runtime). Windows-native training UNVERIFIED (prefer Linux/WSL2/Colab).
- Exact `smolvla_base` config not yet captured from a live `policy.config` in this environment.
- LeRobot dataset API/version and demo-count guidance: **UNVERIFIED** until checked against current LeRobot docs for the chosen version.
- `collect-demos` and `ActionAdapter` implementation pending.
