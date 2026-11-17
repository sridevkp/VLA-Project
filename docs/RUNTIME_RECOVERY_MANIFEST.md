# Runtime recovery source manifest

- Branch: `staging/runtime-recovery`, base commit `aea2a87a4efcb9b2fae5b0a60e0769c0dcb86431` (published PR head `aea2a87`).
- Procedure: files were copied byte-for-byte from the dirty primary worktree (`/home/tj330/programs/VLA-Project`, branch `m1-checkpoint-ik-transit`) into this isolated staging worktree. The primary worktree was not modified.
- Provenance keys: `colab-v3` = `runs/colab-prep-v3/bundle-status.json` (`source_files_sha256`, the source snapshot packaged for the approved Colab training); `baseline-prov` = `runs/eval/baseline-train-s0-fresh-v2-registered/provenance.json` (`files_sha256`, the 12 files hash-locked by the frozen 23/24 baseline and re-verified fail-closed by `run_vla_campaign`).

## Recovered files

| Path | Origin | Class | SHA-256 | Copy integrity | Provenance |
|---|---|---|---|---|---|
| `.gitignore` | primary-modified-tracked | ENV (adds .venv-vla/) | `e55b18f32c376ab5…` | byte-identical | - |
| `configs/calibration/urdf-base-train-v1.yaml` | primary-untracked | CONFIG (frozen baseline calibration) | `dfc542e11f17e97d…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `configs/robot.yaml` | primary-modified-tracked | CONFIG (frozen-baseline-locked) | `1a9f0c2b57d771e4…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `configs/training.yaml` | primary-untracked | CONFIG (approved training) | `f9be85879dfbc552…` | byte-identical | colab-v3:MATCH |
| `configs/vla.yaml` | primary-untracked | CONFIG (VLA runtime) | `0ae9bacb4ec81e0f…` | byte-identical | colab-v3:MATCH |
| `configs/world_model.yaml` | primary-untracked | CONFIG (no code consumer yet; reserved) | `bcd6c10f377813c7…` | byte-identical | colab-v3:MATCH |
| `docs/BASELINE_REPAIR_CHECKPOINT.md` | primary-untracked | DOC (baseline repair history) | `02c078eb0308e4d2…` | byte-identical | colab-v3:MATCH |
| `docs/COLAB_EXECUTION.md` | primary-untracked | DOC (approved Colab procedure) | `025f50d15626c702…` | byte-identical | - |
| `docs/INDEPENDENT_TRAINING_READINESS.md` | primary-untracked | DOC (training readiness) | `6d6f2db39c131bc2…` | byte-identical | - |
| `docs/REPAIRED_BASELINE_REPORT.md` | primary-untracked | DOC (repaired 23/24 baseline) | `fbc0b20a2e28770c…` | byte-identical | colab-v3:MATCH |
| `model/compat/constraints-verified.txt` | primary-untracked | ENV-SPECIFIC (ML env pins) | `635678da72cb72c5…` | byte-identical | colab-v3:MATCH |
| `model/compat/pyproject.toml` | primary-untracked | ENV-SPECIFIC (ML env project) | `a3e4884335c6a380…` | byte-identical | colab-v3:MATCH |
| `model/compat/requirements-colab-py311.txt` | primary-untracked | ENV-SPECIFIC (Colab pins) | `dffc8d7d95349602…` | byte-identical | colab-v3:MATCH |
| `pyproject.toml` | primary-modified-tracked | CONFIG (data-files for recovered YAMLs) | `6959f1989d176ab1…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/backends/webots/backend.py` | primary-modified-tracked | SCRIPTED-PRIMARY-RUNTIME (frozen-baseline-locked) | `37f084766041f3bf…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `src/vlarobot/backends/webots/config.py` | primary-modified-tracked | SCRIPTED-PRIMARY-RUNTIME (registration setting) | `51d031dcaf86d9bc…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/backends/webots/images.py` | primary-untracked | SCRIPTED-PRIMARY-RUNTIME (camera RGB decode) | `1533d84b79fb2354…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `src/vlarobot/backends/webots/kinematics.py` | primary-modified-tracked | SCRIPTED-PRIMARY-RUNTIME (frozen-baseline-locked) | `cd564d9acd030ee1…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `src/vlarobot/backends/webots/pose_solver.py` | primary-untracked | VLA-RUNTIME (bounded IK adapter) | `527550b2ff801b3a…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/backends/webots/urdf_registration.py` | primary-untracked | SCRIPTED-PRIMARY-RUNTIME (URDF registration) | `40b82d6feaff21d7…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `src/vlarobot/cli.py` | primary-modified-tracked | CLI (eval-vla routing + new commands) | `ac57b135e0ea81d6…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/eval/baseline_audit.py` | primary-untracked | EVAL-DIAGNOSTIC (baseline audit) | `2a81f78234d7134b…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/eval/demo_audit.py` | primary-untracked | TRAINING (demo admission audit) | `995ae65eb8910fa2…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/eval/grasp_diagnostics.py` | primary-untracked | EVAL-DIAGNOSTIC | `0f341fc5de86803c…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/eval/runner.py` | primary-modified-tracked | EVAL RUNNER (reconciled, see modifications) | `a70c1b4017827067…` | MODIFIED-IN-STAGING | colab-v3:POST-V3-EDIT |
| `src/vlarobot/eval/vla_metrics.py` | primary-untracked | EVAL-DIAGNOSTIC (VLA metrics) | `df0e527947584274…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/colab_bundle.py` | primary-untracked | TRAINING (Colab handoff) | `ba5d7cc9910db38a…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/collection_health.py` | primary-untracked | TRAINING ops | `6c6fdaa20761cd19…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/collection_watchdog.py` | primary-untracked | TRAINING ops | `04a96ab49cf5eee7…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/dataset_validation.py` | primary-untracked | TRAINING (dataset validation) | `862f653454f9daa1…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/demo_collection.py` | primary-untracked | TRAINING (demo admission) | `e16d69815dd0eca8…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/parallel_eval.py` | primary-untracked | EVAL-DIAGNOSTIC (parallel eval) | `a157a6d22c105b24…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/scripted_baseline.py` | primary-modified-tracked | SCRIPTED-PRIMARY-RUNTIME (frozen-baseline-locked) | `d304fef311e6ec7f…` | byte-identical | colab-v3:MATCH; baseline-prov:MATCH |
| `src/vlarobot/execution/surrogate_validation.py` | primary-untracked | EVAL-DIAGNOSTIC | `ac82a5806dba417a…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/vla_campaign.py` | primary-untracked | VLA-RUNTIME (frozen eval-vla campaign) | `793de9b859d12ec0…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/vla_episode.py` | primary-untracked | VLA-RUNTIME (episode runner) | `ffb73839499392e6…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/execution/vla_validation.py` | primary-untracked | EVAL-DIAGNOSTIC (read-only) | `0e72da67365fae76…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/__init__.py` | primary-untracked | VLA-RUNTIME package | `5f3a147e382339a1…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/action_adapter.py` | primary-untracked | VLA-RUNTIME (normalisation/EE-IK/chunking) | `b95fec69be7af3bd…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/checkpoint_export.py` | primary-untracked | VLA-RUNTIME + TRAINING export | `bbb6c72e8032a202…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/dataset_export.py` | primary-untracked | TRAINING (dataset build) | `d4853484517b3024…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/dataset_integrity.py` | primary-untracked | TRAINING (dataset admission) | `03592a82e898151b…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/encoded_images.py` | primary-untracked | TRAINING (dataset build) | `60ef1646d806444b…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/model_service.py` | primary-untracked | VLA-RUNTIME (model-process ZMQ RPC) | `58f4e1fe499e1060…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/padding.py` | primary-untracked | TRAINING (LeRobot compat) | `b1a42e91b4de8840…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/schema.py` | primary-untracked | VLA-RUNTIME data contract | `71283d4427264798…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/settings.py` | primary-untracked | VLA-RUNTIME config loader | `f86bed34d36489a5…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/smoke.py` | primary-untracked | EVAL-DIAGNOSTIC (CPU smoke) | `63ddde52dfb49807…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/training.py` | primary-untracked | TRAINING (approved resumable run) | `825d0add165ce04e…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/training_evidence.py` | primary-untracked | TRAINING (optimizer evidence) | `87dacb750f7ce352…` | byte-identical | colab-v3:MATCH |
| `src/vlarobot/vla/training_runs.py` | primary-untracked | TRAINING (run selection) | `098ce08cccb4c558…` | byte-identical | colab-v3:MATCH |
| `tests/fixtures/urdf_registration_holdout.json` | primary-untracked | TEST fixture | `4bdfea4adfbc724c…` | byte-identical | colab-v3:MATCH |
| `tests/test_baseline_audit.py` | primary-untracked | TEST | `70b5caa6e31f83f7…` | byte-identical | colab-v3:MATCH |
| `tests/test_collection_health.py` | primary-untracked | TEST | `aebd106c43c05376…` | byte-identical | colab-v3:MATCH |
| `tests/test_collection_monitor.py` | primary-untracked | TEST | `dc98639ccbfd2f60…` | byte-identical | colab-v3:MATCH |
| `tests/test_dataset_integrity.py` | primary-untracked | TEST | `2691a19e559598f8…` | byte-identical | colab-v3:MATCH |
| `tests/test_demo_admission.py` | primary-untracked | TEST | `dc2c96e22c7df41e…` | byte-identical | colab-v3:MATCH |
| `tests/test_demo_collection.py` | primary-untracked | TEST | `2aad4c85f8fe5347…` | byte-identical | colab-v3:MATCH |
| `tests/test_eval_runner.py` | primary-modified-tracked | TEST (coupled to cli.py) | `af018c4bfcda8ede…` | byte-identical | colab-v3:MATCH |
| `tests/test_grasp_diagnostics.py` | primary-untracked | TEST | `44ec432e27c482a2…` | byte-identical | colab-v3:MATCH |
| `tests/test_model_service.py` | primary-untracked | TEST | `ce695b81f8c6c148…` | byte-identical | colab-v3:MATCH |
| `tests/test_optimizer_evidence.py` | primary-untracked | TEST | `52899c269c5184f3…` | byte-identical | colab-v3:MATCH |
| `tests/test_parallel_eval.py` | primary-untracked | TEST | `84a55638c66468a1…` | byte-identical | colab-v3:MATCH |
| `tests/test_t4_preflight.py` | primary-untracked | TEST | `83de2afbbf7561fd…` | byte-identical | - |
| `tests/test_training_runs_bundle.py` | primary-untracked | TEST | `e58fc86ad6f82601…` | byte-identical | colab-v3:MATCH |
| `tests/test_urdf_registration.py` | primary-untracked | TEST | `0a051078a3b06922…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_adapter.py` | primary-untracked | TEST | `94bb63dd95f43e7b…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_checkpoint_export.py` | primary-untracked | TEST | `b923afaa8ca2cfbf…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_encoded_images.py` | primary-untracked | TEST | `8f50347ab396732b…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_episode_gate.py` | primary-untracked | TEST | `4ae4716e17ecc943…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_padding.py` | primary-untracked | TEST | `ebc0c5846b87f524…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_pose_solver.py` | primary-untracked | TEST | `aab78fa7c1ddd70b…` | byte-identical | colab-v3:MATCH |
| `tests/test_vla_training.py` | primary-untracked | TEST | `4554f20391320b78…` | byte-identical | colab-v3:MATCH |
| `tests/test_webots_backend_reset.py` | primary-modified-tracked | TEST (coupled to backend camera API) | `9137d4f8e896a3be…` | byte-identical | colab-v3:MATCH |
| `tests/test_webots_images.py` | primary-untracked | TEST | `80972f4a8ca836bc…` | byte-identical | colab-v3:MATCH |
| `tools/t4_preflight.py` | primary-untracked | TRAINING diagnostic tool | `0a53b07326e04dc9…` | byte-identical | - |

## Intentional modifications on staging copies

| Path | Modification | Reason |
|---|---|---|
| `src/vlarobot/eval/runner.py` | Restored the four contracted pipeline aliases (`scripted_baseline`, `vla_risk_veto`, `vla_world_model`, `full_system`) and derived `IMPLEMENTED_PIPELINES` from the alias table (the committed analytic form). | Reconciles the primary runner (which had narrowed the alias table for the frozen `eval-vla` routing) with the published analytic tree so that all five contracted arms remain available. Arm semantics, seeds, thresholds, split guards, and the CLI `vla_only` → `eval-vla` guard are unchanged. `eval/runner.py` is NOT among the 12 baseline-provenance-locked files, so the frozen `eval-vla` fail-closed source gate is unaffected. |

### CI/type-only modifications (applied after recovery; verified non-behavioral)

| Path | Modification | Reason |
|---|---|---|
| `tools/{independent_candidate_validation,numerical/ad1,planning_equivalence,profile_registered_replay,recovery/restore_gate,resume_validation_parallel}.py` (6 files) | Pure import reordering via `ruff check --fix` (I001). Verified: zero non-import, non-blank changed lines. | With the recovered modules present, Ruff reclassifies `vlarobot.*` as first-party, making these pre-existing late-import blocks unsorted. Lint-only. |
| `src/vlarobot/backends/fake/backend.py` | Added `Sequence` import; two inline `cast()` calls (ground-truth tuple/float reads). | `ty` invalid-argument-type findings; runtime behavior unchanged (values already floats at runtime). |
| `src/vlarobot/eval/metrics.py` | One inline `cast(float, value)` in `_optional_float`. | `ty` finding; no behavior change. |
| `src/vlarobot/eval/recorder.py` | `Iterable` import; one inline `cast(Iterable[object], contact_nodes)`. | `ty` not-iterable finding on a `dict.get(...)` result; no behavior change. |
| `src/vlarobot/eval/runner.py` | `cast` import; three `int(cast(int, …))` sites (contact/rejection counters) and the finite-float comprehension rewritten to filter-then-check (same semantics). | `ty` findings on `dict[str, object]` row access; no behavior change. |
| `src/vlarobot/eval/reset_repro.py` | Four `cast(Position/Quaternion, …)` at construction (lines 64–67); four now-redundant casts removed (145/148); reformatted by `ruff format` (E501 from the casts). | `ty` findings; net line-count change only. |
| `src/vlarobot/execution/episode_runner.py` | `Iterator[WebotsSession]` → `Generator[WebotsSession, None, None]` on the session context-manager factory. | Deprecated `contextmanager` overload; semantically identical type. |
| `src/vlarobot/backends/webots/geometry_probe.py` | Documented `# ty: ignore[import-not-found]` on the `controller` SDK import. | The Webots SDK only exists inside the simulator runtime and can never resolve in CI; the ignore documents that fact instead of hiding a real defect. |
| `pyproject.toml` | Added `[tool.ruff.lint.per-file-ignores]`: `notebooks/smolvla_finetune_colab.ipynb = ["E402"]` with explanatory comment. | The notebook is frozen/approved (`docs/COLAB_EXECUTION.md`) and must not be edited to satisfy lint; the exception is scoped to E402 on that single file. Other notebook checks remain active. |

### Deferred diagnostics (NOT fixed — owner constraints)

| Site | Diagnostic | Constraint |
|---|---|---|
| `src/vlarobot/backends/webots/backend.py:347` | `invalid-argument-type` (float constructor) | File is one of the 12 frozen-baseline-provenance-locked sources; any edit breaks the 23/24 baseline hash gate. |
| `src/vlarobot/diagnostics/ad2/admission.py:86` | `invalid-argument-type` (isfinite argument) | A-D2 closed infrastructure (CLOSED/NO-GO); changes require separate owner review. |

## Excluded from recovery

| Excluded | Reason |
|---|---|
| notebooks/smolvla_finetune_colab.ipynb (primary modified) | Not part of the 76-file recovery copy; adopted separately, byte-for-byte, as its own commit after identity verification (see below). |
| docs/{ARCHITECTURE,DECISIONS,PROGRESS,SETUP,VERSIONS,VLA_NOTES}.md (primary modified) | Divergent documentation histories; doc reconciliation is a separate review topic. Overwriting the committed DECISIONS.md would discard analytic-tree ADRs. |
| src/vlarobot/risk/**, src/vlarobot/world_model/{analytic,kinematic}.py, tests/test_analytic_risk.py, tests/test_kinematic_surrogate.py (primary untracked) | Newer committed analytic versions exist and are retained; older primary copies were NOT imported (owner instruction). |
| model/compat/build/**, *.egg-info, .venv-vla/, runs/** | Generated build artifacts, virtual environments, and gitignored experiment outputs are excluded. |
| model weights (runs/local-model-validation-v1/export-step-020000, model/lerobot) | Large model weights and frozen experiment outputs are excluded; see companion-bundle specification. |

## Training notebook identity (verified; adopted byte-for-byte in its own commit)

- Committed (published PR) notebook: `936d6cfa68fa…` — older scaffold (10 cells, no T4 preflight).
- `runs/colab-prep-v2` snapshot: `d3da4448…`; `runs/colab-prep-v3` zipped snapshot: `51d1a818…` (pre-preflight generation shipped in the Colab bundle).
- Primary disk (adopted): `cccdaf3f…` — the v3 notebook plus an inserted T4-preflight flow (14 cells; train/export cells otherwise unchanged).
- `docs/COLAB_EXECUTION.md` (updated 2026-11-15) explicitly instructs: “Use the current local notebook, not the older copy contained inside frozen source.zip.” The primary disk notebook is therefore the approved executed training notebook; the committed scaffold predates it.
- No committed test or tool pins the training notebook bytes (`tests/test_account_migration_notebook.py` covers `smolvla_account_migration.ipynb`).
- The 160/164 v3-bundle match for the primary tree covers all production source; the four post-packaging edits are `docs/DECISIONS.md`, `docs/SETUP.md`, `docs/VERSIONS.md` and this notebook.
