# Versions and compatibility

| Component | Pin/target | Verification |
|---|---|---|
| Python | 3.11.18 (`.python-version`; project supports Python 3.11) | Installed and tested on Fedora/Linux; Windows CI configured, not yet run |
| Webots | R2025a | `worlds/smoke.wbt` and external-controller smoke tested on Fedora/Linux; Windows unverified |
| PyTorch | Not pinned | Unverified; select a CUDA wheel per OS using the official selector |
| LeRobot | Not pinned | Unverified; model/training dependencies are not installed |
| MuJoCo | Not pinned | Unverified; surrogate backend is not implemented |
| ty | 0.0.99 (lockfile) | Installed and type-check passed on Fedora/Linux; Windows CI pending |
| uv | Lockfile managed by uv | `uv lock --check` passes in this environment |
| msgpack | 1.2.4 (lockfile) | Installed; wire round-trip tests pass |
| Simulator/brain protocol | Version 3 | Backend diagnostics added; Windows and live Webots pairing unverified |
| NumPy | 2.4.6 (lockfile) | Installed; array wire tests and live Webots RGB transfer pass |
| PyYAML | 6.0.3 (lockfile) | Installed; typed robot settings tests pass |
| pyzmq | 27.3.0 (lockfile) | Installed; FakeBackend transport and live Webots RPCs pass |
| Typer | 0.28.0 (lockfile) | Installed; `smoke`, `sim-server`, and `brain` CLI commands load |
| IKPy | 4.4.4 (lockfile, `sim` extra) | Installed; Webots backend initialized in the live factory-world check |
| SciPy | 1.17.1 (IKPy dependency) | Installed for the local backend check |
| SymPy | 1.14.0 (IKPy dependency) | Installed for the local backend check |

`pyproject.toml` constrains the supported interpreter to Python 3.11. CI and local development pin
the exact interpreter patch in `.python-version`. Webots is pinned to R2025a by the project world
files and was verified with the local external-controller smoke test. Cross-platform verification
is incomplete until Webots and CI pass on Windows.

## M1 dependency and backend verification

The versions above are the Fedora/Linux lockfile versions as of 2026-11-07; Windows CI has not yet
run. The Webots backend was exercised against the local R2025a factory world through the brain
process. The RGB observation shape was verified as HxWx3. `T_base_cam` is currently an identity
placeholder and must not be used for calibrated localization before M3. The built wheel contains
both `configs/robot.yaml` and the Webots TIAGo URDF.

## M0 compatibility and open-item audit

The versions below are project selections only; the Webots or learning-library compatibility
questions remain open wherever noted.

| Plan §9 item | M0 evidence | Status / remaining verification |
|---|---|---|
| Webots release, TIAGo PROTO, Python compatibility on Windows/Linux | R2025a smoke test passed on Fedora/Linux with Python 3.11.18 | Windows compatibility and smoke test remain unverified |
| Compatible torch/LeRobot/Transformers set and Windows training | None of these packages are pinned or installed | UNVERIFIED; choose versions and verify Windows/WSL2 training later |
| `smolvla_base` inputs, outputs, state/action dims, camera keys | Existing `model/main.py` references `lerobot/smolvla_base` but its config was not loaded | UNVERIFIED; inspect with the compatible LeRobot/model install and record in `VLA_NOTES.md` |
| TIAGo URDF import into MuJoCo | MuJoCo is not installed and surrogate import is not implemented | UNVERIFIED; test at M5 |
| Licences of selected model weights/software | No detector, segmenter, LLM, VLA runtime, or checkpoint has been selected for the new pipeline | Model licences remain UNVERIFIED until selection; the current locked Python dependencies have permissive licenses (see `DECISIONS.md`) |
| Webots collision/contact-event access | M0 smoke world only checks camera and stepping | UNVERIFIED; validate the contact metric when implementing M2 evaluation |


ADR-0037 adds only standard-library/numpy implementation code. No dependency, model, simulator velocity or training configuration was changed. Local profile uses existing IKPy/scipy through the accepted registered FK adapter; physical collision coverage and probabilistic calibration remain unverified.

The 017000 recovery helper uses the existing Python 3.11 standard library and
approved installed LeRobot stack. It introduces no dependencies or training
configuration changes. Linux guard tests pass; Windows and authenticated Colab
execution remain unverified. Recovery is a separately identified continuation,
not a bit-for-bit replay (dataloader cursor is not checkpointed by this LeRobot).

Recovery helpers v3 add only standard-library canonical project-root handling
and diagnostic path auditing. They supersede v2 for the confirmed Drive shortcut
layout. Model/dataset/configuration and the pre-update CUDA restore gate are unchanged.

Durable recovery v4 adds standard-library transfer/packaging tools and uses the
existing system OpenSSH client with strict host-key checking. The independent
receiver is Linux. Colab's existing Google auth/Drive SDK is used only for optional
read-only diagnostics in the notebook kernel; no training dependency pins change.
Real SSH/Colab execution and Windows sender compatibility remain unverified here.

Account migration v5 preserves the approved Python/uv/LeRobot/Torch requirements.
It adds standard-library account binding and processor relocation helpers only.
The11 metadata files are byte-for-byte copies pinned by the original portable
export manifest. Actual installed config and pre/postprocessor loading passed on
CPU with no model construction/inference or optimizer update; new-account CUDA,
Google authentication, permissions and transport remain unverified locally.
# Recovery transport v6 (2026-11-16)

v7 corrects historical canary-version comparison without altering checkpoint
upload integrity. A numeric version advance requires fresh full SHA-256 read-back,
unchanged identity/ownership/parent/size and stable current version across the read.
The original receipt/config remain unchanged; advances produce a separate local
audit record. Existing Colab authentication is reconnected without new remote files.
No dependency, model, dataset or frozen experiment setting changed.

Drive API resumable checkpoint handoff replaces unavailable SSH for the current
second-account session. No model, optimizer, dataset, batch, seed, step target,
save cadence, source bundle or ML dependency pin changes. Worker transport uses
Python 3.11 stdlib; Google authentication/SDK stays in the already configured
Colab notebook kernel. Actual Google permissions and CUDA restore remain unverified
locally; mocked REST tests do not certify the production session.

Measured recovered step20000 local runtime: Python3.11.18, torch2.7.1+cpu, torchvision0.22.1+cpu, LeRobot0.4.3, transformers4.57.1, safetensors0.8.0, numpy2.2.6. Actual CPU inference passed; no exposed NVIDIA device or CUDA build, so RTX3050 4GB fit is unverified. Physical gripper outputs fail frozen bounds; see LOCAL_MODEL_VALIDATION_STEP20000.md. No dependency changes were made.

A-D1 used the frozen brain interpreter: Python3.11.18, IKPy4.4.4, NumPy2.4.6, SciPy1.17.1, SymPy1.14.0, matching the approved source lock. The ML environment (IKPy4.5.0) only extracted the native row and performed no IK. No dependency change; package/source hashes in the A-D1 evidence.

A-D2 isolated CPU infrastructure: standard-library diagnostic package and existing
NumPy/Typer offline audit only; no new dependencies, model/processor/configuration or
runtime controller changes. Actual pinned-input audit and60 CPU fixtures are documented
in `AD2_CPU_INFRASTRUCTURE_REPORT.md`. Live snapshot behavior and physical safety remain
unverified; no approved numerical reset protocol or live execution entry point is shipped.

Production WorldMind composition (ADR-0047) adds no dependency or version pin. It uses the
existing Python 3.11 standard library, NumPy, PyYAML, Typer, the in-repo FakeBackend and the
analytic surrogate. `configs/worldmind_production.yaml` is added to packaged data files. The
`eval-worldmind` command is fail-closed with the blocked template and empty binding registry;
no Webots-measured, scored VALIDATION, Windows, real-IK or inference result is claimed.

Concrete registration, calibration gates and structured preflight (ADR-0048) add no dependency or
version pin. Existing IKPy/NumPy geometry and safety objects are bound lazily behind
`backends/worldmind_bindings.py`; `configs/report/definitions.candidate.json` is added to packaged
data files. `eval-worldmind` now registers `production_worldmind_bindings()` and fails closed via
`eval/worldmind_preflight.py` until owner-approved report/calibration values and scientific bindings
exist. No Webots-measured, scored VALIDATION, Windows, real-IK or inference result is claimed.

Calibration bootstrap, Eq. 3.8 proposal, swept-arm gate and waypoint cadence (ADR-0049) add no
dependency or version pin. The new `worldmind-diagnostic` and `worldmind-calibration-proposal`
commands, diagnostic config and objective/sweep modules use the existing Python 3.11 standard
library and NumPy only. Linux inert checks pass; Windows, Webots, real IK, inference, training,
actuation, scored VALIDATION and pilot execution remain unverified. Existing version pins and
frozen evidence are unchanged.

No-actuation Webots diagnostic (ADR-0050) adds no dependency or version pin. It reuses the existing
Python 3.11 standard library, NumPy, PyYAML, Typer, ZeroMQ/msgpack and the in-repo Webots backend.
The new protocol operation is `inspect` only; it does not change `core/wire.py`. Linux inert checks
pass. No Webots process, real IK solve, inference, training, actuation, scored evaluation,
TEST/TRAIN access or pilot execution was performed; Windows behavior remains unverified.

Concrete full-arm sweep integration and local geometry diagnostic (ADR-0054) add no dependency or
version pin. They reuse Python 3.11 standard library, NumPy, PyYAML, Typer, the existing IKPy/URDF
registration and the in-repo Webots geometry inventory behind the backend boundary. New packaged
config files are `worldmind_arm_sweep.yaml` and `worldmind_geometry_diagnostic.yaml`; no public
`core/` interface or frozen pin changed. Linux inert checks and the local-only diagnostic pass;
Windows, Webots execution, real IK, actuation, scored VALIDATION and pilot execution remain
unverified. The `unverified-arm-sweep` hard veto is unchanged until an independent safety owner
supplies a certificate.

Local PROTO resolution, conservative static-scene/swept checks and dedicated read-only Webots
geometry inspection (ADR-0055) add no dependency or version pin. They reuse Python 3.11 standard
library, NumPy, PyYAML, Typer, ZeroMQ/msgpack and the in-repo Webots backend. The backend-only
`inspect_geometry` RPC does not change `core/wire.py`; public `core/` interfaces and frozen evidence
are unchanged. New packaged data files are
`configs/worldmind_webots_geometry_inspection.yaml`, `configs/geometry/scene-collision-v1.yaml`,
`configs/geometry/webots-proto-sources-v1.yaml` and 32 byte-pinned R2025a PROTO sources under
`assets/webots/protos/r2025a/`. Linux inert checks pass; Windows, live Webots execution, real IK,
actuation, scored VALIDATION and pilot execution remain unverified. Production still has
`check_transition=None`, sweep status `blocked`, `certificate=None` and motion readiness
`NOT_READY`.

Fail-closed swept-residual accounting, frame-aware gripper evidence, bounded CAN lifecycle,
self-collision pair coverage and the non-approving motion-readiness dossier (ADR-0057) add no
dependency or version pin. They reuse Python 3.11 standard library, NumPy, PyYAML, Typer and the
existing pinned geometry/URDF/PROTO sources. New internal modules and config data files are used
only by inert reconciliation and CLI reporting; no public `core/` interface, production callback,
frozen live-artifact byte, seed, split or safety threshold changed. Linux inert checks pass;
Windows, Webots execution, real IK, actuation, scored VALIDATION and pilot execution remain
unverified. The production `check_transition=None`, `unverified-arm-sweep` veto and
`NOT_READY` status remain in force.

Selector-only five-box read-only geometry inspection (ADR-0058) adds no dependency or version pin.
It reuses Python 3.11 standard library, NumPy, PyYAML, Typer, ZeroMQ/msgpack and the existing
Webots backend boundary. The geometry-inspection config schema moves to `2` and the runtime table
observation schema is `1`; both are internal to this diagnostic. `core/wire.py` and public `core/`
interfaces are unchanged, as are the CLI flags `--config` and `--out` and the four evidence file
names. Linux inert checks, Ruff, `ty`, `compileall` and frozen-evidence verification pass; Windows
and live Webots execution remain unverified. Motion stays `NOT_READY` and the new
`unverified-runtime-table-transform` blocker remains until one authorized read-only Webots
observation grounds all five boxes.

ADR-0073 adds no dependency or version pin. It reuses Python 3.11 standard library, NumPy, SciPy,
IKPy and the existing pinned corrected registration. Endpoint solving now uses full registered
bounds with 20 draw-seeded restarts; execution still uses the unchanged `0.15 rad` per-tick limit
from `configs/robot.yaml`. No public `core/` interface, config, seed, split, safety threshold,
frozen evidence byte or live authorization changed. Linux offline regression, Ruff and formatting
checks pass; Windows and all Webots/physical paths remain unverified.

ADR-0074 adds no dependency or version pin. The tracked `configs/worldmind_diagnostic.yaml` is the
frozen blocked template; `configs/worldmind_diagnostic.approved.yaml` is the separate non-scored
owner-approved diagnostic instance used by offline endpoint reconstruction and any future
authorized evidence collection. The unchanged `corrected_bounded_pose_solver_v1` binding now has a
versioned implementation identity in `docs/DECISIONS.md`: `_FullPoseEndpointSolver` with SHA-256
`4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae`. The 20 restarts and the
deterministic residual/joint-delta tie-break are implementation choices, not calibrated
parameters. No public `core/` interface, binding identity, seed, split, `K=32`, `H=50`, safety
threshold, report definition, preserved evidence byte or live authorization changed. Linux
offline regression, Ruff and formatting checks pass; Windows and all Webots/physical paths remain
unverified.

ADR-0075 adds no dependency or version pin. It reuses Python 3.11 standard library, NumPy, the
existing pinned corrected URDF registration and `SafetyMonitor`. The `scripted_chunk_adapter`
now enforces reviewed absolute joint limits during jittered endpoint construction and one-tick
target construction while retaining the unchanged `0.15 rad` execution bound, endpoint residual
tolerances and every `SafetyMonitor.validate()` call. The eval runner can finalize a partial
18-episode plan without calling the nonempty-only success aggregator on empty scene rows, writes
all four aggregate files, preserves planned scene/seed records and the original EXCEPTION reason,
finalizes WorldMind sidecars, and exits through `CampaignFailStop`. No public `core/` interface,
config, seed, split, safety threshold, report definition, preserved V1/V2 evidence byte or live
authorization changed. Linux offline focused tests, the 32-candidate reconstruction replay,
Ruff and formatting checks pass; the full offline suite has only the pre-existing unrelated
notebook helper SHA mismatch. Windows and all Webots/physical paths remain unverified.

ADR-0077 adds no dependency or version pin. It reuses Python 3.11 standard library, NumPy, the
existing pinned corrected URDF registration, `SafetyMonitor` and the frozen diagnostic config.
The only source change moves open-loop `backend_step_count` accounting ahead of early abort
breaks so safety-aborted chunks retain every accepted control step; no cadence, simulator step,
candidate, safety, verifier, threshold, seed, split, scoring, calibration or full-system behavior
changed. A and B remain diagnosed but uncorrected pending owner approval for frozen behavior
changes. No public `core/` interface, V1-V4 evidence byte or live authorization changed. Linux
offline focused and relevant regression tests, Ruff and formatting checks pass; Windows, Webots,
physical actuation, V5 and scored behavior remain unverified. Current source is **NO-GO** for
another live campaign.

ADR-0078 adds no dependency or version pin. It reuses Python 3.11 standard library, NumPy, the
existing pinned corrected URDF registration, reviewed `configs/robot.yaml` values, the pose
solver's `0.15 rad` execution bound, `SafetyMonitor` and the frozen diagnostic config. The new
`observed-state-feedback.jsonl` schema is version 1 and is created only for the explicit
`diagnostic_observed_state_feedback_chunk` mode; it changes no public `core/` interface, config,
seed, split, candidate, safety threshold, verifier threshold, report definition, scoring,
calibration, `full_system` path, V1-V4 evidence byte or live authorization. Feedback trajectories
are non-comparable with the immutable H=50 prediction and emit no prediction pairing. Linux offline
focused and relevant regression tests, Ruff, formatting and `git diff --check` pass; the full
offline suite has only the pre-existing unrelated notebook helper SHA mismatch. An additional `ty`
probe reports repository-wide pre-existing diagnostics and is not an acceptance gate. Windows,
Webots, physical actuation, V5 campaigns and scored behavior remain unverified. Conditional GO is
limited to a separately authorized, non-scored diagnostic in a fresh output directory.

ADR-0079 adds no dependency or version pin. It reuses Python 3.11 standard library, NumPy, the
existing pinned corrected URDF registration, reviewed `configs/robot.yaml` and `configs/eval.yaml`
values, the pose solver's `0.15 rad` execution bound, `SafetyMonitor`, `PlacementVerifier` and the
frozen diagnostic config. The observed-state feedback sidecar is bumped from schema 1 to schema 2;
schema 1 rows remain readable. Version-1 `diagnostic-events.jsonl` and
`collision-proxy-attribution.jsonl` sidecars are additive, diagnostic-only channels. No public
`core/` interface, config, seed, split, candidate, safety or verifier threshold, report definition,
scoring, calibration, `full_system` path, V1-V5b evidence byte or live authorization changed. No
new numerical parameter was introduced: gripper arrival tolerance and timeout are derived from
existing reviewed values. Linux offline focused and relevant regression tests, Ruff, formatting and
`git diff --check` pass; the full offline suite has only the pre-existing unrelated notebook helper
SHA mismatch. Windows, Webots timing, physical actuation, V5c, scoring and production use remain
unverified. **NO-GO** for a live/scored V5c campaign; conditional GO remains limited to a
separately authorized, non-scored diagnostic in a fresh output directory.
