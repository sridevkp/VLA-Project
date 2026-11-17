# A-D2 isolated CPU infrastructure — live execution remains NO-GO

Owner authorization covers isolated implementation and fixture testing only. This report
supersedes the earlier designs' **not implemented** status only for the offline components
listed below. Neither physical A-D2 nor live snapshot instrumentation is authorized or
implemented. Frozen **0/24, FAILED gate, scripted-primary** remains unchanged, as do the
checkpoint/processor/data, safety limits, thresholds, scene splits/seeds, five arms, HOME
behavior and December12 deadline. No validation/TEST scene was opened.

## Implemented and verified

| Component | Evidence / behavior |
|---|---|
| `src/vlarobot/diagnostics/ad2/evidence.py` | Caller-pinned SHA-256 manifest; all files verified before target processing; original provenance coverage checked; JSON bytes rechecked at parse, duplicate/nonfinite JSON rejected; traversal/symlink paths rejected. Frozen tuples expose original saved targets, with A=B exact and C equal to its independent seal and both A-D1 copies. No IK or action reconstruction. |
| `admission.py` | Immutable typed states/config/units/protocol; exact categorical identity; separately supplied reference/pairwise ceilings; sign-invariant quaternion angle; missing/nonfinite/schema/unit rejection. Named historical omissions require explicit observability version/reasons and never omit live/pairwise measurements. No numerical defaults or activated exploratory bounds. |
| `fixtures.py` | Fixed A/B/C inert session descriptions only; no backend/client/factory/executable accepted. Tests use `fixture:` approval IDs. Separate startup/reset/measurement counts, one step decision after acceptance, nonce/sequence rejection, no retry/replacement, owned-session cleanup and fsynced partial-event records. Cleanup cannot progress physics because no physics API exists. |
| `SnapshotBuffer` / `snapshot` | Read already-captured deeply immutable CPU values only. No getter callbacks, observation helpers, motor/sensor APIs or velocity-history cache reference. This is **not** a live Webots reader or proof of live API behavior. |
| `geometry.py` | Static XML/PROTO/world source enumeration with hashes, link collisions/adjacency/limits, nested table children, controller literals and coverage gaps. No FK, IK, sweep or physical probe. Adjacency is explicitly **not an approved self-collision exclusion**. |
| `tools/diagnostics/ad2_audit.py` | Cross-platform offline Typer command; checks actual immutable inputs and exports static source inventory/recorded TRAIN frame0 object poses. Reads NPZ with pickle disabled. No simulator/model/solver import or entry point. Refuses an existing output directory. |

The evidence trust anchor is
[ad2-immutable-inputs-v1.json](evidence/ad2-immutable-inputs-v1.json), SHA-256
`207f66a7c8e24de619e0034300844a90c5b1ed36bdfb7c97525773bd4dfda220`.
Its71 sealed files cover all65 original A-D1 input hashes plus additional A-D1 records;
it preserves hashes rather than rebasing or repairing missing inputs. Callers must supply
explicit main/analytic roots and the trusted manifest digest. This digest is not inferred
from the file being checked. Portable code does not imply the Linux evidence tree exists
on Windows; Windows execution has not been tested.

**Actual offline audit passed:** all65 original input hashes matched, A=B exactly, and C
retained its saved values. Model weights were hashed, **not loaded or inferred**. Final
audit duration4.488356918s. Python/NumPy/environment details and full targets are recorded
in [artifact-loading evidence](evidence/ad2-artifact-loading-v1.json). Source-derived rows
remain raw recorded values, not reconstructed dynamics.

## Static collision inventory — not certification

[Committed inventory](evidence/ad2-static-geometry-v1.json), generated from existing assets:

- Complete static enumeration of59 URDF links;30 have collision elements. Arm, shoulder,
  wrist connector, gripper body/fingers and relevant robot-body links are explicit.
- 58 URDF joint adjacency records include parent/child/type/limits, all with
  `exclusion_approved=false`. Source limits are not measured runtime motor settings.
- 34 cached PROTO sources cover required robot components and world EXTERNPROTO names,
  including the hyphenated `Darwin-op` identifier. No required source is missing in this
  local cache, but composition/template resolution and geometric containment remain partial.
- Table top and all four nested feet are explicitly identified at Table source lines
  59,67,74,81,88, with SolidBox dependency included. Merely listing those instances does not
  certify their expanded transforms/coverage; the old live export's omission is not repaired
  by pretending it was complete.
- Exact saved TRAIN ScenarioSpec and frame0 APP/CAN/HAM xyz/quaternions are included;
  velocities stay `null`, explicitly unavailable. No extra scene was generated or simulated.
- Floor Plane/support contacts remain **unsupported** by a certified collision check.
  Self-collision is disabled by default in the source; the existing external-contact logger
  filters self contacts. Neither fact establishes a safe exclusion list.
- Literal controller candidates include valve_turner and robot motion controllers.
  They are source candidates, not a claim all run in this scene: world overrides and PROTO
  propagation must be resolved. Their possible motion cannot silently be treated as static.
- Swept-volume coverage is **unsupported**: neither endpoint checks nor primitive counts
  prove conservative coverage under independent motor dynamics, passive joints, base motion,
  moving obstacles and incomplete frame registration. No clearance margin was invented.

Overall materialized collision coverage remains **PARTIAL** and collision safety is
**NOT CERTIFIED**. Complete source availability is not complete conservative geometry proof.
No physical geometry probe or live API call occurred.

## CPU-only acceptance commands and results

Working directory: `/home/tj330/programs/VLA-Project-analytic`.

```text
uv run ruff check src/vlarobot/diagnostics tools/diagnostics tests/test_ad2_infrastructure.py
uv run ruff format --check .
uv run pytest -q tests/test_ad2_infrastructure.py
```

Final fixture output: **60 passed in0.55s**. JUnit record:
`runs/ad2-offline-infrastructure-v3/fixture-tests.xml` (copied from the final test invocation's
v2 output path). Fixtures exercise immutable loading/tampering/provenance, unit/schema gaps,
reference/pairwise admission, quaternion signs/norms, explicit omissions, ordering/isolation,
rejection without stepping, ambiguous apply/step without resend, reply identity mismatch,
preparation/tick/time accounting, partial persistence/storage failure, no fourth session,
owned cleanup and conflicting outputs. A real pure SafetyMonitor rejects synthetic invalid
joint commands without constructing a backend. Fixture import-surface checks prohibit
simulator, model, solver, subprocess and socket dependencies in the diagnostic package.
No fixture targets are physical commands; synthetic tolerances are not approved criteria.

New code lint passes. Repository-wide formatting passes. Repository-wide `ruff check .`
reports one pre-existing E402 at `notebooks/smolvla_finetune_colab.ipynb`, cell3 line6.
The frozen notebook is unchanged. This is a reported lint limitation, not a repaired
training artifact. The broad `pytest -m "not webots"` suite was deliberately **not run**:
non-Webots tests elsewhere may perform IK/inference; the owner's prohibition takes
precedence. Only the isolated, inspected CPU fixture suite ran. No fixture pass certifies
physical safety, motor realization or simulator behavior.

Reproduce the real offline source audit below with a **new** output path. This command
only hashes/reads existing TRAIN evidence and source files; it cannot launch A-D2:

```text
uv run python tools/diagnostics/ad2_audit.py docs/evidence/ad2-immutable-inputs-v1.json 207f66a7c8e24de619e0034300844a90c5b1ed36bdfb7c97525773bd4dfda220 /home/tj330/programs/VLA-Project /home/tj330/programs/VLA-Project-analytic /home/tj330/.cache/Cyberbotics/Webots/assets runs/ad2-offline-audit-owner-check
```

Expected: `status=offline-source-audit-only`, `A_equals_B=true`, `geometry_safety=false`.
Missing/mismatched source, root, manifest or ambiguous cached PROTO stops the audit; no
input repair occurs. Existing v1/v2/v3 offline outputs are retained, not physical episodes.

## Remaining blockers and decisions

| Gate | Status |
|---|---|
| Immutable A/B/C and complete original input verification | PASS offline |
| Reset comparator and inert coordinator fixtures | PASS CPU only |
| Actual numerical reset protocol | **UNAPPROVED**: no production configuration shipped; missing criteria fail closed |
| Historic object twists/full base twist/effective force/PID | Still incomplete; accept named partial observability or keep NO-GO |
| Candidate Phase I revision | Rechecked SHA matches `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`; submitted-content authenticity still awaits owner confirmation |
| One measurement tick versus preparation | Must approve32ms post-reset interpretation and full preparation accounting |
| Exact-scene geometry/transforms/self/support exclusions/motion sweep/margin | Not established or approved; live execution NO-GO |
| Live non-mutating snapshot instrumentation/API behavior | **UNVERIFIED; no live adapter implemented** |
| Physical harness and live simulator work | Not authorized, not implemented or executed |

No policy, runtime controller, safety threshold or evaluation behavior was changed.
The frozen source/model/dataset hashes were verified from the existing seal. New modules
use only standard-library code; the offline tool uses already-declared NumPy/Typer. No
dependency, licence or public core interface changed.

**Next authorization boundary:** confirm the report revision and approve the numerical/
observability/tick protocol and safety preflight requirements. Any subsequent live simulator
instrumentation or A-D2 execution needs separate explicit authorization after remaining
proof obligations are met. This implementation does not grant or imply that authority.
