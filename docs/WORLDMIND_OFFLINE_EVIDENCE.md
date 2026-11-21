# WORLDMIND offline decision evidence v1

This format supports the scripted-primary `vla_world_model` evidence fixture under
ADR-0083. It carries the fixed authority label `OFFLINE_NON_SCIENTIFIC`. Neither a
valid archive nor a complete ranking record grants calibration or execution authority.
The scientific CLI and the comparison-arm policies have no dependency on this codec.

## Bundle and verification

`write_evidence(path, records)` validates before exclusively creating the destination.
`read_evidence(path, expected_digest=...)` verifies its version, authority label,
canonical bytes, envelope SHA-256, and every referenced payload before semantic
reconciliation. An expected digest supplied through independent provenance also pins
the whole bundle. A truncated or corrupt file is invalid. A valid failure record can
describe an incomplete recording using the fields that were actually captured.

The offline envelope uses the existing typed MessagePack codec. It keeps floating
point values, map iteration order, ordered sequences, tuples, bytes, and registered
core types. Canonical v1 bytes retain the captured map order. A new record built with
different map order need not have the same digest. Read and write use the same type
codec without changing simulator protocol v3. The shared decision input and candidate
payloads are hash-addressed within the bundle. The draw digest binds the original ID,
seed, and ordered chunk, and the rollout digest binds that draw to its ordered state
sequence. Capture them at their respective source boundaries. The digests verify
internal consistency. Authentication of the producer and external artifact identity
requires separately trusted provenance.

| Record | v1 contents |
|---|---|
| Decision index | Frozen episode and decision ID, script adapter, seed and draw list, source/config/plan/definitions hashes, actual policy label, binding references, failure reasons, timing and recovery. |
| Input payload | Full typed `WorldState` including `RobotState`, task, base-frame goal, timestamp, expected scene objects and scene completeness, and explicit frame bindings. `gripper_holding=None` is a preserved unknown. |
| Candidate row | ID, draw index and seed, capture digests, check results, disposition, rollout status, and optional finite objective components with binding digests. The objective inputs resolve through the shared decision state and the candidate action and prediction payload. |
| Candidate payload | Typed scripted `ActionChunk` with 50 ordered actions at 32 ms, and an ordered prediction sequence. Each complete rollout has 51 states and its initial state matches the decision input exactly. State index defines the prediction tick. |
| Ranking and selection | Selected ID or null and actual policy. A recorded ranking supplies parameter reference, direction, scores and order. Uncalibrated fixtures leave that record and absent components null. The validator checks consistency but does not compute scores or resolve external parameter artifacts. |
| Execution and pairing | Indexed actual action, independent motor verdict, apply and step outcomes, linked before and after states and timestamps, pairing status, selected candidate, valid steps and boundary. A named deviation may be recorded with non-comparable pairing. Unknown outcomes carry an explicit failure reason and cannot supply an observation. |
| Episode outcome | Same-episode outcome record with in-progress, complete or failed status, terminal classification when available, and digest reference. |

Expected scene identities attest the scope declared by the source. This codec
reconciles those identities against the included states. It does not certify physical
scene coverage or a simulator. Artifact hashes and binding references outside the
bundle are provenance claims for a later reviewer to verify against accepted artifacts.

`pool_complete` means 32 reconciled typed action sequences. It does not imply all
predictions are complete. `ranking_evidence_complete` additionally requires complete
rollouts, complete component and binding references, a reconciled ranking, and no
declared failure or unevaluable candidate. It describes record completeness only.
The deterministic fixture deliberately leaves numeric bindings and ranking unset.
Invalid input can be reported with an absent input and explicit reason. Corrupt,
contradictory, or unverifiable required fields fail closed. A stopped run cannot
supply invented actions, physics ticks, scores, or predicted states.

## ADR-0084 Table 5.1 mapping

The separate bridge checks pool IDs, order, seeds, and first-eligible selection. A
failed finite-chunk check is invalid. Missing or unknown required check results are
unevaluable. An explicit first-command rejection after a finite-chunk pass is a
hard veto. Passing candidates are selected or safe and unselected. The denominator
is hard-vetoed plus safe-unselected plus selected. Invalid and unevaluable counts
remain visible outside it. A zero denominator remains undefined. Execution-time
motor vetoes remain separate, and prediction error stays not applicable with null
value and zero valid paired steps.

## Boundaries

Historical recordings, frozen configurations and scientific definitions remain
unchanged. Future use requires separate review of source and numeric binding
provenance, the collection policy and safety gates, the calibration estimand,
and the approval manifest. No fixture archive qualifies as calibration evidence.
