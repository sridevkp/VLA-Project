"""Deterministic, VALIDATION-only calibration machinery for WorldMind.

The module records and verifies provenance; it never invents weights, scores, objectives or
calibration results. Candidate parameter values and the evaluation callable must come from an
owner-reviewed artifact and an explicit Python binding. Nothing in this module accesses TEST or
writes to frozen campaign evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CALIBRATION_SCHEMA_VERSION = 1
SPLIT = "validation"
SELECTION_RULE = "registered_evaluator_max_score_then_candidate_id"
PERMITTED_PARAMETER_KEYS = frozenset({"weights", "maximize"})
COMPONENT_NAMES = ("Jg", "Jl", "Jr", "Ju")
FORBIDDEN_EVIDENCE_PARTS = frozenset({"test", "tests", "train", "training"})
DRAFT_REVIEW_STATUS = "draft"
# The four non-circular stages. Only stage 3 is an owner decision; no stage invents a score.
CALIBRATION_WORKFLOW = (
    "1=uncalibrated diagnostic config (hard checks intact, no calibrated weights); "
    "2=VALIDATION-only evidence collection (18 frozen episodes, hash pinned); "
    "3=draft selection + owner approval of non-safety parameters; "
    "4=evaluation with the frozen approved parameters"
)
SELECTION_EVALUATION_CLASS = "selection_statistic_not_independent_evaluation"
CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "calibration_id",
        "split",
        "approved",
        "evidence_id",
        "evidence_path",
        "evidence_sha256",
        "approval_record",
        "component_bindings",
        "weights_source",
        "parameters",
        "candidates",
        "selection",
        "provenance",
    }
)
EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "split",
        "approved",
        "episode_ids",
        "source_hashes",
        "measurement_class",
    }
)
DRAFT_KEYS = frozenset(
    {
        "schema_version",
        "review_status",
        "split",
        "calibration_id",
        "evidence_id",
        "evidence_path",
        "evidence_sha256",
        "component_bindings",
        "weights_source",
        "candidates",
        "provenance",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_test_path(path: Path) -> None:
    parts = {part.lower() for part in path.parts}
    if parts & FORBIDDEN_EVIDENCE_PARTS:
        raise ValueError("calibration evidence path references a sealed TEST or TRAIN split")


def _validate_parameters(parameters: Any, name: str) -> dict[str, Any]:
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(f"{name} must be a nonempty mapping")
    unknown = set(parameters) - PERMITTED_PARAMETER_KEYS
    if unknown:
        raise ValueError(f"{name} contains unpermitted calibration fields: {sorted(unknown)}")
    weights = parameters.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != set(COMPONENT_NAMES):
        raise ValueError(f"{name}.weights must name exactly Jg, Jl, Jr and Ju")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in weights.values()
    ):
        raise ValueError(f"{name}.weights must be finite numbers")
    if any(float(value) < 0.0 for value in weights.values()) or not any(
        float(value) > 0.0 for value in weights.values()
    ):
        raise ValueError(f"{name}.weights must be nonnegative and not all zero")
    if not isinstance(parameters.get("maximize"), bool):
        raise ValueError(f"{name}.maximize must be an explicit boolean")
    return {
        "weights": {key: float(value) for key, value in weights.items()},
        "maximize": bool(parameters["maximize"]),
    }


@dataclass(frozen=True)
class CalibrationEvidence:
    """Frozen VALIDATION evidence identity; no campaign data is copied or modified."""

    path: Path
    sha256: str
    evidence_id: str
    episode_ids: tuple[str, ...]
    source_hashes: Mapping[str, str]
    measurement_class: str

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_path": str(self.path),
            "evidence_sha256": self.sha256,
            "episode_ids": list(self.episode_ids),
            "source_hashes": dict(self.source_hashes),
            "measurement_class": self.measurement_class,
            "split": SPLIT,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    """An approved VALIDATION-only candidate with fully declared provenance."""

    path: Path
    sha256: str
    calibration_id: str
    evidence: CalibrationEvidence
    approval_record: str
    component_bindings: Mapping[str, str]
    weights_source: str
    parameters: Mapping[str, Any]
    candidate_parameters: Mapping[str, Mapping[str, Any]]
    selection: Mapping[str, Any]
    provenance: Mapping[str, Any]
    selection_rule: str

    @property
    def provenance_record(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "candidate_path": str(self.path),
            "candidate_sha256": self.sha256,
            "approval_record": self.approval_record,
            "component_bindings": dict(self.component_bindings),
            "weights_source": self.weights_source,
            "selection_rule": self.selection_rule,
            "declared_selection": dict(self.selection),
            "parameters": dict(self.parameters),
            "evidence": self.evidence.provenance,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class CalibrationResult:
    """Deterministically evaluated candidate scores and the selected parameter set."""

    scores: Mapping[str, float]
    selected_candidate_id: str
    selected_parameters: Mapping[str, Any]
    score: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationDraft:
    """A reviewable, NOT approved parameter grid that can be selected before config activation.

    Unlike ``CalibrationCandidate`` a draft has no ``parameters``, ``selection`` or
    ``approval_record``: those are the outputs of stages 3 and 4, not inputs to them. Every
    other evidence, hash, split and parameter-grid gate is identical to the approved form.
    """

    path: Path
    sha256: str
    calibration_id: str
    evidence: CalibrationEvidence
    component_bindings: Mapping[str, str]
    weights_source: str
    candidate_parameters: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    selection_rule: str


@dataclass(frozen=True)
class CalibrationProposal:
    """Stage 3 output: a reproducible selection *statistic*, not an approval or an evaluation.

    The owner must still write an approved candidate artifact (with `parameters`, `selection`
    and `approval_record`) before a `ready` config may be loaded. This object never modifies a
    file and is explicitly labelled `selection_statistic_not_independent_evaluation`.
    """

    draft_path: Path
    draft_sha256: str
    calibration_id: str
    evidence_id: str
    evidence_sha256: str
    scores: Mapping[str, float]
    recommended_candidate_id: str
    recommended_parameters: Mapping[str, Any]
    selection_rule: str
    evaluation_class: str
    provenance: Mapping[str, Any]


CalibrationEvaluator = Callable[[Mapping[str, Any], CalibrationEvidence], float]


def load_calibration_evidence(path: Path) -> CalibrationEvidence:
    """Load hash-pinned, approved VALIDATION evidence without modifying it."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"calibration evidence not found: {resolved}")
    _reject_test_path(resolved)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("calibration evidence must be a mapping")
    missing = EVIDENCE_KEYS - set(data)
    extra = set(data) - EVIDENCE_KEYS
    if missing or extra:
        raise ValueError(
            f"calibration evidence has missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    if data["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration evidence schema_version")
    if data["split"] != SPLIT or data["approved"] is not True:
        raise ValueError("calibration evidence must be approved and VALIDATION-only")
    if not isinstance(data["evidence_id"], str) or not data["evidence_id"].strip():
        raise ValueError("calibration evidence requires an explicit evidence_id")
    episode_ids = data["episode_ids"]
    if (
        not isinstance(episode_ids, Sequence)
        or isinstance(episode_ids, str)
        or len(episode_ids) != 18
        or len(set(episode_ids)) != 18
        or not all(isinstance(value, str) and value for value in episode_ids)
    ):
        raise ValueError("calibration evidence must enumerate 18 unique VALIDATION episodes")
    hashes = data["source_hashes"]
    if (
        not isinstance(hashes, Mapping)
        or not hashes
        or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in hashes.items()
        )
    ):
        raise ValueError("calibration evidence requires explicit source hashes")
    if not isinstance(data["measurement_class"], str) or not data["measurement_class"].strip():
        raise ValueError("calibration evidence requires an explicit measurement class")
    return CalibrationEvidence(
        path=resolved,
        sha256=_sha256(resolved),
        evidence_id=data["evidence_id"],
        episode_ids=tuple(episode_ids),
        source_hashes=dict(hashes),
        measurement_class=data["measurement_class"],
    )


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def build_calibration_evidence(
    path: Path,
    *,
    evidence_id: str,
    episode_ids: Sequence[str],
    source_hashes: Mapping[str, str],
    measurement_class: str,
) -> CalibrationEvidence:
    """Stage 2 helper: record VALIDATION evidence identities supplied by the caller.

    This writes a new file exclusively; it never runs a campaign, reads TEST/TRAIN or computes a
    score. Every value is explicit: episode IDs and source hashes must come from a real, frozen
    VALIDATION collection, and the artifact stays unapproved until an owner signs it off. The
    returned identity is not usable by ``load_calibration_evidence`` until the owner sets the
    approval flag, keeping collection and approval separate.
    """
    resolved = path.expanduser().resolve()
    _reject_test_path(resolved)
    if resolved.exists():
        raise FileExistsError(f"calibration evidence already exists: {resolved}")
    if (
        not isinstance(episode_ids, Sequence)
        or isinstance(episode_ids, str)
        or len(episode_ids) != 18
        or len(set(episode_ids)) != 18
        or not all(isinstance(item, str) and item for item in episode_ids)
    ):
        raise ValueError("calibration evidence must enumerate 18 unique VALIDATION episodes")
    if (
        not isinstance(source_hashes, Mapping)
        or not source_hashes
        or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in source_hashes.items()
        )
    ):
        raise ValueError("calibration evidence requires explicit source hashes")
    identifier = _require_text(evidence_id, "evidence_id")
    measurement = _require_text(measurement_class, "measurement_class")
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "evidence_id": identifier,
        "split": SPLIT,
        "approved": False,
        "episode_ids": list(episode_ids),
        "source_hashes": dict(source_hashes),
        "measurement_class": measurement,
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return CalibrationEvidence(
        path=resolved,
        sha256=_sha256(resolved),
        evidence_id=identifier,
        episode_ids=tuple(episode_ids),
        source_hashes=dict(source_hashes),
        measurement_class=measurement,
    )


def load_calibration_draft(path: Path) -> CalibrationDraft:
    """Load a stage-3 reviewable draft; reject any approved-form field.

    The draft shares every evidence, hash, split and parameter-grid gate with the approved
    candidate. It deliberately has no ``parameters``, ``selection`` or ``approval_record`` so it
    cannot be mistaken for a signed-off result. Unapproved evidence is rejected here in the same
    way as in the approved path; the owner approves the evidence artifact as a separate decision.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"calibration draft not found: {resolved}")
    _reject_test_path(resolved)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("calibration draft must be a mapping")
    missing = DRAFT_KEYS - set(data)
    extra = set(data) - DRAFT_KEYS
    if missing or extra:
        raise ValueError(
            f"calibration draft has missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    if data["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration draft schema_version")
    if data["split"] != SPLIT:
        raise ValueError("calibration draft must be VALIDATION-only")
    if data["review_status"] != DRAFT_REVIEW_STATUS:
        raise ValueError("calibration draft review_status must be 'draft'; owner approval required")
    evidence_path = Path(str(data["evidence_path"])).expanduser()
    if not evidence_path.is_absolute():
        evidence_path = resolved.parent / evidence_path
    evidence = load_calibration_evidence(evidence_path)
    if data["evidence_id"] != evidence.evidence_id:
        raise ValueError("calibration draft evidence_id mismatch")
    if data["evidence_sha256"] != evidence.sha256:
        raise ValueError("frozen VALIDATION evidence was modified after draft creation")
    calibration_id = _require_text(data["calibration_id"], "calibration_id")
    component_bindings = data["component_bindings"]
    if (
        not isinstance(component_bindings, Mapping)
        or set(component_bindings) != set(COMPONENT_NAMES)
        or not all(
            isinstance(value, str) and value.strip() for value in component_bindings.values()
        )
    ):
        raise ValueError("calibration draft requires all four explicit component bindings")
    weights_source = _require_text(data["weights_source"], "weights_source")
    candidates = data["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, str) or not candidates:
        raise ValueError("calibration draft requires at least one parameter candidate")
    candidate_parameters: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        if not isinstance(entry, Mapping) or set(entry) != {"candidate_id", "parameters"}:
            raise ValueError("each draft candidate needs candidate_id and parameters only")
        name = _require_text(entry["candidate_id"], "candidate_id")
        if name in candidate_parameters:
            raise ValueError("calibration candidate ids must be unique strings")
        candidate_parameters[name] = _validate_parameters(entry["parameters"], name)
    provenance = data["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("calibration draft requires provenance")
    return CalibrationDraft(
        path=resolved,
        sha256=_sha256(resolved),
        calibration_id=calibration_id,
        evidence=evidence,
        component_bindings=dict(component_bindings),
        weights_source=weights_source,
        candidate_parameters=candidate_parameters,
        provenance=dict(provenance),
        selection_rule=SELECTION_RULE,
    )


def _rank_parameters(
    candidate_parameters: Mapping[str, Mapping[str, Any]],
    evaluator: CalibrationEvaluator,
    evidence: CalibrationEvidence,
) -> tuple[dict[str, float], str]:
    """Score every candidate with the injected evaluator and apply the declared selection rule.

    The rule is ``registered_evaluator_max_score_then_candidate_id``. This is a deterministic
    selection statistic on the *same* frozen VALIDATION evidence used for tuning; it is not an
    independent estimate of post-calibration performance.
    """
    if not callable(evaluator):
        raise ValueError("calibration ranking requires an explicit registered evaluator")
    scores: dict[str, float] = {}
    for name, parameters in candidate_parameters.items():
        value = float(evaluator(parameters, evidence))
        if not math.isfinite(value):
            raise ValueError(f"calibration score for {name} is not finite")
        scores[name] = value
    if not scores:
        raise ValueError("calibration ranking requires at least one candidate")
    best = min(scores, key=lambda name: (-scores[name], name))
    return scores, best


def rank_calibration_candidates(
    draft: CalibrationDraft, evaluator: CalibrationEvaluator
) -> CalibrationProposal:
    """Produce the stage-3 selection proposal; it does not approve or persist anything."""
    scores, best = _rank_parameters(draft.candidate_parameters, evaluator, draft.evidence)
    provenance = {
        **draft.provenance,
        "draft_path": str(draft.path),
        "draft_sha256": draft.sha256,
        "evidence": draft.evidence.provenance,
        "scores": dict(sorted(scores.items())),
        "recommended_candidate_id": best,
        "evaluation_class": SELECTION_EVALUATION_CLASS,
        "independent_evaluation": False,
        "test_split_accessed": False,
        "frozen_evidence_modified": False,
    }
    return CalibrationProposal(
        draft_path=draft.path,
        draft_sha256=draft.sha256,
        calibration_id=draft.calibration_id,
        evidence_id=draft.evidence.evidence_id,
        evidence_sha256=draft.evidence.sha256,
        scores=scores,
        recommended_candidate_id=best,
        recommended_parameters=dict(draft.candidate_parameters[best]),
        selection_rule=draft.selection_rule,
        evaluation_class=SELECTION_EVALUATION_CLASS,
        provenance=provenance,
    )


def load_calibration_candidate(path: Path) -> CalibrationCandidate:
    """Load and verify an approved candidate; reject missing, forbidden or stale fields."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"calibration candidate not found: {resolved}")
    _reject_test_path(resolved)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("calibration candidate must be a mapping")
    missing = CANDIDATE_KEYS - set(data)
    extra = set(data) - CANDIDATE_KEYS
    if missing or extra:
        raise ValueError(
            f"calibration candidate has missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    if data["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration candidate schema_version")
    if data["split"] != SPLIT or data["approved"] is not True:
        raise ValueError("calibration candidate must be approved and VALIDATION-only")
    evidence_path = Path(str(data["evidence_path"])).expanduser()
    if not evidence_path.is_absolute():
        evidence_path = resolved.parent / evidence_path
    evidence = load_calibration_evidence(evidence_path)
    if data["evidence_id"] != evidence.evidence_id:
        raise ValueError("calibration candidate evidence_id mismatch")
    if data["evidence_sha256"] != evidence.sha256:
        raise ValueError("frozen VALIDATION evidence was modified after candidate approval")
    candidate_id = data["calibration_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("calibration candidate requires an explicit calibration_id")
    approval = data["approval_record"]
    if not isinstance(approval, str) or not approval.strip():
        raise ValueError("calibration candidate requires an approval record")
    component_bindings = data["component_bindings"]
    if (
        not isinstance(component_bindings, Mapping)
        or set(component_bindings) != set(COMPONENT_NAMES)
        or not all(
            isinstance(value, str) and value.strip() for value in component_bindings.values()
        )
    ):
        raise ValueError("calibration candidate requires all four explicit component bindings")
    weights_source = data["weights_source"]
    if not isinstance(weights_source, str) or not weights_source.strip():
        raise ValueError("calibration candidate requires a weights source")
    parameters = _validate_parameters(data["parameters"], "calibration.parameters")
    candidates = data["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, str) or not candidates:
        raise ValueError("calibration candidate requires at least one parameter candidate")
    candidate_parameters: dict[str, dict[str, Any]] = {}
    for entry in candidates:
        if not isinstance(entry, Mapping) or set(entry) != {"candidate_id", "parameters"}:
            raise ValueError("each calibration candidate needs candidate_id and parameters only")
        name = entry["candidate_id"]
        if not isinstance(name, str) or not name.strip() or name in candidate_parameters:
            raise ValueError("calibration candidate ids must be unique strings")
        candidate_parameters[name] = _validate_parameters(entry["parameters"], name)
    selection = data["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "candidate_id",
        "rationale",
    }:
        raise ValueError("selection must declare candidate_id and rationale")
    if selection["candidate_id"] not in candidate_parameters:
        raise ValueError("selected calibration candidate is not in the candidate list")
    if not isinstance(selection["rationale"], str) or not selection["rationale"].strip():
        raise ValueError("selection rationale must be explicit")
    provenance = data["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("calibration candidate requires provenance")
    return CalibrationCandidate(
        path=resolved,
        sha256=_sha256(resolved),
        calibration_id=candidate_id,
        evidence=evidence,
        approval_record=approval,
        component_bindings=dict(component_bindings),
        weights_source=weights_source,
        parameters=parameters,
        candidate_parameters=candidate_parameters,
        selection=dict(selection),
        provenance=dict(provenance),
        selection_rule=SELECTION_RULE,
    )


def evaluate_calibration(
    candidate: CalibrationCandidate, evaluator: CalibrationEvaluator
) -> CalibrationResult:
    """Evaluate approved candidates with an injected rule and verify the recorded selection."""
    scores, best = _rank_parameters(candidate.candidate_parameters, evaluator, candidate.evidence)
    selected = str(candidate.selection["candidate_id"])
    if best != selected:
        raise ValueError("recorded calibration selection is not reproducible from the evidence")
    selected_parameters = candidate.candidate_parameters[selected]
    if selected_parameters != candidate.parameters:
        raise ValueError("selected parameters differ from the production parameters")
    provenance = {
        **candidate.provenance_record,
        "scores": dict(sorted(scores.items())),
        "selected_candidate_id": selected,
        "selected_score": scores[selected],
        "evaluated_candidate_count": len(scores),
        "evaluation_class": SELECTION_EVALUATION_CLASS,
        "independent_evaluation": False,
        "test_split_accessed": False,
        "frozen_evidence_modified": False,
    }
    return CalibrationResult(
        scores=scores,
        selected_candidate_id=selected,
        selected_parameters=selected_parameters,
        score=scores[selected],
        provenance=provenance,
    )
