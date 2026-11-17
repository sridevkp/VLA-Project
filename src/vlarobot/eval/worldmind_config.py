"""Fail-closed configuration contract for the production WorldMind composition.

The module loads owner-reviewed report definitions, approved VALIDATION calibration evidence
and explicit Python bindings. It never supplies a component formula, unit, weight, ranking
direction or risk threshold itself. The shipped ``configs/worldmind_production.yaml`` is a
blocked template: without reviewed values and registered bindings the loader raises before any
backend is constructed.

This module is inert. It does not import a simulator, launch a campaign, run inference, or
access the TEST split.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml

from vlarobot.eval.report_contract import (
    COMPONENTS,
    ReportDefinitions,
    load_definitions_candidate,
)
from vlarobot.eval.runner import EvalEpisode, build_episode_plan
from vlarobot.eval.worldmind_calibration import (
    CalibrationCandidate,
    load_calibration_candidate,
)

WORLD_MIND_SCHEMA_VERSION: Literal[1] = 1
AUTHORITATIVE_REPORT_SHA256 = "23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b"
FROZEN_SPLIT: Literal["validation"] = "validation"
FROZEN_BASE_SEED = 0
FROZEN_SEEDS: tuple[int, ...] = (0, 1, 2)
FROZEN_EPISODE_COUNT = 18
FROZEN_CANDIDATE_COUNT = 32
FROZEN_HORIZON_STEPS = 50
FROZEN_CONTROL_DT_S = 0.032
FROZEN_STATE_SOURCE: Literal["oracle"] = "oracle"
FROZEN_ARM: Literal["full_system"] = "full_system"
FROZEN_PROPOSER: Literal["scripted_chunk_adapter"] = "scripted_chunk_adapter"
GATE_CALL_POLICY = (
    "ComparisonPlanner.select calls RiskGate.select once for all candidates and once per "
    "candidate; K=32 therefore costs 33 gate calls per decision. No wrapper reduces or "
    "reorders those calls."
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "split",
        "base_seed",
        "seeds",
        "episodes",
        "backend",
        "task",
        "pipeline",
        "bounds",
        "report",
        "components",
        "assessment",
        "calibration",
        "surrogate",
        "safety",
        "verification",
        "adapter",
    }
)
CALIBRATION_KEYS = frozenset(
    {"evidence_id", "evidence_path", "candidate_path", "evaluator_binding"}
)
# Stage 1 of the non-circular calibration workflow. It is NOT a scored configuration: the only
# scientific omission is the not-yet-calibrated Eq. 3.8 weights/candidate; every hard constraint
# (frozen plan, oracle state, safety veto, verification, full arm and tick) is unchanged.
DIAGNOSTIC_STATUS = "diagnostic_uncalibrated"
DIAGNOSTIC_KEYS = frozenset(
    {
        "mode",
        "scored_evaluation_allowed",
        "calibration_stage",
        "evidence_collection_split",
    }
)
DIAGNOSTIC_TOP_LEVEL_KEYS = (TOP_LEVEL_KEYS - {"components", "calibration"}) | {"diagnostic"}
DIAGNOSTIC_ASSESSMENT_KEYS = frozenset({"required_constraints", "hard_checks"})


class TrajectoryPredictor(Protocol):
    def predict_trajectory(self, state: Any, chunk: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class BackendSpec:
    kind: Literal["fake", "webots"]
    declaration: str
    state_source: str


@dataclass(frozen=True)
class TaskSpecConfig:
    instruction: str
    target: str
    destination: str


@dataclass(frozen=True)
class PipelineSpec:
    proposer: str
    arm: str
    candidate_count: int
    horizon_steps: int
    control_dt_s: float
    decision_seed_base: int


@dataclass(frozen=True)
class BoundsSpec:
    max_decisions: int
    max_steps: int
    timings_enabled: bool


@dataclass(frozen=True)
class ReportSpec:
    pdf: Path
    definitions: Path
    candidate_definitions: Path
    sha256: str


@dataclass(frozen=True)
class ComponentSpec:
    binding: str
    unit: str


@dataclass(frozen=True)
class AssessmentSpec:
    weights: tuple[float, float, float, float]
    maximize: bool
    required_constraints: tuple[str, ...]
    hard_checks: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationSpec:
    """Approved VALIDATION-only candidate identity and selected parameters.

    The evaluator is an explicit binding in the Python registry; no score or objective is
    inferred from configuration text.
    """

    evidence_id: str
    evidence_path: Path
    calibration_id: str
    candidate_path: Path
    candidate_sha256: str
    evidence_sha256: str
    parameters: Mapping[str, Any]
    selected_candidate_id: str
    provenance: Mapping[str, Any]
    evaluator_binding: str
    candidate: CalibrationCandidate


@dataclass(frozen=True)
class SurrogateSpec:
    binding: str
    inputs: Mapping[str, str]


@dataclass(frozen=True)
class SafetySpec:
    binding: str
    config_path: Path
    inputs: Mapping[str, str]


@dataclass(frozen=True)
class VerificationSpec:
    binding: str
    destination_world: tuple[float, float, float]
    rest_center_world: tuple[float, float, float]
    object_dimensions_m: tuple[float, float, float]
    up_axis_base: tuple[float, float, float]
    distance_tolerance_m: float
    height_tolerance_m: float
    tilt_limit_rad: float
    home_tolerance_rad: float
    home_joints: Mapping[str, float]


@dataclass(frozen=True)
class JitterSpec:
    joint_std_rad: float
    joint_limit_rad: float
    max_attempts: int


@dataclass(frozen=True)
class AdapterSpec:
    waypoints_binding: str
    alternate_waypoints_binding: str | None
    pose_solver_binding: str
    ik_position_tolerance_m: float
    ik_orientation_tolerance_rad: float
    jitter: JitterSpec


@dataclass(frozen=True)
class WorldMindConfig:
    """A fully validated production plan; all scientific inputs are external and explicit."""

    schema_version: int
    status: str
    split: str
    base_seed: int
    seeds: tuple[int, ...]
    episodes: int
    backend: BackendSpec
    task: TaskSpecConfig
    pipeline: PipelineSpec
    bounds: BoundsSpec
    report: ReportSpec
    report_definitions: ReportDefinitions
    components: Mapping[str, ComponentSpec]
    assessment: AssessmentSpec
    calibration: CalibrationSpec
    surrogate: SurrogateSpec
    safety: SafetySpec
    verification: VerificationSpec
    adapter: AdapterSpec
    source_path: Path


@dataclass(frozen=True)
class WorldMindBindings:
    """Reviewed Python bindings; the shipped CLI provides an empty registry.

    A key in a registry is an explicit owner-reviewed identity, never a dynamically imported
    name from configuration. Missing keys fail closed before a backend is constructed.
    """

    components: Mapping[str, Any] = field(default_factory=dict)
    hard_checks: Mapping[str, Any] = field(default_factory=dict)
    surrogates: Mapping[str, Any] = field(default_factory=dict)
    safety: Mapping[str, Any] = field(default_factory=dict)
    verification: Mapping[str, Any] = field(default_factory=dict)
    pose_solvers: Mapping[str, Any] = field(default_factory=dict)
    waypoint_sets: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    waypoint_factories: Mapping[str, Any] = field(default_factory=dict)
    calibration_evaluators: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticSpec:
    """Explicit declaration that a config is uncalibrated and cannot produce scored evidence."""

    mode: str
    scored_evaluation_allowed: bool
    calibration_stage: int
    evidence_collection_split: str


@dataclass(frozen=True)
class DiagnosticAssessmentSpec:
    """Hard constraints only; Eq. 3.8 weights and ranking direction are not yet calibrated."""

    required_constraints: tuple[str, ...]
    hard_checks: tuple[str, ...]


@dataclass(frozen=True)
class WorldMindDiagnosticConfig:
    """Stage-1 uncalibrated config: hard constraints intact, no Eq. 3.8 parameters.

    This object is accepted only by the diagnostic preflight/evidence-scaffolding path. It cannot
    be handed to the scored WorldMind pipeline, which requires a `ready` `WorldMindConfig`
    carrying an approved, frozen calibration candidate.
    """

    schema_version: int
    status: str
    split: str
    base_seed: int
    seeds: tuple[int, ...]
    episodes: int
    backend: BackendSpec
    task: TaskSpecConfig
    pipeline: PipelineSpec
    bounds: BoundsSpec
    report: ReportSpec
    report_definitions: ReportDefinitions
    diagnostic: DiagnosticSpec
    assessment: DiagnosticAssessmentSpec
    surrogate: SurrogateSpec
    safety: SafetySpec
    verification: VerificationSpec
    adapter: AdapterSpec
    source_path: Path


def empty_worldmind_bindings() -> WorldMindBindings:
    """Return the default, intentionally empty production binding registry."""
    return WorldMindBindings()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(mapping: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"{name} has missing={missing} and unexpected={extra} keys")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("<")


def _binding(value: Any, name: str) -> str:
    text = _text(value, name)
    if _placeholder(text):
        raise ValueError(f"{name} is still a blocked template placeholder")
    return text


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _tuple(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return tuple(_float(item, f"{name}[{index}]") for index, item in enumerate(value))


def _path(value: Any, name: str, base: Path) -> Path:
    text = _binding(value, name)
    candidate = Path(text).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _existing_path(value: Any, name: str, base: Path) -> Path:
    path = _path(value, name, base)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _input_map(value: Any, name: str) -> Mapping[str, str]:
    raw = _mapping(value, name)
    if not raw:
        raise ValueError(f"{name} must name at least one explicit input binding")
    return {_text(key, f"{name} key"): _binding(item, f"{name}.{key}") for key, item in raw.items()}


def _joint_map(value: Any, name: str) -> Mapping[str, float]:
    raw = _mapping(value, name)
    if not raw:
        raise ValueError(f"{name} must contain at least one joint")
    return {_text(key, f"{name} key"): _float(item, f"{name}.{key}") for key, item in raw.items()}


def load_worldmind_config(path: Path) -> WorldMindConfig:
    """Load and validate a reviewed WorldMind config; fail closed on every missing input."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"worldmind config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "worldmind config")
    _keys(root, TOP_LEVEL_KEYS, "worldmind config")
    if root["schema_version"] != WORLD_MIND_SCHEMA_VERSION:
        raise ValueError("unsupported worldmind config schema_version")
    if root["status"] != "ready":
        raise ValueError("worldmind production config is blocked; reviewed status must be 'ready'")
    if root["split"] != FROZEN_SPLIT:
        raise ValueError("WorldMind production is VALIDATION-only")
    base_seed = _int(root["base_seed"], "base_seed")
    if base_seed != FROZEN_BASE_SEED:
        raise ValueError(f"frozen base seed {FROZEN_BASE_SEED} required")
    seeds = tuple(_int(value, "seeds") for value in _sequence(root["seeds"], "seeds"))
    if seeds != FROZEN_SEEDS:
        raise ValueError(f"frozen seed contract {FROZEN_SEEDS} required")
    episodes = _int(root["episodes"], "episodes")
    if episodes != FROZEN_EPISODE_COUNT:
        raise ValueError(f"frozen VALIDATION episode count {FROZEN_EPISODE_COUNT} required")

    backend_raw = _mapping(root["backend"], "backend")
    _keys(backend_raw, frozenset({"kind", "declaration", "state_source"}), "backend")
    kind = _text(backend_raw["kind"], "backend.kind")
    if kind not in ("fake", "webots"):
        raise ValueError("backend.kind must be 'fake' or 'webots'")
    state_source = _text(backend_raw["state_source"], "backend.state_source")
    if state_source != FROZEN_STATE_SOURCE:
        raise ValueError("WorldMind production requires declared oracle state source")
    backend = BackendSpec(
        kind=kind,  # type: ignore[arg-type]
        declaration=_binding(backend_raw["declaration"], "backend.declaration"),
        state_source=state_source,
    )

    task_raw = _mapping(root["task"], "task")
    _keys(task_raw, frozenset({"instruction", "target", "destination"}), "task")
    task = TaskSpecConfig(
        instruction=_binding(task_raw["instruction"], "task.instruction"),
        target=_binding(task_raw["target"], "task.target"),
        destination=_binding(task_raw["destination"], "task.destination"),
    )

    pipeline_raw = _mapping(root["pipeline"], "pipeline")
    _keys(
        pipeline_raw,
        frozenset(
            {
                "proposer",
                "arm",
                "candidate_count",
                "horizon_steps",
                "control_dt_s",
                "decision_seed_base",
            }
        ),
        "pipeline",
    )
    if _text(pipeline_raw["proposer"], "pipeline.proposer") != FROZEN_PROPOSER:
        raise ValueError(f"only the reviewed {FROZEN_PROPOSER} is allowed")
    if _text(pipeline_raw["arm"], "pipeline.arm") != FROZEN_ARM:
        raise ValueError("WorldMind production requires the full_system arm")
    candidate_count = _int(pipeline_raw["candidate_count"], "pipeline.candidate_count")
    horizon_steps = _int(pipeline_raw["horizon_steps"], "pipeline.horizon_steps")
    control_dt_s = _float(pipeline_raw["control_dt_s"], "pipeline.control_dt_s", positive=True)
    decision_seed_base = _int(pipeline_raw["decision_seed_base"], "pipeline.decision_seed_base")
    if candidate_count != FROZEN_CANDIDATE_COUNT:
        raise ValueError(f"frozen candidate count {FROZEN_CANDIDATE_COUNT} required")
    if horizon_steps != FROZEN_HORIZON_STEPS:
        raise ValueError(f"frozen horizon {FROZEN_HORIZON_STEPS} required")
    if abs(control_dt_s - FROZEN_CONTROL_DT_S) > 1e-12:
        raise ValueError(f"frozen control tick {FROZEN_CONTROL_DT_S} required")
    if decision_seed_base != FROZEN_BASE_SEED:
        raise ValueError(f"frozen decision seed base {FROZEN_BASE_SEED} required")
    pipeline = PipelineSpec(
        proposer=FROZEN_PROPOSER,
        arm=FROZEN_ARM,
        candidate_count=candidate_count,
        horizon_steps=horizon_steps,
        control_dt_s=control_dt_s,
        decision_seed_base=decision_seed_base,
    )

    bounds_raw = _mapping(root["bounds"], "bounds")
    _keys(bounds_raw, frozenset({"max_decisions", "max_steps", "timings_enabled"}), "bounds")
    bounds = BoundsSpec(
        max_decisions=_int(bounds_raw["max_decisions"], "bounds.max_decisions", minimum=1),
        max_steps=_int(bounds_raw["max_steps"], "bounds.max_steps", minimum=1),
        timings_enabled=_bool(bounds_raw["timings_enabled"], "bounds.timings_enabled"),
    )

    report_raw = _mapping(root["report"], "report")
    _keys(
        report_raw,
        frozenset({"pdf", "definitions", "candidate_definitions", "sha256"}),
        "report",
    )
    sha256 = _text(report_raw["sha256"], "report.sha256").lower()
    if sha256 != AUTHORITATIVE_REPORT_SHA256:
        raise ValueError("report SHA-256 does not match the authoritative Phase I report")
    pdf = _existing_path(report_raw["pdf"], "report.pdf", source.parent)
    definitions_path = _existing_path(
        report_raw["definitions"], "report.definitions", source.parent
    )
    candidate_definitions = _existing_path(
        report_raw["candidate_definitions"], "report.candidate_definitions", source.parent
    )
    load_definitions_candidate(candidate_definitions, pdf)
    report_definitions = ReportDefinitions.load(definitions_path, pdf)
    if report_definitions.report_sha256 != AUTHORITATIVE_REPORT_SHA256:
        raise ValueError("definitions do not bind the authoritative report bytes")
    report = ReportSpec(
        pdf=pdf,
        definitions=definitions_path,
        candidate_definitions=candidate_definitions,
        sha256=sha256,
    )

    components_raw = _mapping(root["components"], "components")
    if set(components_raw) != set(COMPONENTS):
        raise ValueError("exactly four report component bindings required")
    components: dict[str, ComponentSpec] = {}
    for name in COMPONENTS:
        entry = _mapping(components_raw[name], f"components.{name}")
        _keys(entry, frozenset({"binding", "unit"}), f"components.{name}")
        components[name] = ComponentSpec(
            binding=_binding(entry["binding"], f"components.{name}.binding"),
            unit=_binding(entry["unit"], f"components.{name}.unit"),
        )

    assessment_raw = _mapping(root["assessment"], "assessment")
    _keys(
        assessment_raw,
        frozenset({"weights", "maximize", "required_constraints", "hard_checks"}),
        "assessment",
    )
    weights_raw = _mapping(assessment_raw["weights"], "assessment.weights")
    if set(weights_raw) != set(COMPONENTS):
        raise ValueError("all four Eq. 3.8 weights must be explicitly present")
    weights = cast(
        tuple[float, float, float, float],
        tuple(_float(weights_raw[name], f"assessment.weights.{name}") for name in COMPONENTS),
    )
    if any(weight < 0.0 for weight in weights) or not any(weight > 0.0 for weight in weights):
        raise ValueError("Eq. 3.8 weights must be nonnegative and not all zero")
    constraints = tuple(
        _binding(value, "assessment.required_constraints")
        for value in _sequence(assessment_raw["required_constraints"], "assessment.constraints")
    )
    hard_checks = tuple(
        _binding(value, "assessment.hard_checks")
        for value in _sequence(assessment_raw["hard_checks"], "assessment.hard_checks")
    )
    if not constraints or len(set(constraints)) != len(constraints):
        raise ValueError("unique required hard constraints required")
    if not hard_checks or len(set(hard_checks)) != len(hard_checks):
        raise ValueError("unique hard-check bindings required")
    if any(name not in hard_checks for name in constraints):
        raise ValueError("every required constraint must name an explicit hard-check binding")
    assessment = AssessmentSpec(
        weights=weights,
        maximize=_bool(assessment_raw["maximize"], "assessment.maximize"),
        required_constraints=constraints,
        hard_checks=hard_checks,
    )

    calibration_raw = _mapping(root["calibration"], "calibration")
    _keys(calibration_raw, CALIBRATION_KEYS, "calibration")
    evidence_id = _binding(calibration_raw["evidence_id"], "calibration.evidence_id")
    evidence_path = _existing_path(
        calibration_raw["evidence_path"], "calibration.evidence_path", source.parent
    )
    candidate_path = _existing_path(
        calibration_raw["candidate_path"], "calibration.candidate_path", source.parent
    )
    evaluator_binding = _binding(
        calibration_raw["evaluator_binding"], "calibration.evaluator_binding"
    )
    candidate = load_calibration_candidate(candidate_path)
    if candidate.evidence.evidence_id != evidence_id:
        raise ValueError("calibration evidence ID binding mismatch")
    if candidate.evidence.path != evidence_path:
        raise ValueError("calibration evidence path binding mismatch")
    source_text = candidate.provenance.get("source")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("calibration candidate provenance requires an explicit source")
    if tuple(candidate.parameters["weights"][name] for name in COMPONENTS) != weights:
        raise ValueError("calibration parameters differ from the Eq. 3.8 assessment weights")
    if candidate.parameters["maximize"] != assessment.maximize:
        raise ValueError("calibration parameters differ from the assessment ranking direction")
    calibration = CalibrationSpec(
        evidence_id=evidence_id,
        evidence_path=evidence_path,
        calibration_id=candidate.calibration_id,
        candidate_path=candidate_path,
        candidate_sha256=candidate.sha256,
        evidence_sha256=candidate.evidence.sha256,
        parameters=candidate.parameters,
        selected_candidate_id=str(candidate.selection["candidate_id"]),
        provenance=candidate.provenance,
        evaluator_binding=evaluator_binding,
        candidate=candidate,
    )

    surrogate_raw = _mapping(root["surrogate"], "surrogate")
    _keys(surrogate_raw, frozenset({"binding", "inputs"}), "surrogate")
    surrogate = SurrogateSpec(
        binding=_binding(surrogate_raw["binding"], "surrogate.binding"),
        inputs=_input_map(surrogate_raw["inputs"], "surrogate.inputs"),
    )

    safety_raw = _mapping(root["safety"], "safety")
    _keys(safety_raw, frozenset({"binding", "config_path", "inputs"}), "safety")
    safety = SafetySpec(
        binding=_binding(safety_raw["binding"], "safety.binding"),
        config_path=_existing_path(safety_raw["config_path"], "safety.config_path", source.parent),
        inputs=_input_map(safety_raw["inputs"], "safety.inputs"),
    )

    verification_raw = _mapping(root["verification"], "verification")
    _keys(
        verification_raw,
        frozenset(
            {
                "binding",
                "destination_world",
                "rest_center_world",
                "object_dimensions_m",
                "up_axis_base",
                "distance_tolerance_m",
                "height_tolerance_m",
                "tilt_limit_rad",
                "home_tolerance_rad",
                "home_joints",
            }
        ),
        "verification",
    )
    destination = cast(
        tuple[float, float, float],
        _tuple(verification_raw["destination_world"], "verification.destination_world", 3),
    )
    rest_center = cast(
        tuple[float, float, float],
        _tuple(verification_raw["rest_center_world"], "verification.rest_center_world", 3),
    )
    dimensions = cast(
        tuple[float, float, float],
        _tuple(verification_raw["object_dimensions_m"], "verification.object_dimensions_m", 3),
    )
    if any(value <= 0.0 for value in dimensions):
        raise ValueError("verification.object_dimensions_m must all be positive")
    verification = VerificationSpec(
        binding=_binding(verification_raw["binding"], "verification.binding"),
        destination_world=destination,
        rest_center_world=rest_center,
        object_dimensions_m=dimensions,
        up_axis_base=cast(
            tuple[float, float, float],
            _tuple(verification_raw["up_axis_base"], "verification.up_axis_base", 3),
        ),
        distance_tolerance_m=_float(
            verification_raw["distance_tolerance_m"],
            "verification.distance_tolerance_m",
            positive=True,
        ),
        height_tolerance_m=_float(
            verification_raw["height_tolerance_m"], "verification.height_tolerance_m", positive=True
        ),
        tilt_limit_rad=_float(
            verification_raw["tilt_limit_rad"], "verification.tilt_limit_rad", positive=True
        ),
        home_tolerance_rad=_float(
            verification_raw["home_tolerance_rad"], "verification.home_tolerance_rad", positive=True
        ),
        home_joints=_joint_map(verification_raw["home_joints"], "verification.home_joints"),
    )

    adapter_raw = _mapping(root["adapter"], "adapter")
    _keys(
        adapter_raw,
        frozenset(
            {
                "waypoints_binding",
                "alternate_waypoints_binding",
                "pose_solver_binding",
                "ik_position_tolerance_m",
                "ik_orientation_tolerance_rad",
                "jitter",
            }
        ),
        "adapter",
    )
    jitter_raw = _mapping(adapter_raw["jitter"], "adapter.jitter")
    _keys(
        jitter_raw,
        frozenset({"joint_std_rad", "joint_limit_rad", "max_attempts"}),
        "adapter.jitter",
    )
    jitter = JitterSpec(
        joint_std_rad=_float(jitter_raw["joint_std_rad"], "adapter.jitter.joint_std_rad"),
        joint_limit_rad=_float(jitter_raw["joint_limit_rad"], "adapter.jitter.joint_limit_rad"),
        max_attempts=_int(jitter_raw["max_attempts"], "adapter.jitter.max_attempts", minimum=1),
    )
    alternate = adapter_raw["alternate_waypoints_binding"]
    if alternate is not None:
        alternate = _binding(alternate, "adapter.alternate_waypoints_binding")
    adapter = AdapterSpec(
        waypoints_binding=_binding(adapter_raw["waypoints_binding"], "adapter.waypoints_binding"),
        alternate_waypoints_binding=alternate,
        pose_solver_binding=_binding(
            adapter_raw["pose_solver_binding"], "adapter.pose_solver_binding"
        ),
        ik_position_tolerance_m=_float(
            adapter_raw["ik_position_tolerance_m"],
            "adapter.ik_position_tolerance_m",
            positive=True,
        ),
        ik_orientation_tolerance_rad=_float(
            adapter_raw["ik_orientation_tolerance_rad"],
            "adapter.ik_orientation_tolerance_rad",
            positive=True,
        ),
        jitter=jitter,
    )

    return WorldMindConfig(
        schema_version=root["schema_version"],
        status=root["status"],
        split=root["split"],
        base_seed=base_seed,
        seeds=seeds,
        episodes=episodes,
        backend=backend,
        task=task,
        pipeline=pipeline,
        bounds=bounds,
        report=report,
        report_definitions=report_definitions,
        components=components,
        assessment=assessment,
        calibration=calibration,
        surrogate=surrogate,
        safety=safety,
        verification=verification,
        adapter=adapter,
        source_path=source,
    )


def _load_diagnostic_report(root: Mapping[str, Any], source: Path) -> tuple[ReportSpec, Any]:
    """Reuse every report gate of the production path without weakening the report SHA."""
    report_raw = _mapping(root["report"], "report")
    _keys(
        report_raw,
        frozenset({"pdf", "definitions", "candidate_definitions", "sha256"}),
        "report",
    )
    sha256 = _text(report_raw["sha256"], "report.sha256").lower()
    if sha256 != AUTHORITATIVE_REPORT_SHA256:
        raise ValueError("report SHA-256 does not match the authoritative Phase I report")
    pdf = _existing_path(report_raw["pdf"], "report.pdf", source.parent)
    definitions_path = _existing_path(
        report_raw["definitions"], "report.definitions", source.parent
    )
    candidate_definitions = _existing_path(
        report_raw["candidate_definitions"], "report.candidate_definitions", source.parent
    )
    load_definitions_candidate(candidate_definitions, pdf)
    definitions = ReportDefinitions.load(definitions_path, pdf)
    if definitions.report_sha256 != AUTHORITATIVE_REPORT_SHA256:
        raise ValueError("definitions do not bind the authoritative report bytes")
    return ReportSpec(pdf, definitions_path, candidate_definitions, sha256), definitions


def load_worldmind_diagnostic_config(path: Path) -> WorldMindDiagnosticConfig:
    """Load stage 1 of the calibration bootstrap: uncalibrated, hard-constrained, non-scored.

    Unlike :func:`load_worldmind_config` this accepts only ``status: diagnostic_uncalibrated``
    and has no Eq. 3.8 components, weights, calibration candidate or evaluator. Every other gate
    — frozen split/seeds/18 episodes, K=32, H=50, tick, oracle state, full arm, report bytes and
    reviewed definitions — is identical to the production path.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"worldmind diagnostic config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "worldmind diagnostic config")
    _keys(root, DIAGNOSTIC_TOP_LEVEL_KEYS, "worldmind diagnostic config")
    if root["schema_version"] != WORLD_MIND_SCHEMA_VERSION:
        raise ValueError("unsupported worldmind config schema_version")
    if root["status"] != DIAGNOSTIC_STATUS:
        raise ValueError(f"diagnostic config status must be '{DIAGNOSTIC_STATUS}'")
    if root["split"] != FROZEN_SPLIT:
        raise ValueError("WorldMind diagnostics are VALIDATION-only")
    if _int(root["base_seed"], "base_seed") != FROZEN_BASE_SEED:
        raise ValueError(f"frozen base seed {FROZEN_BASE_SEED} required")
    seeds = tuple(_int(value, "seeds") for value in _sequence(root["seeds"], "seeds"))
    if seeds != FROZEN_SEEDS:
        raise ValueError(f"frozen seed contract {FROZEN_SEEDS} required")
    if _int(root["episodes"], "episodes") != FROZEN_EPISODE_COUNT:
        raise ValueError(f"frozen VALIDATION episode count {FROZEN_EPISODE_COUNT} required")

    backend_raw = _mapping(root["backend"], "backend")
    _keys(backend_raw, frozenset({"kind", "declaration", "state_source"}), "backend")
    kind = _text(backend_raw["kind"], "backend.kind")
    if kind not in ("fake", "webots"):
        raise ValueError("backend.kind must be 'fake' or 'webots'")
    if _text(backend_raw["state_source"], "backend.state_source") != FROZEN_STATE_SOURCE:
        raise ValueError("WorldMind diagnostics require declared oracle state source")
    backend = BackendSpec(
        kind=kind,  # type: ignore[arg-type]
        declaration=_binding(backend_raw["declaration"], "backend.declaration"),
        state_source=FROZEN_STATE_SOURCE,
    )

    task_raw = _mapping(root["task"], "task")
    _keys(task_raw, frozenset({"instruction", "target", "destination"}), "task")
    task = TaskSpecConfig(
        instruction=_binding(task_raw["instruction"], "task.instruction"),
        target=_binding(task_raw["target"], "task.target"),
        destination=_binding(task_raw["destination"], "task.destination"),
    )

    pipeline_raw = _mapping(root["pipeline"], "pipeline")
    _keys(
        pipeline_raw,
        frozenset(
            {
                "proposer",
                "arm",
                "candidate_count",
                "horizon_steps",
                "control_dt_s",
                "decision_seed_base",
            }
        ),
        "pipeline",
    )
    if _text(pipeline_raw["proposer"], "pipeline.proposer") != FROZEN_PROPOSER:
        raise ValueError(f"only the reviewed {FROZEN_PROPOSER} is allowed")
    if _text(pipeline_raw["arm"], "pipeline.arm") != FROZEN_ARM:
        raise ValueError("WorldMind diagnostics require the full_system arm")
    if _int(pipeline_raw["candidate_count"], "pipeline.candidate_count") != FROZEN_CANDIDATE_COUNT:
        raise ValueError(f"frozen candidate count {FROZEN_CANDIDATE_COUNT} required")
    if _int(pipeline_raw["horizon_steps"], "pipeline.horizon_steps") != FROZEN_HORIZON_STEPS:
        raise ValueError(f"frozen horizon {FROZEN_HORIZON_STEPS} required")
    dt = _float(pipeline_raw["control_dt_s"], "pipeline.control_dt_s", positive=True)
    if abs(dt - FROZEN_CONTROL_DT_S) > 1e-12:
        raise ValueError(f"frozen control tick {FROZEN_CONTROL_DT_S} required")
    if _int(pipeline_raw["decision_seed_base"], "pipeline.decision_seed_base") != FROZEN_BASE_SEED:
        raise ValueError(f"frozen decision seed base {FROZEN_BASE_SEED} required")
    pipeline = PipelineSpec(
        FROZEN_PROPOSER, FROZEN_ARM, FROZEN_CANDIDATE_COUNT, FROZEN_HORIZON_STEPS, dt, 0
    )

    bounds_raw = _mapping(root["bounds"], "bounds")
    _keys(bounds_raw, frozenset({"max_decisions", "max_steps", "timings_enabled"}), "bounds")
    bounds = BoundsSpec(
        max_decisions=_int(bounds_raw["max_decisions"], "bounds.max_decisions", minimum=1),
        max_steps=_int(bounds_raw["max_steps"], "bounds.max_steps", minimum=1),
        timings_enabled=_bool(bounds_raw["timings_enabled"], "bounds.timings_enabled"),
    )

    report, definitions = _load_diagnostic_report(root, source)

    assessment_raw = _mapping(root["assessment"], "assessment")
    _keys(assessment_raw, DIAGNOSTIC_ASSESSMENT_KEYS, "assessment")
    constraints = tuple(
        _binding(value, "assessment.required_constraints")
        for value in _sequence(assessment_raw["required_constraints"], "assessment.constraints")
    )
    hard_checks = tuple(
        _binding(value, "assessment.hard_checks")
        for value in _sequence(assessment_raw["hard_checks"], "assessment.hard_checks")
    )
    if not constraints or len(set(constraints)) != len(constraints):
        raise ValueError("unique required hard constraints required")
    if not hard_checks or len(set(hard_checks)) != len(hard_checks):
        raise ValueError("unique hard-check bindings required")
    if any(name not in hard_checks for name in constraints):
        raise ValueError("every required constraint must name an explicit hard-check binding")
    assessment = DiagnosticAssessmentSpec(constraints, hard_checks)

    diagnostic_raw = _mapping(root["diagnostic"], "diagnostic")
    _keys(diagnostic_raw, DIAGNOSTIC_KEYS, "diagnostic")
    if diagnostic_raw["mode"] != "evidence_collection_only":
        raise ValueError("diagnostic mode must be 'evidence_collection_only'")
    if diagnostic_raw["scored_evaluation_allowed"] is not False:
        raise ValueError("an uncalibrated diagnostic config can never allow scored evaluation")
    if diagnostic_raw["calibration_stage"] != 1:
        raise ValueError("diagnostic config belongs to calibration stage 1")
    if diagnostic_raw["evidence_collection_split"] != FROZEN_SPLIT:
        raise ValueError("diagnostic evidence collection is VALIDATION-only")
    diagnostic = DiagnosticSpec(
        mode="evidence_collection_only",
        scored_evaluation_allowed=False,
        calibration_stage=1,
        evidence_collection_split=FROZEN_SPLIT,
    )

    surrogate_raw = _mapping(root["surrogate"], "surrogate")
    _keys(surrogate_raw, frozenset({"binding", "inputs"}), "surrogate")
    surrogate = SurrogateSpec(
        binding=_binding(surrogate_raw["binding"], "surrogate.binding"),
        inputs=_input_map(surrogate_raw["inputs"], "surrogate.inputs"),
    )
    safety_raw = _mapping(root["safety"], "safety")
    _keys(safety_raw, frozenset({"binding", "config_path", "inputs"}), "safety")
    safety = SafetySpec(
        binding=_binding(safety_raw["binding"], "safety.binding"),
        config_path=_existing_path(safety_raw["config_path"], "safety.config_path", source.parent),
        inputs=_input_map(safety_raw["inputs"], "safety.inputs"),
    )

    verification_raw = _mapping(root["verification"], "verification")
    _keys(
        verification_raw,
        frozenset(
            {
                "binding",
                "destination_world",
                "rest_center_world",
                "object_dimensions_m",
                "up_axis_base",
                "distance_tolerance_m",
                "height_tolerance_m",
                "tilt_limit_rad",
                "home_tolerance_rad",
                "home_joints",
            }
        ),
        "verification",
    )
    diagnostic_dimensions = cast(
        tuple[float, float, float],
        _tuple(verification_raw["object_dimensions_m"], "verification.object_dimensions_m", 3),
    )
    if any(value <= 0.0 for value in diagnostic_dimensions):
        raise ValueError("verification.object_dimensions_m must all be positive")
    verification = VerificationSpec(
        binding=_binding(verification_raw["binding"], "verification.binding"),
        destination_world=cast(
            tuple[float, float, float],
            _tuple(verification_raw["destination_world"], "verification.destination_world", 3),
        ),
        rest_center_world=cast(
            tuple[float, float, float],
            _tuple(verification_raw["rest_center_world"], "verification.rest_center_world", 3),
        ),
        object_dimensions_m=diagnostic_dimensions,
        up_axis_base=cast(
            tuple[float, float, float],
            _tuple(verification_raw["up_axis_base"], "verification.up_axis_base", 3),
        ),
        distance_tolerance_m=_float(
            verification_raw["distance_tolerance_m"],
            "verification.distance_tolerance_m",
            positive=True,
        ),
        height_tolerance_m=_float(
            verification_raw["height_tolerance_m"],
            "verification.height_tolerance_m",
            positive=True,
        ),
        tilt_limit_rad=_float(
            verification_raw["tilt_limit_rad"], "verification.tilt_limit_rad", positive=True
        ),
        home_tolerance_rad=_float(
            verification_raw["home_tolerance_rad"], "verification.home_tolerance_rad", positive=True
        ),
        home_joints=_joint_map(verification_raw["home_joints"], "verification.home_joints"),
    )

    adapter_raw = _mapping(root["adapter"], "adapter")
    _keys(
        adapter_raw,
        frozenset(
            {
                "waypoints_binding",
                "alternate_waypoints_binding",
                "pose_solver_binding",
                "ik_position_tolerance_m",
                "ik_orientation_tolerance_rad",
                "jitter",
            }
        ),
        "adapter",
    )
    jitter_raw = _mapping(adapter_raw["jitter"], "adapter.jitter")
    _keys(
        jitter_raw,
        frozenset({"joint_std_rad", "joint_limit_rad", "max_attempts"}),
        "adapter.jitter",
    )
    jitter = JitterSpec(
        joint_std_rad=_float(jitter_raw["joint_std_rad"], "adapter.jitter.joint_std_rad"),
        joint_limit_rad=_float(jitter_raw["joint_limit_rad"], "adapter.jitter.joint_limit_rad"),
        max_attempts=_int(jitter_raw["max_attempts"], "adapter.jitter.max_attempts", minimum=1),
    )
    alternate = adapter_raw["alternate_waypoints_binding"]
    if alternate is not None:
        alternate = _binding(alternate, "adapter.alternate_waypoints_binding")
    adapter = AdapterSpec(
        waypoints_binding=_binding(adapter_raw["waypoints_binding"], "adapter.waypoints_binding"),
        alternate_waypoints_binding=alternate,
        pose_solver_binding=_binding(
            adapter_raw["pose_solver_binding"], "adapter.pose_solver_binding"
        ),
        ik_position_tolerance_m=_float(
            adapter_raw["ik_position_tolerance_m"],
            "adapter.ik_position_tolerance_m",
            positive=True,
        ),
        ik_orientation_tolerance_rad=_float(
            adapter_raw["ik_orientation_tolerance_rad"],
            "adapter.ik_orientation_tolerance_rad",
            positive=True,
        ),
        jitter=jitter,
    )
    return WorldMindDiagnosticConfig(
        schema_version=root["schema_version"],
        status=root["status"],
        split=root["split"],
        base_seed=FROZEN_BASE_SEED,
        seeds=seeds,
        episodes=FROZEN_EPISODE_COUNT,
        backend=backend,
        task=task,
        pipeline=pipeline,
        bounds=bounds,
        report=report,
        report_definitions=definitions,
        diagnostic=diagnostic,
        assessment=assessment,
        surrogate=surrogate,
        safety=safety,
        verification=verification,
        adapter=adapter,
        source_path=source,
    )


def validate_worldmind_bindings(
    config: WorldMindConfig, bindings: WorldMindBindings
) -> tuple[WorldMindConfig, WorldMindBindings]:
    """Require every configuration binding to resolve to an explicitly registered object."""
    missing: list[str] = []
    invalid: list[str] = []
    for name, spec in config.components.items():
        obj = bindings.components.get(spec.binding)
        if obj is None:
            missing.append(f"components.{name}={spec.binding}")
        elif not callable(obj):
            invalid.append(f"components.{name}={spec.binding}")
    for name in config.assessment.hard_checks:
        obj = bindings.hard_checks.get(name)
        if obj is None:
            missing.append(f"hard_checks.{name}")
        elif not callable(obj):
            invalid.append(f"hard_checks.{name}")
    evaluator = bindings.calibration_evaluators.get(config.calibration.evaluator_binding)
    if evaluator is None:
        missing.append(f"calibration_evaluators.{config.calibration.evaluator_binding}")
    elif not callable(evaluator):
        invalid.append(f"calibration_evaluators.{config.calibration.evaluator_binding}")
    factories = (
        ("surrogates", config.surrogate.binding),
        ("safety", config.safety.binding),
        ("verification", config.verification.binding),
    )
    for group, binding in factories:
        obj = getattr(bindings, group).get(binding)
        if obj is None:
            missing.append(f"{group}.{binding}")
        elif not callable(obj):
            invalid.append(f"{group}.{binding}")
    solver = bindings.pose_solvers.get(config.adapter.pose_solver_binding)
    if solver is None:
        missing.append(f"pose_solvers.{config.adapter.pose_solver_binding}")
    elif not callable(getattr(solver, "solve", None)):
        invalid.append(f"pose_solvers.{config.adapter.pose_solver_binding}")
    for binding in (
        config.adapter.waypoints_binding,
        config.adapter.alternate_waypoints_binding,
    ):
        if binding is None:
            continue
        waypoints = bindings.waypoint_sets.get(binding)
        factory = bindings.waypoint_factories.get(binding)
        if waypoints is None and factory is None:
            missing.append(f"waypoints.{binding}")
        elif factory is not None and not callable(factory):
            invalid.append(f"waypoint_factories.{binding}")
        elif waypoints is not None and (not isinstance(waypoints, Sequence) or not waypoints):
            invalid.append(f"waypoint_sets.{binding}")
    for group in ("surrogate", "safety"):
        declared = getattr(config, group).inputs
        registered = bindings.inputs.get(group, {})
        for name in declared:
            if name not in registered:
                missing.append(f"inputs.{group}.{name}")
    if missing or invalid:
        if missing:
            raise ValueError("unregistered WorldMind bindings: " + ", ".join(sorted(missing)))
        raise ValueError("invalid WorldMind bindings: " + ", ".join(sorted(invalid)))
    return config, bindings


def validate_worldmind_diagnostic_bindings(
    config: WorldMindDiagnosticConfig, bindings: WorldMindBindings
) -> tuple[WorldMindDiagnosticConfig, WorldMindBindings]:
    """Resolve every hard-constraint binding, skipping only not-yet-calibrated scientific ones.

    Hard checks, surrogate, safety, verification, pose solver, waypoints and input bindings are
    validated exactly as in production. The Eq. 3.8 component registry and calibration evaluator
    are intentionally absent at stage 1 and must not be consulted by a diagnostic run.
    """
    missing: list[str] = []
    invalid: list[str] = []
    # Hard checks are safety-relevant: every configured veto stays mandatory in diagnostics.
    for name in config.assessment.hard_checks:
        obj = bindings.hard_checks.get(name)
        if obj is None:
            missing.append(f"hard_checks.{name}")
        elif not callable(obj):
            invalid.append(f"hard_checks.{name}")
    for group, binding in (
        ("surrogates", config.surrogate.binding),
        ("safety", config.safety.binding),
        ("verification", config.verification.binding),
    ):
        obj = getattr(bindings, group).get(binding)
        if obj is None:
            missing.append(f"{group}.{binding}")
        elif not callable(obj):
            invalid.append(f"{group}.{binding}")
    solver = bindings.pose_solvers.get(config.adapter.pose_solver_binding)
    if solver is None:
        missing.append(f"pose_solvers.{config.adapter.pose_solver_binding}")
    elif not callable(getattr(solver, "solve", None)):
        invalid.append(f"pose_solvers.{config.adapter.pose_solver_binding}")
    for binding in (config.adapter.waypoints_binding, config.adapter.alternate_waypoints_binding):
        if binding is None:
            continue
        waypoints = bindings.waypoint_sets.get(binding)
        factory = bindings.waypoint_factories.get(binding)
        if waypoints is None and factory is None:
            missing.append(f"waypoints.{binding}")
        elif factory is not None and not callable(factory):
            invalid.append(f"waypoint_factories.{binding}")
        elif waypoints is not None and (not isinstance(waypoints, Sequence) or not waypoints):
            invalid.append(f"waypoint_sets.{binding}")
    for group in ("surrogate", "safety"):
        declared = getattr(config, group).inputs
        registered = bindings.inputs.get(group, {})
        for name in declared:
            if name not in registered:
                missing.append(f"inputs.{group}.{name}")
    if missing or invalid:
        if missing:
            raise ValueError("unregistered diagnostic bindings: " + ", ".join(sorted(missing)))
        raise ValueError("invalid diagnostic bindings: " + ", ".join(sorted(invalid)))
    return config, bindings


def plan_worldmind_validation(
    config_path: Path,
    out_dir: Path,
    *,
    split: str,
    backend: str,
    state_source: str,
    bindings: WorldMindBindings | None = None,
) -> tuple[WorldMindConfig, tuple[EvalEpisode, ...]]:
    """Pure CLI preflight: VALIDATION-only, frozen plan, fresh output, all bindings present."""
    if split != FROZEN_SPLIT:
        raise ValueError("TEST and TRAIN are sealed for this command; VALIDATION only")
    if out_dir.exists():
        raise FileExistsError(f"preserve prior evidence; output already exists: {out_dir}")
    resolved_bindings = bindings or empty_worldmind_bindings()
    config = load_worldmind_config(config_path)
    validate_worldmind_bindings(config, resolved_bindings)
    if config.backend.kind != backend:
        raise ValueError("CLI backend declaration differs from validated config")
    if state_source != FROZEN_STATE_SOURCE:
        raise ValueError("WorldMind production state source must be declared as oracle")
    plan = build_episode_plan(FROZEN_SPLIT, FROZEN_BASE_SEED, n=0, allow_test=False)
    if len(plan) != FROZEN_EPISODE_COUNT:
        raise ValueError("frozen VALIDATION plan does not contain 18 episodes")
    if sorted({entry.seed for entry in plan}) != list(FROZEN_SEEDS):
        raise ValueError("frozen VALIDATION seed contract violated")
    pairs = [(entry.scene.name.rsplit("-s", 1)[0], entry.seed) for entry in plan]
    if len(set(pairs)) != FROZEN_EPISODE_COUNT:
        raise ValueError("frozen VALIDATION (scene, seed) pairs are not unique")
    return config, plan


def plan_worldmind_diagnostic(
    config_path: Path,
    out_dir: Path,
    *,
    split: str,
    backend: str,
    state_source: str,
    bindings: WorldMindBindings | None = None,
) -> tuple[WorldMindDiagnosticConfig, tuple[EvalEpisode, ...]]:
    """Stage-1/2 pure gate: uncalibrated config, frozen plan and every hard binding present.

    This function deliberately cannot return a scored plan or a calibration result. It exists so
    evidence collection can be configured before an approved candidate exists, closing the
    bootstrap loop without weakening any hard constraint.
    """
    if split != FROZEN_SPLIT:
        raise ValueError("TEST and TRAIN are sealed for this command; VALIDATION only")
    if out_dir.exists():
        raise FileExistsError(f"preserve prior evidence; output already exists: {out_dir}")
    if state_source != FROZEN_STATE_SOURCE:
        raise ValueError("WorldMind diagnostic state source must be declared as oracle")
    config = load_worldmind_diagnostic_config(config_path)
    validate_worldmind_diagnostic_bindings(config, bindings or empty_worldmind_bindings())
    if config.backend.kind != backend:
        raise ValueError("CLI backend declaration differs from validated diagnostic config")
    if config.diagnostic.scored_evaluation_allowed:
        raise ValueError("uncalibrated diagnostic config cannot allow scored evaluation")
    plan = build_episode_plan(FROZEN_SPLIT, FROZEN_BASE_SEED, n=0, allow_test=False)
    if len(plan) != FROZEN_EPISODE_COUNT:
        raise ValueError("frozen VALIDATION plan does not contain 18 episodes")
    if sorted({entry.seed for entry in plan}) != list(FROZEN_SEEDS):
        raise ValueError("frozen VALIDATION seed contract violated")
    if len({(entry.scene.name.rsplit("-s", 1)[0], entry.seed) for entry in plan}) != 18:
        raise ValueError("frozen VALIDATION (scene, seed) pairs are not unique")
    return config, plan
