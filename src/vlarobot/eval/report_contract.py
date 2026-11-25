"""Report bindings require authoritative definitions, never guessed metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

COMPONENTS = ("Jg", "Jl", "Jr", "Ju")
TABLE51_NAMES = (
    "Task success rate",
    "Prediction error",
    "Collision rate",
    "Path length",
    "Execution time",
    "Planning latency",
    "Risk rejection rate",
    "Reliability",
)
APPROVED_DEFINITION_STATUS = "approved"


@dataclass(frozen=True)
class ReportDefinitions:
    """Checksum binds an explicitly reviewed transcription; code cannot attest its semantics."""

    report_sha256: str
    component_definitions: Mapping[str, str]
    table51_definitions: Mapping[str, dict[str, str]]
    review_status: str

    @classmethod
    def load(cls, definitions: Path, report_pdf: Path) -> ReportDefinitions:
        data = json.loads(definitions.read_text(encoding="utf-8"))
        digest = hashlib.sha256(report_pdf.read_bytes()).hexdigest()
        if data.get("report_sha256") != digest:
            raise ValueError("definitions differ from authoritative report bytes")
        status = data.get("review_status", APPROVED_DEFINITION_STATUS)
        if status != APPROVED_DEFINITION_STATUS:
            raise ValueError("definitions candidate is not owner-approved for scored execution")
        components = data.get("components", {})
        metrics = data.get("table51", {})
        if set(components) != set(COMPONENTS) or any(
            not isinstance(v, str) or not v.strip() for v in components.values()
        ):
            raise ValueError("exact four authoritative J component definitions required")
        if set(metrics) != set(TABLE51_NAMES) or any(
            not isinstance(v, dict)
            or set(v) != {"definition", "unit"}
            or not all(isinstance(s, str) and s.strip() for s in v.values())
            for v in metrics.values()
        ):
            raise ValueError("all eight verbatim Table 5.1 names, definitions and units required")
        return cls(digest, components, metrics, status)


def load_definitions_candidate(definitions: Path, report_pdf: Path) -> dict[str, Any]:
    """Load a reviewable candidate without authorizing it for scored execution."""
    data = json.loads(definitions.read_text(encoding="utf-8"))
    digest = hashlib.sha256(report_pdf.read_bytes()).hexdigest()
    if data.get("report_sha256") != digest:
        raise ValueError("definitions candidate differs from authoritative report bytes")
    if data.get("review_status") != "candidate":
        raise ValueError("definitions candidate must explicitly remain review_status=candidate")
    if set(data.get("components", {})) != set(COMPONENTS):
        raise ValueError("definitions candidate must name exactly Jg, Jl, Jr and Ju")
    if set(data.get("table51", {})) != set(TABLE51_NAMES):
        raise ValueError("definitions candidate must use the exact eight Table 5.1 names")
    return data


def evaluate_components(
    definitions: ReportDefinitions | None,
    bindings: Mapping[str, Callable[[Mapping[str, Any]], float]],
    features: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    if definitions is None:
        raise ValueError("authoritative Phase I definitions unavailable; no guessed J")
    if set(bindings) != set(COMPONENTS):
        raise ValueError("all four explicit component implementations required")
    values = [float(bindings[name](features)) for name in COMPONENTS]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite report component")
    return values[0], values[1], values[2], values[3]


def table51_values(
    definitions: ReportDefinitions | None, values: Mapping[str, float]
) -> dict[str, float]:
    if definitions is None:
        raise ValueError("authoritative Table 5.1 definitions unavailable")
    if set(values) != set(definitions.table51_definitions):
        raise ValueError("Table 5.1 names differ from authoritative definitions")
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("nonfinite Table 5.1 metric")
    return dict(values)


@dataclass(frozen=True)
class Table51MetricValue:
    """Additive metric coverage. Callers retain the approved formulas and denominators."""

    status: Literal["measured", "unavailable", "not_applicable"]
    value: float | None
    reason: str | None = None
    denominator: int | None = None
    valid_paired_steps: int | None = None

    def __post_init__(self) -> None:
        if self.status not in ("measured", "unavailable", "not_applicable"):
            raise ValueError("unknown Table 5.1 metric status")
        if self.status == "measured":
            if type(self.value) not in (int, float) or not math.isfinite(cast("float", self.value)):
                raise ValueError("measured Table 5.1 values must be finite")
            if self.reason is not None or self.denominator == 0 or self.valid_paired_steps == 0:
                raise ValueError("measured Table 5.1 values require valid evidence")
        elif self.value is not None or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("absent Table 5.1 values require null and a reason")
        for count in (self.denominator, self.valid_paired_steps):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("Table 5.1 counts must be nonnegative integers")


def table51_nullable_values(
    definitions: ReportDefinitions | None, values: Mapping[str, Table51MetricValue]
) -> dict[str, Any]:
    """Serialize every approved metric, including absent values, without changing numeric export."""
    if definitions is None or definitions.review_status != APPROVED_DEFINITION_STATUS:
        raise ValueError("approved Table 5.1 definitions required")
    if set(values) != set(TABLE51_NAMES) or set(values) != set(definitions.table51_definitions):
        raise ValueError("all approved Table 5.1 metrics must be represented")
    for name in ("Task success rate", "Collision rate", "Risk rejection rate", "Reliability"):
        if values[name].status == "measured" and values[name].denominator is None:
            raise ValueError(f"{name} requires its approved denominator")
    prediction = values["Prediction error"]
    if prediction.status == "measured" and prediction.valid_paired_steps is None:
        raise ValueError("measured prediction error requires valid paired steps")
    if prediction.status == "not_applicable" and prediction.valid_paired_steps != 0:
        raise ValueError("not-applicable prediction error requires zero valid paired steps")
    return {
        "schema_version": 1,
        "metrics": {
            name: {
                **asdict(values[name]),
                "unit": definitions.table51_definitions[name]["unit"],
            }
            for name in TABLE51_NAMES
        },
    }


def table51_comparison_eligible(
    metric: str,
    expected_arms: tuple[str, ...],
    by_arm: Mapping[str, Mapping[str, Table51MetricValue]],
) -> bool:
    """Require measured coverage for every requested arm before a numeric comparison."""
    if metric not in TABLE51_NAMES:
        raise ValueError("unknown Table 5.1 metric")
    if not expected_arms or len(set(expected_arms)) != len(expected_arms):
        raise ValueError("explicit unique comparison arms required")
    return all(
        arm in by_arm and metric in by_arm[arm] and by_arm[arm][metric].status == "measured"
        for arm in expected_arms
    )
