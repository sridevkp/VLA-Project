"""Report bindings require authoritative definitions, never guessed metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
