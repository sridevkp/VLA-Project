"""Non-overwriting sidecar writer for the production WorldMind composition.

The frozen episode recorder is untouched. These records are additive evidence: every absent
measurement remains absent, and no sidecar value is fed back into generation, prediction,
RiskGate ranking, safety, recovery, execution, or scoring.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vlarobot.eval.decision_evidence import DecisionEvidence, PredictionPairingEvidence
from vlarobot.eval.decision_metrics import (
    candidate_disposition_counts,
    risk_rejection_aggregate,
    summarize_phase_latencies,
)
from vlarobot.eval.observed_state_feedback import (
    COLLISION_PROXY_ATTRIBUTION_SIDECAR_NAME,
    DIAGNOSTIC_EVENT_SIDECAR_NAME,
    FEEDBACK_SIDECAR_NAME,
    OBSERVED_STATE_FEEDBACK_EXECUTION_MODE,
    read_feedback_evidence,
    validate_feedback_evidence_row,
)
from vlarobot.eval.worldmind_config import GATE_CALL_POLICY


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_instrumentation(
    decisions: Sequence[DecisionEvidence],
    *,
    measurement_class: str,
    gate_policy: str = GATE_CALL_POLICY,
    expected_gate_calls_per_decision: int | None = None,
    latency_note: str | None = None,
) -> dict[str, Any]:
    """Aggregate real, recorded decision evidence into Table 5.1 instrumentation.

    Every count and latency comes from the additive decision sidecars. No decision count is
    inferred from control-step counts and no latency is substituted from a nominal kernel
    profile; a phase that was not measured stays absent with its recorded reason.
    """
    gate_values = [
        int(value)
        for evidence in decisions
        if (value := evidence.provenance.get("gate_select_calls")) is not None
    ]
    expected_gate_calls = [
        evidence.candidate_count + 1
        if expected_gate_calls_per_decision is None
        else expected_gate_calls_per_decision
        for evidence in decisions
    ]
    latency = summarize_phase_latencies(
        [evidence.timings for evidence in decisions if evidence.timings is not None]
    )
    per_episode: dict[str, int] = {}
    for evidence in decisions:
        per_episode[evidence.episode_id] = per_episode.get(evidence.episode_id, 0) + 1
    return {
        "schema_version": 1,
        "measurement_class": measurement_class,
        "decisions_total": len(decisions),
        "decisions_per_episode": dict(sorted(per_episode.items())),
        "candidate_evaluations_total": sum(e.candidate_count for e in decisions),
        "gate_select_calls": {
            "policy": gate_policy,
            "recorded_decisions": len(gate_values),
            "total": sum(gate_values),
            "expected_total": sum(expected_gate_calls),
            "all_decisions_expected": (
                gate_values == expected_gate_calls if len(gate_values) == len(decisions) else False
            ),
        },
        "phase_latency": {name: asdict(summary) for name, summary in sorted(latency.items())},
        "note": latency_note
        or (
            "Decision counts, candidate evaluations and phase latencies are measured from real "
            "WorldMind decision sidecars; no count is inferred from control steps and no latency "
            "is taken from a nominal kernel profile."
        ),
    }


@dataclass
class WorldMindEvidenceWriter:
    """Create a fresh sidecar directory and append only measured decision evidence."""

    out_dir: Path
    folder_name: str = "worldmind-sidecars"
    decisions: list[DecisionEvidence] = field(default_factory=list, init=False)
    pairings: list[PredictionPairingEvidence] = field(default_factory=list, init=False)
    errors: list[dict[str, Any]] = field(default_factory=list, init=False)
    feedback_rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    diagnostic_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    proxy_rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    measurement_class: str = field(default="synthetic", init=False)
    resumed: bool = field(default=False, init=False)
    feedback_enabled: bool = field(default=False, init=False)

    @property
    def folder(self) -> Path:
        return self.out_dir / self.folder_name

    @property
    def feedback_path(self) -> Path:
        return self.folder / FEEDBACK_SIDECAR_NAME

    @property
    def diagnostic_event_path(self) -> Path:
        return self.folder / DIAGNOSTIC_EVENT_SIDECAR_NAME

    @property
    def collision_proxy_path(self) -> Path:
        return self.folder / COLLISION_PROXY_ATTRIBUTION_SIDECAR_NAME

    def _sidecar_names(self) -> list[str]:
        names = ["decisions.jsonl", "prediction_pairing.jsonl", "errors.jsonl"]
        if self.feedback_enabled:
            names.extend(
                [
                    FEEDBACK_SIDECAR_NAME,
                    DIAGNOSTIC_EVENT_SIDECAR_NAME,
                    COLLISION_PROXY_ATTRIBUTION_SIDECAR_NAME,
                ]
            )
        return names

    @property
    def episode_ids(self) -> set[str]:
        """Episode IDs with existing decision or error evidence, including resumed records."""
        return {item.episode_id for item in self.decisions} | {
            str(item["episode_id"]) for item in self.errors
        }

    def start(self, provenance: Mapping[str, str], *, resume: bool = False) -> None:
        # Never claim a measured class unless a real Webots backend was declared.
        self.measurement_class = (
            "measured" if provenance.get("backend") == "webots" else "synthetic"
        )
        self.resumed = resume
        self.feedback_enabled = (
            provenance.get("execution_mode") == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
        )
        if resume:
            self._resume_existing(provenance)
            return
        self.folder.mkdir(parents=True, exist_ok=False)
        _write_json(self.folder / "provenance.json", dict(provenance))
        names = self._sidecar_names()
        for name in names:
            with (self.folder / name).open("x", encoding="utf-8"):
                pass

    def _resume_existing(self, provenance: Mapping[str, str]) -> None:
        """Reuse an interrupted sidecar set only when its provenance is byte-identical."""
        if not self.folder.is_dir():
            raise FileNotFoundError(f"cannot resume without sidecars: {self.folder}")
        previous = json.loads((self.folder / "provenance.json").read_text(encoding="utf-8"))
        if previous != dict(provenance):
            raise ValueError("resume provenance differs from the existing sidecar provenance")
        names = self._sidecar_names()
        for name in names:
            if not (self.folder / name).is_file():
                raise FileNotFoundError(f"cannot resume without sidecar file: {name}")
        self.decisions.extend(
            DecisionEvidence.from_dict(row) for row in _read_jsonl(self.folder / "decisions.jsonl")
        )
        self.pairings.extend(
            PredictionPairingEvidence.from_dict(row)
            for row in _read_jsonl(self.folder / "prediction_pairing.jsonl")
        )
        self.errors.extend(_read_jsonl(self.folder / "errors.jsonl"))
        if self.feedback_enabled:
            self.feedback_rows.extend(read_feedback_evidence(self.feedback_path))
            self.diagnostic_events.extend(_read_jsonl(self.diagnostic_event_path))
            self.proxy_rows.extend(_read_jsonl(self.collision_proxy_path))

    def record_decision(
        self,
        evidence: DecisionEvidence,
        pairing: PredictionPairingEvidence | None = None,
    ) -> None:
        """Append one decision and its optional executed-candidate prediction pairing."""
        if not self.folder.is_dir():
            raise RuntimeError("evidence writer must be started before recording decisions")
        _append_jsonl(self.folder / "decisions.jsonl", evidence.to_dict())
        self.decisions.append(evidence)
        if pairing is not None:
            _append_jsonl(self.folder / "prediction_pairing.jsonl", pairing.to_dict())
            self.pairings.append(pairing)

    def record_feedback(self, evidence: Any) -> None:
        """Append versioned observed-state feedback evidence; it is never paired or scored."""
        if not self.feedback_enabled:
            raise RuntimeError("feedback evidence cannot be recorded outside the feedback mode")
        if not self.folder.is_dir():
            raise RuntimeError("evidence writer must be started before recording feedback")
        payload = validate_feedback_evidence_row(evidence.to_dict())
        _append_jsonl(self.feedback_path, payload)
        self.feedback_rows.append(payload)

    def record_diagnostic_event(self, payload: Mapping[str, Any]) -> None:
        """Append one explicitly labelled diagnostic-only event; it is never scored."""
        if not self.feedback_enabled:
            raise RuntimeError("diagnostic events belong only to the feedback mode")
        if not self.folder.is_dir():
            raise RuntimeError("evidence writer must be started before recording diagnostics")
        row = dict(payload)
        _append_jsonl(self.diagnostic_event_path, row)
        self.diagnostic_events.append(row)

    def record_collision_proxy(self, evidence: Any) -> None:
        """Append versioned per-object proxy attribution; proxy hits are not contacts."""
        if not self.feedback_enabled:
            raise RuntimeError("proxy attribution belongs only to the feedback mode")
        if not self.folder.is_dir():
            raise RuntimeError("evidence writer must be started before recording proxy evidence")
        payload = evidence.to_dict()
        _append_jsonl(self.collision_proxy_path, payload)
        self.proxy_rows.append(payload)

    def record_error(self, episode_id: str, error: BaseException) -> None:
        """Record a real exception; never synthesize missing decision records for it."""
        if not self.folder.is_dir():
            raise RuntimeError("evidence writer must be started before recording errors")
        payload = {
            "episode_id": episode_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _append_jsonl(self.folder / "errors.jsonl", payload)
        self.errors.append(payload)

    def finish(
        self,
        *,
        gate_policy: str = GATE_CALL_POLICY,
        expected_gate_calls_per_decision: int | None = None,
        latency_note: str | None = None,
    ) -> dict[str, Path]:
        """Write risk, latency and recovery summaries from recorded decisions only."""
        decisions: Sequence[DecisionEvidence] = tuple(self.decisions)
        aggregate = risk_rejection_aggregate(decisions)
        latency = summarize_phase_latencies(
            [evidence.timings for evidence in decisions if evidence.timings is not None]
        )
        gate_values = [
            int(value)
            for evidence in decisions
            if (value := evidence.provenance.get("gate_select_calls")) is not None
        ]
        expected_gate_calls = [
            evidence.candidate_count + 1
            if expected_gate_calls_per_decision is None
            else expected_gate_calls_per_decision
            for evidence in decisions
        ]
        rejection_payload = {
            "schema_version": 1,
            "decisions": len(decisions),
            "pooled": asdict(aggregate.pooled),
            "macro": None if aggregate.macro is None else asdict(aggregate.macro),
            "decisions_with_defined_rate": aggregate.decisions_with_defined_rate,
            "decisions_with_undefined_rate": aggregate.decisions_with_undefined_rate,
            "disposition_counts": candidate_disposition_counts(decisions),
            "pairing_records": len(self.pairings),
            "gate_calls": {
                "policy": gate_policy,
                "recorded_decisions": len(gate_values),
                "total": sum(gate_values),
                "expected_total": sum(expected_gate_calls),
                "all_decisions_expected": (
                    gate_values == expected_gate_calls
                    if len(gate_values) == len(decisions)
                    else False
                ),
            },
        }
        latency_payload = {
            "schema_version": 1,
            "decisions_with_timings": sum(evidence.timings is not None for evidence in decisions),
            "phases": {name: asdict(summary) for name, summary in sorted(latency.items())},
            "note": latency_note
            or (
                "Whole-phase perf_counter_ns durations. Inclusive phases overlap and are never "
                "added. Hard checks and RiskGate ranking are fused inside ComparisonPlanner; "
                "K=32 incurs 33 gate calls per decision inside selection_s."
            ),
        }
        transitions: dict[str, int] = {}
        recovery_events: dict[str, int] = {}
        for evidence in decisions:
            if evidence.verification is not None and evidence.verification.next_action is not None:
                name = evidence.verification.next_action
                transitions[name] = transitions.get(name, 0) + 1
            for name in evidence.recovery_events:
                recovery_events[name] = recovery_events.get(name, 0) + 1
        recovery_payload = {
            "schema_version": 1,
            "decisions": len(decisions),
            "no_safe_set_decisions": sum(evidence.no_safe_set for evidence in decisions),
            "verification_transitions": dict(sorted(transitions.items())),
            "no_safe_recovery_events": dict(sorted(recovery_events.items())),
        }
        instrumentation_payload = build_instrumentation(
            decisions,
            measurement_class=self.measurement_class,
            gate_policy=gate_policy,
            expected_gate_calls_per_decision=expected_gate_calls_per_decision,
            latency_note=latency_note,
        )
        paths = {
            "decisions": self.folder / "decisions.jsonl",
            "prediction_pairing": self.folder / "prediction_pairing.jsonl",
            "errors": self.folder / "errors.jsonl",
            "risk_rejection": self.folder / "risk_rejection.json",
            "phase_latency": self.folder / "phase_latency.json",
            "verification_recovery": self.folder / "verification_recovery.json",
            "instrumentation": self.folder / "instrumentation.json",
            "provenance": self.folder / "provenance.json",
        }
        if self.feedback_enabled:
            paths["observed_state_feedback"] = self.feedback_path
            paths["diagnostic_events"] = self.diagnostic_event_path
            paths["collision_proxy_attribution"] = self.collision_proxy_path
        for name in ("risk_rejection", "phase_latency", "verification_recovery", "instrumentation"):
            payload = {
                "risk_rejection": rejection_payload,
                "phase_latency": latency_payload,
                "verification_recovery": recovery_payload,
                "instrumentation": instrumentation_payload,
            }[name]
            _write_json(paths[name], payload, overwrite=self.resumed)
        return paths
