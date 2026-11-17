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
from vlarobot.eval.worldmind_config import GATE_CALL_POLICY


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass
class WorldMindEvidenceWriter:
    """Create a fresh sidecar directory and append only measured decision evidence."""

    out_dir: Path
    folder_name: str = "worldmind-sidecars"
    decisions: list[DecisionEvidence] = field(default_factory=list, init=False)
    pairings: list[PredictionPairingEvidence] = field(default_factory=list, init=False)

    @property
    def folder(self) -> Path:
        return self.out_dir / self.folder_name

    def start(self, provenance: Mapping[str, str]) -> None:
        self.folder.mkdir(parents=True, exist_ok=False)
        _write_json(self.folder / "provenance.json", dict(provenance))
        for name in ("decisions.jsonl", "prediction_pairing.jsonl", "errors.jsonl"):
            with (self.folder / name).open("x", encoding="utf-8"):
                pass

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

    def record_error(self, episode_id: str, error: BaseException) -> None:
        """Record a real exception; never synthesize missing decision records for it."""
        if not self.folder.is_dir():
            raise RuntimeError("evidence writer must be started before recording errors")
        _append_jsonl(
            self.folder / "errors.jsonl",
            {
                "episode_id": episode_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def finish(self) -> dict[str, Path]:
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
        expected_gate_calls = [evidence.candidate_count + 1 for evidence in decisions]
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
                "policy": GATE_CALL_POLICY,
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
            "note": (
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
        paths = {
            "decisions": self.folder / "decisions.jsonl",
            "prediction_pairing": self.folder / "prediction_pairing.jsonl",
            "errors": self.folder / "errors.jsonl",
            "risk_rejection": self.folder / "risk_rejection.json",
            "phase_latency": self.folder / "phase_latency.json",
            "verification_recovery": self.folder / "verification_recovery.json",
            "provenance": self.folder / "provenance.json",
        }
        for name in ("risk_rejection", "phase_latency", "verification_recovery"):
            payload = {
                "risk_rejection": rejection_payload,
                "phase_latency": latency_payload,
                "verification_recovery": recovery_payload,
            }[name]
            _write_json(paths[name], payload)
        return paths
