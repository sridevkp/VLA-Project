"""Offline ADR-0084 disposition bridge for the frozen Table 5.1 population."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vlarobot.core.types import ApplyResult
from vlarobot.eval.report_contract import Table51MetricValue
from vlarobot.eval.worldmind_config import FROZEN_CANDIDATE_COUNT

if TYPE_CHECKING:
    from vlarobot.execution.worldmind_heuristic_comparison import (
        HeuristicCandidate,
        HeuristicDecision,
    )

Disposition = Literal["hard_vetoed", "safe_unselected", "selected", "invalid", "unevaluable"]


@dataclass(frozen=True)
class HeuristicTable51Population:
    """Candidate population and separate execution-time motor veto count."""

    candidate_dispositions: tuple[tuple[str, Disposition], ...]
    hard_vetoed: int
    evaluated: int
    invalid: int
    unevaluable: int
    motor_vetoes: int
    prediction_error: Table51MetricValue

    @property
    def rate(self) -> float | None:
        return self.hard_vetoed / self.evaluated if self.evaluated else None


def _disposition(row: HeuristicCandidate) -> Disposition:
    first = row.first_command
    known = isinstance(first, ApplyResult) and type(first.accepted) is bool
    if row.finite_chunk is False:
        result = "invalid"
    elif row.finite_chunk is not True or not known:
        result = "unevaluable"
    elif not first.accepted:
        result = "hard_vetoed"
    else:
        result = "selected" if row.disposition == "selected" else "safe_unselected"
    expected = {"selected": "selected", "safe_unselected": "eligible"}.get(result, "rejected")
    if row.disposition != expected:
        raise ValueError("heuristic disposition contradicts recorded check outcomes")
    return result


def heuristic_table51_population(
    decisions: Iterable[HeuristicDecision],
) -> HeuristicTable51Population:
    """Reconcile fixture dispositions without treating invalid inputs as evaluated vetoes."""
    rows = []
    decisions_seen = set()
    motor_vetoes = 0
    for decision in decisions:
        key = decision.generation.decision_id
        if key in decisions_seen:
            raise ValueError("duplicate heuristic decision")
        decisions_seen.add(key)
        candidates = decision.candidates
        if (
            len(candidates) != FROZEN_CANDIDATE_COUNT
            or decision.generation.candidate_count != len(candidates)
            or decision.generation.proposer_source != "scripted"
            or len(decision.generation.draw_seeds) != len(candidates)
            or any(
                row.candidate_id != f"candidate-{index:02d}"
                or row.draw_index != index
                or row.draw_seed != decision.generation.draw_seeds[index]
                for index, row in enumerate(candidates)
            )
        ):
            raise ValueError("unreconciled heuristic candidate pool")
        converted = [(row.candidate_id, _disposition(row)) for row in candidates]
        eligible = [
            cid for cid, disposition in converted if disposition in {"selected", "safe_unselected"}
        ]
        selected = [cid for cid, disposition in converted if disposition == "selected"]
        if (
            selected != eligible[:1]
            or (selected[0] if selected else None) != decision.selected_candidate_id
        ):
            raise ValueError("heuristic selection contradicts draw-order eligibility")
        rows.extend((f"{key}/{cid}", disposition) for cid, disposition in converted)
        if decision.motor_verdict is not None:
            if not selected or type(decision.motor_verdict.accepted) is not bool:
                raise ValueError("invalid execution motor verdict")
            motor_vetoes += not decision.motor_verdict.accepted

    return HeuristicTable51Population(
        tuple(rows),
        sum(disposition == "hard_vetoed" for _, disposition in rows),
        sum(
            disposition in {"hard_vetoed", "safe_unselected", "selected"} for _, disposition in rows
        ),
        sum(disposition == "invalid" for _, disposition in rows),
        sum(disposition == "unevaluable" for _, disposition in rows),
        motor_vetoes,
        Table51MetricValue("not_applicable", None, "arm_has_no_prediction", valid_paired_steps=0),
    )
