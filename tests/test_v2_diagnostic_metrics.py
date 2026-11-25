"""Regression check for the reproducible V2 diagnostic metric derivation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlarobot.eval.v2_diagnostics import summarize

RUN = Path(__file__).resolve().parents[1] / "runs/worldmind/v2-train-endpoint-1"


def _recorded() -> list[dict[str, object]]:
    path = RUN / "derived-motion-summary.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_reproduces_recorded_endpoint_summary() -> None:
    derived = summarize(RUN)
    recorded = _recorded()
    assert len(derived) == len(recorded) == 3
    for actual, expected in zip(derived, recorded, strict=True):
        assert actual["trial"] == expected["trial"]
        assert actual["scenario"]["name"] == expected["scenario"]["name"]
        assert actual["terminal"] == expected["terminal"]
        assert actual["queries"] == expected["queries"]
        assert actual["actions"] == expected["actions"]
        assert actual["fallbacks"] == expected["fallbacks"]
        assert actual["closest_m"] == pytest.approx(expected["closest_m"], abs=1e-9)
        assert actual["rise_m"] == pytest.approx(expected["rise_m"], abs=1e-12)
        assert actual["raw_outside"] == expected["raw_outside"]
        assert actual["verification"] == expected["verification"]
        assert actual["gripper_events"] == expected["gripper_events"]


def test_no_episodes_is_an_empty_summary(tmp_path: Path) -> None:
    assert summarize(tmp_path) == []
