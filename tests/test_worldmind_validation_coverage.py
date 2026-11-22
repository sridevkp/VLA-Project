"""Coverage checks for the fixed VALIDATION report, with no campaign or scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlarobot.eval.worldmind_validation_coverage import (
    load_canonical_baseline_coverage,
    reconcile_validation_outcomes,
)

BASELINE = (
    Path(__file__).resolve().parents[1] / "docs/evidence/validation-v2-canonical-physical.json"
)


def test_canonical_baseline_reconciles_all_frozen_trials():
    coverage = load_canonical_baseline_coverage(BASELINE)
    assert coverage.complete
    assert len(coverage.measured_episodes) == 18
    assert coverage.successes == 15
    assert coverage.recorded_failure_states == (("PREGRASP", 1), ("TRANSIT", 2))


def test_partial_outcomes_distinguish_missing_from_measured_failure():
    row = json.loads(BASELINE.read_bytes())["records"][2]
    coverage = reconcile_validation_outcomes([row])
    assert coverage.measured_episodes == ("validation-reach-limited-i02-s0",)
    assert len(coverage.missing_episodes) == 17
    assert coverage.successes == 0
    assert coverage.recorded_failure_states == (("PREGRASP", 1),)
    assert not coverage.complete


@pytest.mark.parametrize(
    "defect",
    [{"index": True}, {"seed": False}, {"scene": "validation-clear-i03"}, {"success": None}],
)
def test_out_of_plan_or_nonterminal_record_rejected(defect):
    row = json.loads(BASELINE.read_bytes())["records"][0]
    with pytest.raises(ValueError):
        reconcile_validation_outcomes([{**row, **defect}])


def test_duplicate_and_noncanonical_artifact_are_rejected(tmp_path):
    body = json.loads(BASELINE.read_bytes())
    row = body["records"][0]
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_validation_outcomes([row, row])
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed canonical"):
        load_canonical_baseline_coverage(path)
