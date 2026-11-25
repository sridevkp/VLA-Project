"""Parallel continuation refuses source drift and never rolls out missing physical data in fake."""

import json
from types import SimpleNamespace

import pytest

from vlarobot.execution import parallel_eval


def test_parallel_resume_refuses_active_owner_and_frozen_source_drift(tmp_path):
    owner = tmp_path / "active.pid"
    owner.write_text("1", encoding="utf-8")
    with pytest.raises(ValueError, match="serial"):
        parallel_eval.resume_parallel_baseline(tmp_path, webots="unused", workers=2)
    owner.unlink()
    (tmp_path / "provenance.json").write_text(
        json.dumps({"files_sha256": {"configs/eval.yaml": "changed"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source changed"):
        parallel_eval.resume_parallel_baseline(tmp_path, webots="unused", workers=2)


def test_completed_continuation_aggregates_only_retained_physical_outcomes(tmp_path, monkeypatch):
    (tmp_path / "provenance.json").write_text(json.dumps({"files_sha256": {}}), encoding="utf-8")
    monkeypatch.setattr(parallel_eval, "_completed_recording", lambda *args: object())

    def aggregate(plan, backend, config, **kwargs):
        assert len(plan) == 24
        assert config.resume is True
        assert config.measurement == "measured"
        return SimpleNamespace(summary={"success": {"rate": 0.75, "successes": 18}})

    def no_launch(**kwargs):
        raise AssertionError("completed physical recordings must not launch replacement episodes")

    monkeypatch.setattr(parallel_eval, "run_evaluation", aggregate)
    monkeypatch.setattr(parallel_eval, "launch_webots_session", no_launch)
    result = parallel_eval.resume_parallel_baseline(tmp_path, webots="unused", workers=2)
    assert result["literal_vla_gate_threshold"] == 0.4
    assert result["vla_gate_result"].startswith("UNMEASURED")
