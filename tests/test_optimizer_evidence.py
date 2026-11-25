"""Post-step evidence is absent for forward-only work and records actual parameter changes."""

import json

import pytest

from vlarobot.vla.training_evidence import OptimizerStepEvidence

torch = pytest.importorskip("torch", reason="optional installed optimizer integration")


def test_actual_adamw_step_is_required_for_evidence(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    path = tmp_path / "steps.jsonl"
    OptimizerStepEvidence(
        optimizer, path, start_step=0, identity={"run_kind": "toy-regression-only"}
    )
    loss = model(torch.ones(1, 2)).square().mean()
    assert not path.exists()
    loss.backward()
    assert not path.exists()
    optimizer.step()
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["optimizer_step"] == 1 and row["actual_post_step_hook"] is True
    assert row["sampled_parameters_changed"] is True
    assert row["run_kind"] == "toy-regression-only"
