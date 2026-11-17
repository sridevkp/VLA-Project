"""Exercise approved optimizer logging semantics without changing production code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve previous evidence")
    import torch

    source = args.project / "src/vlarobot/vla/training_evidence.py"
    spec = importlib.util.spec_from_file_location("approved_optimizer_evidence", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="optimizer-log-semantics-") as directory:
        parameter = torch.nn.Parameter(torch.ones(64))
        initial = parameter.detach().clone()
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = Path(directory) / "optimizer-steps.jsonl"
        module.OptimizerStepEvidence(
            optimizer,
            events,
            start_step=19999,
            identity={"run_kind": "synthetic-logging-regression"},
        )
        # Only entries outside the logger's 32-value sample have a gradient.
        parameter.grad = torch.cat([torch.zeros(32), torch.ones(32)])
        optimizer.step()
        record = json.loads(events.read_text(encoding="utf-8"))
        assert record["optimizer_step"] == 20000
        assert record["actual_post_step_hook"] is True
        assert record["sampled_parameters_changed"] is False
        assert not torch.equal(parameter, initial)
        result = {
            "synthetic_logging_regression_only": True,
            "actual_production_weights_examined": False,
            "optimizer_step": record["optimizer_step"],
            "sampled_parameters_changed": record["sampled_parameters_changed"],
            "full_tensor_changed": not torch.equal(parameter, initial),
            "conclusion": "per-update sampled equality does not establish full-weight equality",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
