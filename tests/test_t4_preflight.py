"""T4 estimates must use completed updates, checkpoint cost and post-warmup observations."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "t4_preflight", Path(__file__).parents[1] / "tools/t4_preflight.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_completed_updates_and_checkpoint_cost_drive_estimate():
    rows = [
        {
            "actual_optimizer_update": True,
            "update_in_process": i,
            "update_s": 99 if i < 3 else 1.5,
            "dataload_s": 0.5,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        }
        for i in range(1, 5)
    ]
    result = MODULE.summarize(rows, [{"write_and_integrity_s": 10, "checkpoint_bytes": 1000}])
    assert result["steady_steps_per_s"] == 0.5
    assert result["estimated_20000_steps_hours"] == pytest.approx(40200 / 3600)
    assert result["estimated_1000_step_interval_minutes"] == pytest.approx(2010 / 60)
    assert result["estimated_20_checkpoint_bytes"] == 20000
    rows[-1]["actual_optimizer_update"] = False
    with pytest.raises(ValueError, match="completed optimizer"):
        MODULE.summarize(rows, [{"write_and_integrity_s": 10}])


def test_notebook_embeds_companion_and_preserves_production_config():
    import hashlib
    import json

    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/smolvla_finetune_colab.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in cells:
        compile(cell, "notebook", "exec")
    setup = next(c for c in cells if "archive.extractall" in c)
    assert setup.index("MANIFEST_SHA256") < setup.index("archive.extractall")
    assert setup.index("hashlib.file_digest") < setup.index("archive.extractall")
    assert "smolvla-t4-preflight" in next(c for c in cells if "PREFLIGHT_SCRIPT" in c)
    manifest = root / "runs/colab-prep-v3/bundle-status.json"
    if manifest.exists():
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() in setup
