"""Moving a checkpoint must also carry tokenizer metadata and detect changed exports."""

import json

import pytest
from test_vla_training import make_checkpoint

from vlarobot.vla.checkpoint_export import export_complete_checkpoint, load_portable_policy
from vlarobot.vla.training import mark_checkpoint_complete


def test_export_carries_external_metadata_and_refuses_overwrite_or_modified_files(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "training", 1000)
    processor = tmp_path / "external-processor"
    processor.mkdir()
    (processor / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pretrained_model/config.json").write_text(
        json.dumps({"vlm_model_name": str(processor)}), encoding="utf-8"
    )
    mark_checkpoint_complete(checkpoint)
    out = export_complete_checkpoint(tmp_path / "training", tmp_path / "export")
    assert (out / "processor/config.json").is_file()
    assert not any(p.is_symlink() for p in out.rglob("*"))
    with pytest.raises(FileExistsError):
        export_complete_checkpoint(tmp_path / "training", out)
    # Verification runs before importing any optional ML dependency.
    (out / "processor/config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_portable_policy(out)
