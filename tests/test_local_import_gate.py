"""Frozen final-export contract checks require no policy loading or GPU."""

import importlib

import pytest


def test_frozen_contract_and_wrong_camera_fail_closed(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/recovery"))
    gate = importlib.import_module("local_import_gate")
    config = {
        "input_features": {
            "observation.state": {"shape": [8]},
            "observation.images.camera1": {"shape": [3, 480, 640]},
        },
        "output_features": {"action": {"shape": [7]}},
        "chunk_size": 50,
        "n_action_steps": 50,
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    }
    saved = {"steps": 20000, "batch_size": 8, "seed": 0, "save_freq": 1000}
    gate.check_config(config, saved)
    with pytest.raises(ValueError, match="setting"):
        gate.check_config(config, {**saved, "steps": 17000})
    config["input_features"]["observation.images.wrong"] = config["input_features"].pop(
        "observation.images.camera1"
    )
    with pytest.raises(ValueError, match="camera"):
        gate.check_config(config, saved)
