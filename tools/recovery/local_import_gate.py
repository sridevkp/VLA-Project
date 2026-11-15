"""Require final-step contract and provenance before the existing CPU inference preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from checkpoint_guard import verify_export


def check_config(config: dict[str, Any], saved: dict[str, Any]) -> None:
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    if (
        set(inputs) != {"observation.state", "observation.images.camera1"}
        or inputs["observation.state"]["shape"] != [8]
        or inputs["observation.images.camera1"]["shape"] != [3, 480, 640]
        or set(outputs) != {"action"}
        or outputs["action"]["shape"] != [7]
        or config.get("chunk_size") != 50
        or config.get("n_action_steps") != 50
        or config.get("normalization_mapping")
        != {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    ):
        raise ValueError(
            "final export differs from frozen state8/action7/camera/chunk/normalization"
        )
    for key, value in {"steps": 20000, "batch_size": 8, "seed": 0, "save_freq": 1000}.items():
        if saved.get(key) != value:
            raise ValueError(f"frozen saved training setting differs: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "src"))
    from vlarobot.vla.checkpoint_export import require_trained_export
    from vlarobot.vla.training import CHECKPOINT_FILES

    manifest = verify_export(args.export, CHECKPOINT_FILES, 20000)
    approval = require_trained_export(args.export)
    evidence = manifest["training_evidence"]
    if evidence["optimizer_step"] != 20000 or len(approval["episodes"]) != 80:
        raise ValueError("final20000 evidence and exact approved80 dataset required")
    check_config(
        json.loads((args.export / "config.json").read_text(encoding="utf-8")),
        json.loads((args.export / "train_config.json").read_text(encoding="utf-8")),
    )
    subprocess.run(
        [
            sys.executable,
            str(args.project / "runs/week2-progress-v1/validate-import.py"),
            "--export",
            str(args.export),
            "--out",
            str(args.out),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
