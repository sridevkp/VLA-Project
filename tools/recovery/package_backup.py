"""Package the read-only Linux golden 017000 backup; never invoke training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from checkpoint_guard import sha256, verify_checkpoint
from durable_sink import archive_tree

MODEL_SHA256 = "d8ef2562ce5579210352cfe7c208cc31c7f6f63499e835264ee11a9c735f847b"
OPTIMIZER_SHA256 = "8520369d01d0af9f7399db8fc372e4f856006aa59c1e7ec14093fdbf508974c9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project / "src"))
    from vlarobot.vla.training import CHECKPOINT_FILES

    record = verify_checkpoint(args.checkpoint, CHECKPOINT_FILES, expected_step=17000)
    pins = {
        "pretrained_model/model.safetensors": MODEL_SHA256,
        "training_state/optimizer_state.safetensors": OPTIMIZER_SHA256,
    }
    if any(record["files_sha256"].get(n) != h for n, h in pins.items()):
        raise ValueError("backup differs from owner-verified original 017000 tensors")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    archive_tree(args.checkpoint, args.out)
    report = {
        "step": 17000,
        "sha256": sha256(args.out),
        "bytes": args.out.stat().st_size,
        "source_complete_sha256": sha256(args.checkpoint / "COMPLETE.json"),
        "files_sha256": record["files_sha256"],
        "source_modified": False,
    }
    with args.out.with_suffix(".sha256.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
