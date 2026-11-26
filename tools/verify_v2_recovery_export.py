"""Verify a recovered V2 SmolVLA export and emit a provenance/telemetry receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from vlarobot.vla.checkpoint_export import require_trained_export

PARENT_MANIFEST = "5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1"
STATS_SHA256 = "62669ddd180ccd830bf3403e47104ed7d9241015b2f4017620fad87e33b554f6"
SOURCE_PROVENANCE_SHA256 = "de7aff693700bbb0c8f0abb0b5a41564e52a6d1b17fb838f4590cef39564f602"
PROFILE = "train_transition_balanced_warmstart_v1"
UPDATES = 1000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(export: Path, expected_step: int = 21000) -> dict[str, Any]:
    """Check production-loader acceptance, provenance and all optimizer records."""
    approval = require_trained_export(export)
    manifest_path = export / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = manifest["training_evidence"]
    if manifest["checkpoint_step"] != expected_step:
        raise ValueError("unexpected recovered checkpoint step")
    expected_evidence = {
        "run_kind": "smolvla-finetune",
        "profile": PROFILE,
        "actual_post_step_hook": True,
        "optimizer_step": expected_step,
        "recovery_update": UPDATES,
        "parent_manifest": PARENT_MANIFEST,
        "stats_sha256": STATS_SHA256,
        "source_provenance_sha256": SOURCE_PROVENANCE_SHA256,
    }
    for key, value in expected_evidence.items():
        if evidence.get(key) != value:
            raise ValueError(f"recovered training evidence mismatch: {key}")
    records = [
        json.loads(line)
        for line in (export / "optimizer-steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != UPDATES:
        raise ValueError("optimizer log does not contain exactly 1000 updates")
    if [row["recovery_update"] for row in records] != list(range(1, UPDATES + 1)):
        raise ValueError("recovery updates are missing or out of order")
    if [row["optimizer_step"] for row in records] != list(
        range(expected_step - UPDATES + 1, expected_step + 1)
    ):
        raise ValueError("optimizer global steps do not follow the source checkpoint")
    for row in records:
        if row["actual_post_step_hook"] is not True or row["parent_manifest"] != PARENT_MANIFEST:
            raise ValueError("optimizer step lacks an accepted hook or correct parent")
        if not all(
            math.isfinite(float(row[key]))
            for key in ("pose_loss", "gripper_loss", "gradient_norm", "elapsed_s")
        ):
            raise ValueError("nonfinite recovery telemetry")
        if not row["dataset_indices"] or any(
            not isinstance(index, int) or index < 0 for index in row["dataset_indices"]
        ):
            raise ValueError("invalid TRAIN dataset index in optimizer telemetry")
    if records[-1] != evidence:
        raise ValueError("manifest evidence is not the final optimizer record")

    receipt: dict[str, Any] = {
        "status": "verified",
        "run_kind": "smolvla-v2-recovery-export",
        "profile": PROFILE,
        "export_path": str(export),
        "checkpoint_step": manifest["checkpoint_step"],
        "export_manifest_sha256": _sha256(manifest_path),
        "model_sha256": manifest["files_sha256"]["model.safetensors"],
        "parent_manifest_sha256": PARENT_MANIFEST,
        "stats_sha256": STATS_SHA256,
        "source_provenance_sha256": SOURCE_PROVENANCE_SHA256,
        "approved_train_episodes": len(approval["episodes"]),
        "optimizer_updates": len(records),
        "first_optimizer_step": records[0]["optimizer_step"],
        "last_optimizer_step": records[-1]["optimizer_step"],
        "learning_rate": evidence["learning_rates"][-1],
        "first_pose_loss": records[0]["pose_loss"],
        "last_pose_loss": records[-1]["pose_loss"],
        "first_gripper_loss": records[0]["gripper_loss"],
        "last_gripper_loss": records[-1]["gripper_loss"],
        "final_gradient_norm": records[-1]["gradient_norm"],
        "elapsed_s": records[-1]["elapsed_s"],
        "production_loader": "require_trained_export",
    }
    try:
        import torch

        with (export / "recovery-optimizer.pt").open("rb") as stream:
            state = torch.load(stream, map_location="cpu", weights_only=True)
        if (
            state["update"] != UPDATES
            or state["parent_manifest"] != PARENT_MANIFEST
            or state["batch"] != 8
        ):
            raise ValueError("recovery optimizer state identity mismatch")
        receipt["optimizer_state_reloaded"] = True
        receipt["optimizer_state_update"] = state["update"]
        receipt["optimizer_state_batch"] = state["batch"]
    except ImportError:
        receipt["optimizer_state_reloaded"] = False
        receipt["optimizer_state_note"] = "torch unavailable in this interpreter"
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=21000)
    args = parser.parse_args()
    receipt = verify(args.export, expected_step=args.expected_step)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
