"""Warm-start an approved TRAIN SmolVLA export using transition-balanced windows."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import tempfile
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from vlarobot.vla.checkpoint_export import load_portable_policy, require_trained_export
from vlarobot.vla.padding import install_padding_alias
from vlarobot.vla.schema import CAMERA_KEY
from vlarobot.vla.training import validate_training_approval


def transition_starts(root: Path, horizon: int) -> tuple[list[int], list[int]]:
    """Select TRAIN observation indices whose full window contains one endpoint switch."""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "action"])
        rows.extend((r["index"], r["episode_index"], r["action"][6]) for r in table.to_pylist())
    rows.sort()
    if [r[0] for r in rows] != list(range(len(rows))):
        raise ValueError("TRAIN dataset indices are not contiguous")
    low, high = 0.0, float(np.float32(0.045))
    if {r[2] for r in rows} != {low, high}:
        raise ValueError("endpoint warm-start requires the approved binary motor labels")
    index = np.asarray([r[0] for r in rows])
    episode = np.asarray([r[1] for r in rows])
    values = np.asarray([r[2] for r in rows])
    close, release = [], []
    for i in np.flatnonzero((values[1:] != values[:-1]) & (episode[1:] == episode[:-1])) + 1:
        group = release if values[i] == high else close
        for start in range(max(0, int(i) - horizon + 1), int(i) + 1):
            if (
                start + horizon <= len(rows)
                and episode[start] == episode[start + horizon - 1] == episode[i]
            ):
                group.append(int(index[start]))
    if not close or not release:
        raise ValueError("both grasp and release TRAIN windows are required")
    return sorted(set(close)), sorted(set(release))


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> None:
    """Train with intact normalization, native padding, measured loss and resumable exports."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import default_collate

    if args.steps < 1 or args.batch < 4 or args.save_every < 1:
        raise ValueError(
            "positive steps and save cadence and at least four samples per batch required"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("a CUDA runtime is required for this Colab profile")
    torch.set_num_threads(4)
    approval = validate_training_approval(args.dataset)
    parent = require_trained_export(args.source)
    for key in ("stats_sha256", "source_provenance_sha256"):
        if approval[key] != parent[key]:
            raise ValueError("recovery dataset differs from the source checkpoint")
    source_manifest = json.loads((args.source / "export-manifest.json").read_text(encoding="utf-8"))
    start_step = int(source_manifest["checkpoint_step"])
    lr = source_manifest["training_evidence"]["learning_rates"][-1]
    candidates = sorted(
        p for p in args.out.glob("step-*") if p.is_dir() and (p / "export-manifest.json").exists()
    )
    if args.out.exists() and not args.resume:
        raise FileExistsError("preserve an existing recovery run or explicitly resume")
    if args.resume and not candidates:
        raise FileNotFoundError("no complete checkpoint to resume; use a fresh output directory")
    state = None
    model_source = candidates[-1] if candidates else args.source
    if candidates:
        require_trained_export(model_source)
        with (model_source / "recovery-optimizer.pt").open("rb") as checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state["parent_manifest"] != _hash(args.source / "export-manifest.json")
            or state["batch"] != args.batch
        ):
            raise ValueError("recovery checkpoint identity differs")
    policy, pre, _ = load_portable_policy(model_source, device=args.device)
    install_padding_alias(type(policy))
    horizon = policy.config.chunk_size
    data = LeRobotDataset(
        json.loads((args.dataset / "source-provenance.json").read_text(encoding="utf-8"))[
            "repo_id"
        ],
        root=args.dataset,
        video_backend="pyav",
        delta_timestamps={"action": [i / 31.25 for i in range(horizon)]},
    )
    if data.fps != 31.25 and data.fps != 31:
        raise ValueError(f"unexpected TRAIN recording rate: {data.fps!r}")
    close, release = transition_starts(args.dataset, horizon)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=lr,
        betas=tuple(policy.config.optimizer_betas),
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    update = 0
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        update = state["update"]
    if update >= args.steps:
        raise ValueError("requested update already completed")
    args.out.mkdir(parents=True, exist_ok=True)
    policy.train()
    started = perf_counter()
    for step in range(update + 1, args.steps + 1):
        torch.manual_seed(step)
        rng = np.random.default_rng(step)
        per_transition = args.batch // 4
        indices = [
            *rng.choice(close, per_transition),
            *rng.choice(release, per_transition),
            *rng.integers(0, len(data), args.batch - 2 * per_transition),
        ]
        fields = ("observation.state", CAMERA_KEY, "task", "action", "action_is_pad")
        samples = [data[int(i)] for i in indices]
        batch = pre(default_collate([{k: sample[k] for k in fields} for sample in samples]))
        _, metrics = policy.forward(batch)
        terms = metrics["losses_after_rm_padding"][:, :, :7]
        valid = ~batch["action_is_pad"]
        scale = valid.sum()
        if not scale:
            raise ValueError("training batch contains no valid horizon labels")
        pose_loss = terms[:, :, :6].sum() / (6 * scale)
        grip_loss = terms[:, :, 6].sum() / scale
        loss = pose_loss + grip_loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite recovery loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), policy.config.optimizer_grad_clip_norm, error_if_nonfinite=True
        )
        calls = []
        handle = optimizer.register_step_post_hook(lambda *_, recorded=calls: recorded.append(True))
        try:
            optimizer.step()
        finally:
            handle.remove()
        if calls != [True]:
            raise RuntimeError("recovery optimizer step did not complete")
        record = {
            "run_kind": "smolvla-finetune",
            "profile": "train_transition_balanced_warmstart_v1",
            "actual_post_step_hook": True,
            "optimizer_step": start_step + step,
            "recovery_update": step,
            "stats_sha256": approval["stats_sha256"],
            "source_provenance_sha256": approval["source_provenance_sha256"],
            "parent_manifest": _hash(args.source / "export-manifest.json"),
            "learning_rates": [lr],
            "pose_loss": float(pose_loss.detach()),
            "gripper_loss": float(grip_loss.detach()),
            "gradient_norm": float(norm),
            "elapsed_s": perf_counter() - started,
            "dataset_indices": [int(i) for i in indices],
        }
        with (args.out / "optimizer-steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if step == args.steps or step % args.save_every == 0:
            target = args.out / f"step-{start_step + step:06d}"
            with tempfile.TemporaryDirectory(dir=args.out) as scratch:
                export = Path(scratch) / "export"
                shutil.copytree(args.source, export, copy_function=shutil.copyfile)
                # copytree inherits the immutable source directory mode.
                export.chmod(export.stat().st_mode | stat.S_IWUSR)
                weights = Path(scratch) / "weights"
                policy.save_pretrained(weights)
                shutil.copyfile(weights / "model.safetensors", export / "model.safetensors")
                shutil.copyfile(
                    args.out / "optimizer-steps.jsonl", export / "optimizer-steps.jsonl"
                )
                with (export / "recovery-optimizer.pt").open("wb") as checkpoint:
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "update": step,
                            "parent_manifest": record["parent_manifest"],
                            "batch": args.batch,
                        },
                        checkpoint,
                    )
                manifest = {
                    "checkpoint_step": start_step + step,
                    "training_evidence": record,
                    "files_sha256": {
                        f.relative_to(export).as_posix(): _hash(f)
                        for f in export.rglob("*")
                        if f.is_file() and f.name != "export-manifest.json"
                    },
                }
                (export / "export-manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                require_trained_export(export)
                export.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
