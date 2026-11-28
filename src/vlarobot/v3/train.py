"""Train/evaluate compact V3 waypoint prototype on cached frozen features."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.v3.offline import (
    _phase_position_errors,
    align_samples,
    choose_validation_bases,
    load_metadata_rows,
    load_sources,
    split_by_bases,
)
from vlarobot.v3.policy import CompactWaypointPolicy, PolicyConfig, evaluate_policy, train_policy


def _load_dino_cache(cache_path: Path, samples) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(cache_path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["meta"].tolist()).decode("utf-8"))
        key_to_idx = {(int(k[0]), int(k[1])): i for i, k in enumerate(data["keys"].tolist())}
        idx = [key_to_idx[(s.episode_index, s.frame_index)] for s in samples]
        pooled = data["pooled"][idx].astype(np.float32)
        spatial = data["spatial"][idx].astype(np.float32)
    n, h, w, c = spatial.shape
    spatial_coarse = spatial.reshape(n, 4, h // 4, 4, w // 4, c).mean(axis=(2, 4)).reshape(n, -1)
    return pooled, spatial_coarse.astype(np.float32), meta


def _feature_matrix(samples, pooled, spatial_coarse, mode: str) -> np.ndarray:
    state = np.stack([s.state for s in samples]).astype(np.float32)
    prev = np.stack([s.prev_action for s in samples]).astype(np.float32)
    if mode == "state":
        return state
    if mode == "state_prev_action":
        return np.column_stack([state, prev])
    if mode == "state_prev_action_pretrained_image":
        return np.column_stack([state, prev, pooled, spatial_coarse])
    if mode == "state_pretrained_image":
        return np.column_stack([state, pooled, spatial_coarse])
    raise ValueError(mode)


def run_training(
    dataset_root: Path,
    cache_path: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    epochs: int = 30,
) -> dict[str, Any]:
    sources = load_sources(dataset_root)
    rows = load_metadata_rows(dataset_root)
    samples = align_samples(sources, rows)
    pooled, spatial_coarse, dino_meta = _load_dino_cache(cache_path, samples)
    val_bases = choose_validation_bases(samples)
    train_samples, val_samples = split_by_bases(samples, val_bases)
    train_ids = {id(s) for s in train_samples}
    train_mask = np.array([id(s) in train_ids for s in samples])
    val_mask = ~train_mask

    def pack(mask, mode):
        idx = np.where(mask)[0]
        x = _feature_matrix([samples[i] for i in idx], pooled[idx], spatial_coarse[idx], mode)
        yxyz = np.stack([samples[i].target_xyz for i in idx]).astype(np.float32)
        ygrip = np.array([samples[i].gripper_target for i in idx], dtype=np.float32)
        return idx, x, yxyz, ygrip

    mode = "state_prev_action_pretrained_image"
    tr_idx, x_tr, yxyz_tr, yg_tr = pack(train_mask, mode)
    va_idx, x_va, yxyz_va, yg_va = pack(val_mask, mode)
    cfg = PolicyConfig(input_dim=x_tr.shape[1], seed=seed, epochs=epochs)
    trained = train_policy(x_tr, yxyz_tr, yg_tr, x_va, yxyz_va, yg_va, cfg)
    model: CompactWaypointPolicy = trained["model"]
    pred = model.predict(x_va)
    phase = _phase_position_errors(samples, pred["xyz"], yxyz_va, va_idx)
    overall = evaluate_policy(model, x_va, yxyz_va, yg_va)

    # observational visual-conditioning proxy: correlate predicted xyz delta with can offset delta
    # for same base scene and phase, across episodes with different can offsets.
    can_by_ep = {}
    for ep, src in enumerate(sources):
        off = src["scene"].get("can_xy_yaw_offset", [0.0, 0.0, 0.0])
        can_by_ep[ep] = np.array(off[:2], dtype=float)
    intervention = _visual_sensitivity_proxy(samples, va_idx, pred["xyz"], can_by_ep)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "compact_waypoint_policy.pt")
    result = {
        "status": "measured-offline-prototype",
        "model": "compact_mlp_waypoint_v3",
        "config": asdict(cfg),
        "dataset": str(dataset_root),
        "dino_cache": str(cache_path),
        "dino_meta": dino_meta,
        "split": {
            "validation_bases": list(val_bases),
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
        },
        "overall_validation": overall,
        "validation_by_phase": phase,
        "visual_sensitivity_proxy": intervention,
        "history_tail": trained["history"][-5:],
        "notes": [
            "Prototype trained on existing 80-demo TRAIN dataset only; no holdout.",
            "Prev-action is causally available previous frame action.",
            "Visual sensitivity is an observational proxy over existing can offsets, "
            "not a physically rendered counterfactual.",
        ],
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _visual_sensitivity_proxy(samples, val_idx, pred_xyz, can_by_ep) -> dict[str, Any]:
    # Compare pairs in same base scene and phase with different can xy offsets.
    from collections import defaultdict

    buckets = defaultdict(list)
    for pos, idx in enumerate(val_idx.tolist()):
        s = samples[idx]
        buckets[(s.base_scene, s.phase)].append((pos, idx))
    dx, dy, dpredx, dpredy = [], [], [], []
    for items in buckets.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pi, ii = items[i]
                pj, ij = items[j]
                ci = can_by_ep.get(samples[ii].episode_index)
                cj = can_by_ep.get(samples[ij].episode_index)
                if ci is None or cj is None:
                    continue
                dcan = cj - ci
                if np.linalg.norm(dcan) < 1e-6:
                    continue
                dp = pred_xyz[pj, :2] - pred_xyz[pi, :2]
                dx.append(float(dcan[0]))
                dy.append(float(dcan[1]))
                dpredx.append(float(dp[0]))
                dpredy.append(float(dp[1]))
    if len(dx) < 5:
        return {"status": "insufficient_pairs", "pairs": len(dx)}
    dx, dy, dpredx, dpredy = map(lambda a: np.array(a), (dx, dy, dpredx, dpredy))

    def corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "status": "measured_proxy",
        "pairs": int(len(dx)),
        "corr_dcanx_dpredx": corr(dx, dpredx),
        "corr_dcany_dpredy": corr(dy, dpredy),
        "mean_abs_dpred_xy_m": round(float(np.mean(np.abs(np.stack([dpredx, dpredy], 1)))), 6),
        "mean_abs_dcan_xy_m": round(float(np.mean(np.abs(np.stack([dx, dy], 1)))), 6),
    }
