"""Lightweight CAN localization from frozen DINOv2 features (offline GT labels only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.v3.offline import (
    align_samples,
    base_scene_name,
    choose_validation_bases,
    load_metadata_rows,
    load_sources,
    split_by_bases,
)
from vlarobot.v3.train import _load_dino_cache, _pca_fit_transform


def _can_base_at_frame(raw_dir: Path, frame_t: float) -> np.ndarray | None:
    best = None
    best_dt = 1e9
    with (raw_dir / "policy.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r.get("sim_time_s")
            can = r.get("can_base_xyz_m")
            if t is None or can is None:
                continue
            dt = abs(float(t) - frame_t)
            if dt < best_dt:
                best_dt = dt
                best = np.array(can, dtype=np.float32)
    return best


def build_localization_dataset(dataset_root: Path, cache_path: Path):
    sources = load_sources(dataset_root)
    rows = load_metadata_rows(dataset_root)
    samples = align_samples(sources, rows)
    pooled, spatial, meta = _load_dino_cache(cache_path, samples)
    y = []
    keep = []
    for i, s in enumerate(samples):
        src = sources[s.episode_index]
        can = _can_base_at_frame(Path(src["raw_dir"]), s.t)
        if can is None:
            continue
        y.append(can)
        keep.append(i)
    idx = np.array(keep, dtype=int)
    return samples, pooled[idx], spatial[idx], np.stack(y), idx, meta


def _ridge_fit(Xtr, ytr, alpha=1.0):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xn = (Xtr - mu) / sd
    A = np.concatenate([np.ones((len(Xn), 1)), Xn], axis=1)
    I = np.eye(A.shape[1])
    I[0, 0] = 0.0
    W = np.linalg.solve(A.T @ A + alpha * I, A.T @ ytr)
    return {"mu": mu, "sd": sd, "W": W}


def _ridge_pred(model, X):
    Xn = (X - model["mu"]) / model["sd"]
    A = np.concatenate([np.ones((len(Xn), 1)), Xn], axis=1)
    return A @ model["W"]


def _metrics(pred, gt):
    err = pred - gt
    ae = np.abs(err)
    eu = np.linalg.norm(err, axis=1)
    return {
        "mae_x_m": float(ae[:, 0].mean()),
        "mae_y_m": float(ae[:, 1].mean()),
        "mae_z_m": float(ae[:, 2].mean()),
        "mae_euclidean_m": float(eu.mean()),
        "median_euclidean_m": float(np.median(eu)),
    }


def run_localization_eval(dataset_root: Path, cache_path: Path, out: Path) -> dict[str, Any]:
    samples, pooled, spatial, y, idx, dino_meta = build_localization_dataset(dataset_root, cache_path)
    samples = [samples[i] for i in idx.tolist()]
    val_bases = choose_validation_bases(samples)
    train, val = split_by_bases(samples, val_bases)
    tr_mask = np.array([s.base_scene not in set(val_bases) for s in samples])
    va_mask = ~tr_mask
    state = np.stack([s.state for s in samples]).astype(np.float32)
    # features
    # spatial is already 4x4 coarse flattened [N, 6144] from _load_dino_cache
    s4 = spatial.astype(np.float32)
    pca64 = _pca_fit_transform(s4, tr_mask, 64)

    feats = {
        "state_only": state,
        "dino_cls": pooled,
        "dino_spatial4x4_pca64": pca64,
        "state_plus_dino_pca64": np.concatenate([state, pca64], axis=1),
    }
    results = {}
    for name, X in feats.items():
        m = _ridge_fit(X[tr_mask], y[tr_mask])
        pred = _ridge_pred(m, X[va_mask])
        results[name] = _metrics(pred, y[va_mask])

    # displacement response: pairwise same base scene, different can y/x
    # use GT can delta vs predicted delta
    by_base = {}
    for i, s in enumerate(samples):
        if va_mask[i]:
            by_base.setdefault(s.base_scene, []).append(i)
    dx_gt, dy_gt, dx_pr, dy_pr = [], [], [], []
    m = _ridge_fit(feats["dino_spatial4x4_pca64"][tr_mask], y[tr_mask])
    pred_all = _ridge_pred(m, feats["dino_spatial4x4_pca64"])
    for base, ids in by_base.items():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                gt_d = y[j] - y[i]
                pr_d = pred_all[j] - pred_all[i]
                dx_gt.append(gt_d[0]); dy_gt.append(gt_d[1])
                dx_pr.append(pr_d[0]); dy_pr.append(pr_d[1])
    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    displacement = {
        "n_pairs": len(dx_gt),
        "corr_dx": corr(dx_gt, dx_pr),
        "corr_dy": corr(dy_gt, dy_pr),
    }

    report = {
        "status": "measured",
        "n_samples": len(samples),
        "n_train": int(tr_mask.sum()),
        "n_val": int(va_mask.sum()),
        "val_bases": val_bases,
        "dino_meta": dino_meta,
        "results": results,
        "displacement_response": displacement,
        "note": "GT CAN positions used as offline training labels/eval refs only.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
