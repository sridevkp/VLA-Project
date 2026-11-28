"""Object-conditioned waypoint policy: predicted CAN localization + causal state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.v3.localization import _ridge_fit, _ridge_pred, build_localization_dataset
from vlarobot.v3.offline import (
    _phase_position_errors,
    choose_validation_bases,
    fit_ridge,
    mean_abs_error,
    predict_ridge,
    split_by_bases,
)
from vlarobot.v3.train import _pca_fit_transform, _fit_gripper_logistic


def _oof_can_predictions(samples, X_loc, y_can):
    pred = np.zeros_like(y_can)
    bases = sorted({s.base_scene for s in samples})
    for base in bases:
        va = np.array([s.base_scene == base for s in samples])
        tr = ~va
        m = _ridge_fit(X_loc[tr], y_can[tr])
        pred[va] = _ridge_pred(m, X_loc[va])
    return pred


def run_object_conditioned_eval(dataset_root: Path, cache_path: Path, out: Path) -> dict[str, Any]:
    samples, pooled, spatial, y_can, idx, dino_meta = build_localization_dataset(dataset_root, cache_path)
    samples = [samples[i] for i in idx.tolist()]
    val_bases = choose_validation_bases(samples)
    tr_mask = np.array([s.base_scene not in set(val_bases) for s in samples])
    va_mask = ~tr_mask
    state = np.stack([s.state for s in samples]).astype(np.float32)
    prev = np.stack([s.prev_action for s in samples]).astype(np.float32)
    prev_no_grip = prev.copy(); prev_no_grip[:, 6] = 0.0
    targets = np.stack([s.target_xyz for s in samples]).astype(np.float32)
    yg = np.array([s.gripper_target for s in samples], dtype=np.float32)
    s4 = spatial.astype(np.float32)
    pca64 = _pca_fit_transform(s4, tr_mask, 64)
    X_loc = np.concatenate([state, pca64], axis=1)
    # OOF predicted CAN for training, fold-fit for val
    can_oof = _oof_can_predictions(samples, X_loc, y_can)
    can_loc_model = _ridge_fit(X_loc[tr_mask], y_can[tr_mask])
    can_pred = can_oof.copy()
    can_pred[va_mask] = _ridge_pred(can_loc_model, X_loc[va_mask])

    # Object-relative target: waypoint - predicted CAN (learned from predicted CAN)
    rel_target = targets - can_pred

    # Features for waypoint policy
    X_base = np.concatenate([state, prev_no_grip, can_pred], axis=1)
    X_vis = np.concatenate([X_base, pca64], axis=1)

    # Baselines
    tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
    rows = []
    # A: absolute waypoint from state+prev
    m = fit_ridge(np.concatenate([state,prev_no_grip],axis=1)[tr_idx], targets[tr_idx])
    p = predict_ridge(m, np.concatenate([state,prev_no_grip],axis=1)[va_idx])
    rows.append({"name":"state_prev_absolute","val":mean_abs_error(p,targets[va_idx]),"phase":_phase_position_errors(samples,p,targets[va_idx],va_idx)})
    # B: absolute with predicted can
    m = fit_ridge(X_base[tr_idx], targets[tr_idx])
    p = predict_ridge(m, X_base[va_idx])
    rows.append({"name":"state_prev_predcan_absolute","val":mean_abs_error(p,targets[va_idx]),"phase":_phase_position_errors(samples,p,targets[va_idx],va_idx)})
    # C: object-relative with predicted can (predict rel, reconstruct abs)
    m = fit_ridge(X_base[tr_idx], rel_target[tr_idx])
    p_rel = predict_ridge(m, X_base[va_idx])
    p = p_rel + can_pred[va_idx]
    rows.append({"name":"state_prev_predcan_object_relative","val":mean_abs_error(p,targets[va_idx]),"phase":_phase_position_errors(samples,p,targets[va_idx],va_idx)})
    # D: object-relative + visual
    m = fit_ridge(X_vis[tr_idx], rel_target[tr_idx])
    p_rel = predict_ridge(m, X_vis[va_idx])
    p = p_rel + can_pred[va_idx]
    rows.append({"name":"state_prev_predcan_pca64_object_relative","val":mean_abs_error(p,targets[va_idx]),"phase":_phase_position_errors(samples,p,targets[va_idx],va_idx)})

    # gripper without prev grip
    Xg = np.concatenate([state, prev_no_grip, pca64, can_pred], axis=1)
    grip = _fit_gripper_logistic(Xg[tr_idx], yg[tr_idx], Xg[va_idx], yg[va_idx])
    from vlarobot.v3.policy import evaluate_policy, CompactWaypointPolicy, PolicyConfig
    dummy = CompactWaypointPolicy(PolicyConfig(input_dim=1))
    logit = grip["predict_logit"](Xg[va_idx])
    grip_metrics = evaluate_policy(dummy, np.zeros((len(va_idx),1)), targets[va_idx], yg[va_idx], pred={"xyz":p,"gripper_logit":logit})

    report = {
        "status": "measured",
        "n_samples": len(samples),
        "n_train": int(tr_mask.sum()),
        "n_val": int(va_mask.sum()),
        "val_bases": val_bases,
        "localization_val_mae_m": float(np.linalg.norm(can_pred[va_mask]-y_can[va_mask],axis=1).mean()),
        "rows": rows,
        "gripper_no_prev_grip": {k: grip_metrics[k] for k in ["open_precision","open_recall","closed_precision","closed_recall","confusion","gripper_accuracy"]},
        "note": "OOF predicted CAN used for training; val uses train-fold localization model. No GT CAN at runtime.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
