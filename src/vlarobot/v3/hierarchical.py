"""Hierarchical learned task-intent/progression policy (V3 recovery).

Offline supervision uses expert phase annotations only as training labels.
Runtime predicts intent; no expert phase labels, privileged GT, hardcoded
task transitions, or scripted task-level fallback at deployment.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.v3.localization import _ridge_fit, _ridge_pred
from vlarobot.v3.offline import (
    align_samples,
    base_scene_name,
    choose_validation_bases,
    fit_ridge,
    load_metadata_rows,
    load_sources,
    mean_abs_error,
    predict_ridge,
    split_by_bases,
)
from vlarobot.v3.train import _load_dino_cache, _pca_fit_transform


PHASES = [
    "INIT",
    "PREGRASP_REALIGN",
    "PREGRASP",
    "DESCEND",
    "LIFT",
    "TRANSIT",
    "PLACE_DESCEND",
    "RELEASE",
    "RETREAT",
    "HOME",
    "DONE",
]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASES)}
ID_TO_PHASE = {i: p for p, i in PHASE_TO_ID.items()}


@dataclass
class HierSample:
    episode_index: int
    scene: str
    base_scene: str
    family: str
    t: float
    frame_index: int
    state: np.ndarray        # (8,)
    action: np.ndarray       # (7,) expert action at frame
    prev_action: np.ndarray  # (7,) causally available previous frame action
    target_xyz: np.ndarray   # (3,) commanded waypoint xyz
    target_rpy: np.ndarray   # (3,) orientation delta from action[3:6]
    gripper_target: float    # action[6]
    phase: str
    phase_id: int


def build_hier_samples(dataset_root: Path) -> list[HierSample]:
    sources = load_sources(dataset_root)
    rows = load_metadata_rows(dataset_root)
    ws = align_samples(sources, rows)
    out = []
    for s in ws:
        phase = s.phase if s.phase in PHASE_TO_ID else "INIT"
        out.append(HierSample(
            episode_index=s.episode_index, scene=s.scene, base_scene=s.base_scene,
            family=s.family, t=s.t, frame_index=s.frame_index,
            state=s.state.astype(np.float32), action=s.action.astype(np.float32),
            prev_action=s.prev_action.astype(np.float32),
            target_xyz=s.target_xyz.astype(np.float32),
            target_rpy=s.action[3:6].astype(np.float32),
            gripper_target=float(s.gripper_target),
            phase=phase, phase_id=PHASE_TO_ID[phase],
        ))
    return out, sources


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(X, y, n_classes, *, lr=0.1, steps=3000, l2=1e-3, seed=0, class_weight=None):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    Xn = (X - mu) / sd
    n, d = Xn.shape
    W = rng.normal(0, 0.01, size=(d, n_classes)).astype(np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    sw = np.ones(n, dtype=np.float64)
    if class_weight == 'balanced':
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        cw = n / (n_classes * np.maximum(counts, 1))
        sw = cw[y]
    sw = sw / sw.mean()
    for _ in range(steps):
        P = _softmax(Xn @ W + b)
        err = (P - Y) * sw[:, None]
        gW = (Xn.T @ err) / n + l2 * W
        gb = err.mean(0)
        W -= lr * gW
        b -= lr * gb
    return {'W': W, 'b': b, 'mu': mu, 'sd': sd, 'n_classes': n_classes}

def predict_softmax(model, X):
    Xn = (X - model['mu']) / model['sd']
    logits = Xn @ model['W'] + model['b']
    return _softmax(logits), logits


def fit_ridge_local(X, Y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    Xn = (X - mu) / sd
    A = np.column_stack([np.ones(len(Xn)), Xn])
    pen = alpha * np.eye(A.shape[1])
    pen[0, 0] = 0.0
    W = np.linalg.solve(A.T @ A + pen, A.T @ Y)
    return {'W': W, 'mu': mu, 'sd': sd}

def predict_ridge_local(model, X):
    Xn = (X - model['mu']) / model['sd']
    A = np.column_stack([np.ones(len(Xn)), Xn])
    return A @ model['W']


def _oof_can_predictions(samples, X_loc, y_can):
    pred = np.zeros_like(y_can)
    bases = sorted({s.base_scene for s in samples})
    for base in bases:
        va = np.array([s.base_scene == base for s in samples])
        tr = ~va
        if tr.sum() == 0 or va.sum() == 0:
            continue
        m = _ridge_fit(X_loc[tr], y_can[tr])
        pred[va] = _ridge_pred(m, X_loc[va])
    return pred


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


def build_hier_dataset(dataset_root: Path, cache_path: Path):
    samples, sources = build_hier_samples(dataset_root)
    pooled, spatial, dino_meta = _load_dino_cache(cache_path, samples)
    val_bases = choose_validation_bases(samples)
    train_s, val_s = split_by_bases(samples, val_bases)
    train_ids = {id(s) for s in train_s}
    tr_mask = np.array([id(s) in train_ids for s in samples])
    va_mask = ~tr_mask

    pca64 = _pca_fit_transform(spatial, tr_mask, 64)
    state = np.stack([s.state for s in samples]).astype(np.float32)
    prev = np.stack([s.prev_action for s in samples]).astype(np.float32)
    target_xyz = np.stack([s.target_xyz for s in samples]).astype(np.float32)
    target_rpy = np.stack([s.target_rpy for s in samples]).astype(np.float32)
    grip = np.array([s.gripper_target for s in samples], dtype=np.float32)
    phase_id = np.array([s.phase_id for s in samples], dtype=np.int64)
    done = (phase_id == PHASE_TO_ID["DONE"]).astype(np.float32)

    # CAN GT for offline training labels only
    y_can = []
    for s in samples:
        c = _can_base_at_frame(Path(sources[s.episode_index]['raw_dir']), s.t)
        y_can.append(c if c is not None else np.zeros(3, dtype=np.float32))
    y_can = np.stack(y_can).astype(np.float32)
    can_valid = np.array([
        _can_base_at_frame(Path(sources[s.episode_index]['raw_dir']), s.t) is not None
        for s in samples
    ])

    # Localization: compare modes; use visual-only by default for deployment robustness.
    X_loc_vis = pca64
    X_loc_state = state
    X_loc_fused = np.concatenate([state, pca64], axis=1)
    can_oof_vis = _oof_can_predictions(samples, X_loc_vis, y_can)
    can_oof_state = _oof_can_predictions(samples, X_loc_state, y_can)
    can_oof_fused = _oof_can_predictions(samples, X_loc_fused, y_can)
    can_mean = y_can[can_valid].mean(0) if can_valid.any() else np.zeros(3, np.float32)
    for arr in (can_oof_vis, can_oof_state, can_oof_fused):
        arr[~can_valid] = can_mean
    can_oof = can_oof_vis  # deployment default: robust on failed live trajectory

    X_base = np.concatenate([state, prev, can_oof, pca64], axis=1).astype(np.float32)
    X_noprev = np.concatenate([state, can_oof, pca64], axis=1).astype(np.float32)

    return {
        'samples': samples, 'sources': sources, 'dino_meta': dino_meta,
        'val_bases': val_bases, 'tr_mask': tr_mask, 'va_mask': va_mask,
        'X_base': X_base, 'X_noprev': X_noprev,
        'X_loc_vis': X_loc_vis, 'X_loc_state': X_loc_state, 'X_loc_fused': X_loc_fused,
        'pca64': pca64, 'state': state, 'prev': prev,
        'can_oof': can_oof, 'can_oof_vis': can_oof_vis, 'can_oof_state': can_oof_state, 'can_oof_fused': can_oof_fused,
        'y_can': y_can,
        'target_xyz': target_xyz, 'target_rpy': target_rpy, 'grip': grip,
        'phase_id': phase_id, 'done': done, 'can_valid': can_valid,
    }


def _onehot(y, n):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def _grip_metrics(pred_open, true_open):
    tp = int(np.sum((pred_open==1)&(true_open==1)))
    fp = int(np.sum((pred_open==1)&(true_open==0)))
    fn = int(np.sum((pred_open==0)&(true_open==1)))
    tn = int(np.sum((pred_open==0)&(true_open==0)))
    def pr(p,r): return float(p), float(r)
    prec_open = tp/(tp+fp) if tp+fp else 0.0
    rec_open = tp/(tp+fn) if tp+fn else 0.0
    prec_closed = tn/(tn+fn) if tn+fn else 0.0
    rec_closed = tn/(tn+fp) if tn+fp else 0.0
    return {
        'accuracy': float(np.mean(pred_open==true_open)),
        'open_precision': prec_open, 'open_recall': rec_open,
        'closed_precision': prec_closed, 'closed_recall': rec_closed,
        'confusion': {'tp_open':tp,'fp_open':fp,'fn_open':fn,'tn_open':tn},
    }


def train_hierarchical(dataset_root: Path, cache_path: Path, out_dir: Path) -> dict[str, Any]:
    ds = build_hier_dataset(dataset_root, cache_path)
    samples = ds['samples']
    tr, va = ds['tr_mask'], ds['va_mask']
    X, Xn = ds['X_base'], ds['X_noprev']
    phase_id = ds['phase_id']
    target_xyz = ds['target_xyz']
    target_rpy = ds['target_rpy']
    grip = ds['grip']
    done = ds['done']

    n_classes = len(PHASES)
    # Intent classifier
    intent_model = fit_softmax(X[tr], phase_id[tr], n_classes, lr=0.1, steps=4000, l2=1e-3, seed=0, class_weight='balanced')
    intent_model_noprev = fit_softmax(Xn[tr], phase_id[tr], n_classes, lr=0.1, steps=4000, l2=1e-3, seed=0, class_weight='balanced')

    P_tr, _ = predict_softmax(intent_model, X[tr])
    P_va, _ = predict_softmax(intent_model, X[va])
    pred_tr = P_tr.argmax(1)
    pred_va = P_va.argmax(1)

    # Intent-conditioned heads. Use true intent one-hot for training, predicted for eval.
    oh_tr = _onehot(phase_id[tr], n_classes)
    oh_va_pred = _onehot(pred_va, n_classes)
    oh_va_true = _onehot(phase_id[va], n_classes)
    P_va_feat = P_va.astype(np.float32)

    def Xh(Xpart, oh, prob=None):
        parts = [Xpart, oh]
        if prob is not None:
            parts.append(prob)
        return np.concatenate(parts, axis=1).astype(np.float32)

    P_tr_feat = P_tr.astype(np.float32)
    Xh_tr = Xh(X[tr], oh_tr, P_tr_feat)
    Xh_va_pred = Xh(X[va], oh_va_pred, P_va_feat)
    Xh_va_true = Xh(X[va], oh_va_true, P_va_feat)

    # Waypoint/orientation/gripper heads
    wp_model = fit_ridge_local(Xh_tr, target_xyz[tr], alpha=1.0)
    rpy_model = fit_ridge_local(Xh_tr, target_rpy[tr], alpha=1.0)
    # gripper: logistic on conditioned features
    grip_model = _fit_grip_logistic(Xh_tr, grip[tr], threshold=0.02)
    # termination: from intent DONE prob, but also train a small logistic
    term_model = _fit_grip_logistic(Xh_tr, done[tr], threshold=0.5, y_is_binary=True)

    # Eval
    wp_va = predict_ridge_local(wp_model, Xh_va_pred)
    rpy_va = predict_ridge_local(rpy_model, Xh_va_pred)
    grip_logit_va = _predict_grip_logistic(grip_model, Xh_va_pred)
    term_logit_va = _predict_grip_logistic(term_model, Xh_va_pred)

    intent_acc = float(np.mean(pred_va == phase_id[va]))
    # confusion
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t,p in zip(phase_id[va], pred_va): conf[t,p]+=1
    confusion = {PHASES[i]: {PHASES[j]: int(conf[i,j]) for j in range(n_classes) if conf[i,j]>0} for i in range(n_classes)}

    # per-intent waypoint/orientation
    per_intent = {}
    for pid, pname in ID_TO_PHASE.items():
        mask = phase_id[va] == pid
        if not mask.any(): continue
        per_intent[pname] = {
            'n': int(mask.sum()),
            'intent_recall': float(np.mean(pred_va[mask]==pid)),
            'waypoint': mean_abs_error(wp_va[mask], target_xyz[va][mask]),
            'orientation_mae_rad': {k: float(v) for k,v in zip(['roll','pitch','yaw'], np.abs(rpy_va[mask]-target_rpy[va][mask]).mean(0).round(6))},
        }

    grip_true_open = (grip[va] > 0.02).astype(int)
    grip_pred_open = (_sigmoid_np(grip_logit_va) > 0.5).astype(int)
    term_true = (done[va] > 0.5).astype(int)
    term_pred = (_sigmoid_np(term_logit_va) > 0.5).astype(int)

    # transition accuracy
    trans = _transition_accuracy(samples, va, pred_va, phase_id)

    # prev-action dominance
    P_va_noprev, _ = predict_softmax(intent_model_noprev, Xn[va])
    pred_va_noprev = P_va_noprev.argmax(1)
    prev_dom = {
        'intent_acc_with_prev': intent_acc,
        'intent_acc_without_prev': float(np.mean(pred_va_noprev == phase_id[va])),
        'delta': float(intent_acc - np.mean(pred_va_noprev == phase_id[va])),
    }

    # also eval with true intent (oracle intent upper bound for heads, diagnostic only)
    wp_va_true_intent = predict_ridge_local(wp_model, Xh_va_true)
    oracle_intent_wp = mean_abs_error(wp_va_true_intent, target_xyz[va])

    # localization OOF comparison on val
    y_can = ds['y_can']
    can_valid = ds['can_valid']
    loc_mask = va & can_valid
    def loc_metrics(pred, gt, mask, samples_list):
        err = pred[mask] - gt[mask]
        ae = np.abs(err)
        eu = np.linalg.norm(err, axis=1)
        out = {
            'n': int(mask.sum()),
            'mae_x_m': float(ae[:,0].mean()), 'mae_y_m': float(ae[:,1].mean()), 'mae_z_m': float(ae[:,2].mean()),
            'mae_euclidean_m': float(eu.mean()), 'median_euclidean_m': float(np.median(eu)),
            'p90_euclidean_m': float(np.percentile(eu,90)),
            'bias_x_m': float(err[:,0].mean()), 'bias_y_m': float(err[:,1].mean()), 'bias_z_m': float(err[:,2].mean()),
        }
        by_phase = {}
        for ph in ['PREGRASP','DESCEND','LIFT','PLACE_DESCEND']:
            pm = mask & np.array([s.phase==ph for s in samples_list])
            if not pm.any():
                continue
            e = pred[pm]-gt[pm]; ae2=np.abs(e); eu2=np.linalg.norm(e,axis=1)
            by_phase[ph] = {
                'n': int(pm.sum()),
                'mae_x_m': float(ae2[:,0].mean()), 'mae_y_m': float(ae2[:,1].mean()), 'mae_z_m': float(ae2[:,2].mean()),
                'mae_euclidean_m': float(eu2.mean()), 'median_euclidean_m': float(np.median(eu2)),
                'p90_euclidean_m': float(np.percentile(eu2,90)),
            }
        out['by_phase'] = by_phase
        return out
    localization_compare = {
        'vis': loc_metrics(ds['can_oof_vis'], y_can, loc_mask, samples),
        'state': loc_metrics(ds['can_oof_state'], y_can, loc_mask, samples),
        'fused': loc_metrics(ds['can_oof_fused'], y_can, loc_mask, samples),
        'selected_for_training_runtime': 'vis',
    }
    loc_mae = localization_compare['vis']['mae_euclidean_m'] if loc_mask.any() else None

    result = {
        'status': 'measured-offline-hierarchical-v3',
        'model': 'hierarchical_intent_conditioned_waypoint_v1',
        'dataset_root': str(dataset_root),
        'cache_path': str(cache_path),
        'val_bases': list(ds['val_bases']),
        'n_train': int(tr.sum()), 'n_val': int(va.sum()),
        'phases': PHASES,
        'intent_accuracy': intent_acc,
        'intent_confusion': confusion,
        'per_intent': per_intent,
        'waypoint_overall_pred_intent': mean_abs_error(wp_va, target_xyz[va]),
        'waypoint_overall_oracle_intent_diagnostic': oracle_intent_wp,
        'orientation_overall_mae_rad': {k: float(v) for k,v in zip(['roll','pitch','yaw'], np.abs(rpy_va-target_rpy[va]).mean(0).round(6))},
        'gripper': _grip_metrics(grip_pred_open, grip_true_open),
        'termination': _grip_metrics(term_pred, term_true),
        'transition_accuracy': trans,
        'prev_action_dominance': prev_dom,
        'localization_oof_val_mae_m': loc_mae,
        'localization_oof_compare': localization_compare,
        'notes': [
            'Expert phase annotations used only as offline supervision labels.',
            'Runtime must predict intent; no expert phase labels/GT/hardcoded transitions/scripted fallback.',
            'CAN localization inputs are OOF predicted, not ground truth.',
            'Oracle-intent waypoint is diagnostic only.',
            'Visual-only CAN localization selected for deployment robustness after live OOD drift audit.',
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    # save models
    np.savez_compressed(out_dir/'hierarchical_models.npz',
        intent_W=intent_model['W'], intent_b=intent_model['b'], intent_mu=intent_model['mu'], intent_sd=intent_model['sd'],
        intent_np_W=intent_model_noprev['W'], intent_np_b=intent_model_noprev['b'], intent_np_mu=intent_model_noprev['mu'], intent_np_sd=intent_model_noprev['sd'],
        wp_W=wp_model['W'], wp_mu=wp_model['mu'], wp_sd=wp_model['sd'],
        rpy_W=rpy_model['W'], rpy_mu=rpy_model['mu'], rpy_sd=rpy_model['sd'],
        grip_W=grip_model['W'], grip_b=np.array([grip_model['b']]), grip_mu=grip_model['mu'], grip_sd=grip_model['sd'],
        term_W=term_model['W'], term_b=np.array([term_model['b']]), term_mu=term_model['mu'], term_sd=term_model['sd'],
        phases=np.array(PHASES),
    )
    (out_dir/'eval.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # save dataset metadata for runtime
    meta = {
        'val_bases': list(ds['val_bases']),
        'feature_dim': int(X.shape[1]),
        'head_feature_dim': int(Xh_tr.shape[1]),
        'phases': PHASES,
    }
    (out_dir/'meta.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    return result


def _sigmoid_np(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))

def _fit_grip_logistic(X, y, threshold=0.02, y_is_binary=False, lr=0.1, steps=3000):
    mu, sd = X.mean(0), X.std(0); sd[sd<1e-6]=1.0
    Xn=(X-mu)/sd
    yy = y.astype(np.float32) if y_is_binary else (y>threshold).astype(np.float32)
    pos=yy.sum(); neg=len(yy)-pos
    w_pos=len(yy)/(2*max(pos,1)); w_neg=len(yy)/(2*max(neg,1))
    sw=np.where(yy>0.5,w_pos,w_neg).astype(np.float32)
    rng=np.random.default_rng(0)
    W=rng.normal(0,0.01,size=Xn.shape[1]).astype(np.float64)
    b=0.0
    for _ in range(steps):
        z=Xn@W+b
        p=_sigmoid_np(z)
        err=(p-yy)*sw
        W -= lr*(Xn.T@err)/len(yy)
        b -= lr*float(err.mean())
    return {'W':W,'b':b,'mu':mu,'sd':sd}

def _predict_grip_logistic(model, X):
    Xn=(X-model['mu'])/model['sd']
    return Xn@model['W']+model['b']


def _transition_accuracy(samples, va_mask, pred_va, phase_id):
    # evaluate transition prediction: when true phase changes between consecutive samples in episode, did predicted phase change correctly?
    va_idx = np.where(va_mask)[0]
    by_ep={}
    for pos, idx in enumerate(va_idx.tolist()):
        by_ep.setdefault(samples[idx].episode_index,[]).append((pos,idx))
    trans_total=0; trans_correct=0; trans_pred_any=0
    for items in by_ep.values():
        items=sorted(items, key=lambda x: samples[x[1]].t)
        for k in range(1,len(items)):
            ppos,pi=items[k-1]; cpos,ci=items[k]
            true_t = phase_id[ci]!=phase_id[pi]
            pred_t = pred_va[cpos]!=pred_va[ppos]
            if true_t:
                trans_total+=1
                if pred_t and pred_va[cpos]==phase_id[ci]: trans_correct+=1
            if pred_t: trans_pred_any+=1
    return {
        'true_transitions': trans_total,
        'predicted_transitions': trans_pred_any,
        'transition_correct': trans_correct,
        'transition_accuracy': float(trans_correct/trans_total) if trans_total else None,
    }


def load_hierarchical_models(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as d:
        phases = [str(x) for x in d['phases'].tolist()]
        def m(prefix, has_b=True):
            out = {'W': d[f'{prefix}_W'], 'mu': d[f'{prefix}_mu'], 'sd': d[f'{prefix}_sd']}
            if has_b: out['b'] = d[f'{prefix}_b']
            return out
        intent = m('intent', True)
        intent['b'] = intent['b'].reshape(-1)
        grip = m('grip', True); grip['b'] = float(np.array(grip['b']).reshape(-1)[0])
        term = m('term', True); term['b'] = float(np.array(term['b']).reshape(-1)[0])
        return {
            'intent': intent,
            'wp': m('wp', False),
            'rpy': m('rpy', False),
            'grip': grip,
            'term': term,
            'phases': phases,
        }
