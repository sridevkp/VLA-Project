"""Minimal V3 closed-loop diagnostic runner (TRAIN only, no holdout)."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from vlarobot.backends.webots.pose_solver import BoundedPoseSolver
from vlarobot.eval.runner import EvalEpisode, EvalRunConfig, build_episode_plan
from vlarobot.eval.settings import load_eval_settings
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import DEFAULT_WORLD, launch_webots_session
from vlarobot.execution.vla_campaign import CanEpisodeScorer, make_vla_adapter
from vlarobot.execution.vla_episode import VLAEpisodePolicy
from vlarobot.vla.settings import load_vla_settings
from vlarobot.v3.offline import load_metadata_rows, load_sources, align_samples, choose_validation_bases, split_by_bases, fit_ridge, predict_ridge
from vlarobot.v3.train import _load_dino_cache, _pca_fit_transform, _fit_gripper_logistic
from vlarobot.v3.vla_source import V3ChunkSource


class _DinoRuntime:
    def __init__(self, model_name='facebook/dinov2-small'):
        from vlarobot.v3.features import _load_dinov2, describe_encoder
        import torch
        self.torch = torch
        self.processor, self.model = _load_dinov2(model_name)[1:]
        self.spec = describe_encoder(model_name)
        self.model.eval()

    def __call__(self, jpeg_bytes: bytes):
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(jpeg_bytes)) as im:
            img = im.convert('RGB')
        inputs = self.processor(images=[img], return_tensors='pt')
        with self.torch.inference_mode():
            out = self.model(**inputs)
            tokens = out.last_hidden_state
            cls = tokens[:, 0].cpu().numpy().astype(np.float32)[0]
            reg = self.spec.num_register_tokens
            h, w = self.spec.spatial_hw
            patches = tokens[:, 1 + reg :, :][:, : h * w, :].reshape(1, h, w, -1)
            n, _, _, c = patches.shape
            spatial4 = patches.reshape(n, 4, h // 4, 4, w // 4, c).mean(axis=(2, 4)).reshape(n, -1)
        return cls, spatial4[0].cpu().numpy().astype(np.float32)


def build_v3_source(dataset_root: Path, cache_path: Path) -> V3ChunkSource:
    sources = load_sources(dataset_root)
    rows = load_metadata_rows(dataset_root)
    samples = align_samples(sources, rows)
    pooled, spatial, _ = _load_dino_cache(cache_path, samples)
    val_bases = choose_validation_bases(samples)
    train, _ = split_by_bases(samples, val_bases)
    train_ids = {id(s) for s in train}
    tr = np.array([id(s) in train_ids for s in samples])
    tr_idx = np.where(tr)[0]
    state = np.stack([s.state for s in samples]).astype(np.float32)
    prev = np.stack([s.prev_action for s in samples]).astype(np.float32)
    prev_no_grip = prev.copy(); prev_no_grip[:, 6] = 0.0
    targets = np.stack([s.target_xyz for s in samples]).astype(np.float32)
    yg = np.array([s.gripper_target for s in samples], dtype=np.float32)
    pca64 = _pca_fit_transform(spatial, tr, 64)
    Xb = np.column_stack([state, prev_no_grip])
    ridge_base = fit_ridge(Xb[tr_idx], targets[tr_idx])
    resid = targets[tr_idx] - (np.column_stack([np.ones(len(tr_idx)), (Xb[tr_idx]-ridge_base['mean'])/ridge_base['scale']]) @ ridge_base['coefficients'])
    ridge_resid = fit_ridge(pca64[tr_idx], resid)
    # CAN localization model (offline GT labels only)
    from vlarobot.v3.localization import _can_base_at_frame
    y_can=[]
    for s in samples:
        c=_can_base_at_frame(Path(sources[s.episode_index]['raw_dir']), s.t)
        y_can.append(c if c is not None else np.zeros(3,dtype=np.float32))
    y_can=np.stack(y_can).astype(np.float32)
    X_loc=np.concatenate([state,pca64],axis=1)
    can_model=fit_ridge(X_loc[tr_idx], y_can[tr_idx])
    can_pred=predict_ridge(can_model, X_loc)
    # object-relative waypoint model
    rel_target=targets-can_pred
    X_wp=np.concatenate([state,prev_no_grip,can_pred,pca64],axis=1)
    waypoint_model=fit_ridge(X_wp[tr_idx], rel_target[tr_idx])
    # gripper without prev gripper action (deployment candidate)
    Xg = np.column_stack([state, prev_no_grip, pca64, can_pred])
    grip = _fit_gripper_logistic(Xg[tr_idx], yg[tr_idx], Xg[tr_idx], yg[tr_idx])
    xtr = spatial[tr_idx]
    mu = xtr.mean(axis=0)
    _, _, vt = np.linalg.svd(xtr - mu, full_matrices=False)
    comp = vt[:64]
    dino = _DinoRuntime()
    return V3ChunkSource(
        ridge_base=ridge_base, ridge_resid=ridge_resid,
        grip_w=grip['W'], grip_b=grip['b'], grip_mu=grip['mu'], grip_sd=grip['sd'],
        pca_mu=mu, pca_comp=comp, dino=dino, feature_mode='no_prev_grip',
        can_model=can_model, waypoint_mode='object_relative', waypoint_model=waypoint_model,
    )


def run_train_diagnostic(out: Path, *, webots: str, dataset_root: Path, cache_path: Path, indices=(0,1,2)) -> dict[str, Any]:
    import os
    os.environ.setdefault("VLAROBOT_URDF_REGISTRATION_FILE", "calibration/urdf-base-train-v1.yaml")
    out.mkdir(parents=True, exist_ok=True)
    source = build_v3_source(dataset_root, cache_path)
    eval_settings = load_eval_settings()
    plan = build_episode_plan("train", 0, n=0, settings=eval_settings)
    selected = [e for e in plan if e.index in set(indices)]
    vla = load_vla_settings()
    adapter = make_vla_adapter(recovery_ik=True)
    rows = []
    for entry in selected:
        trace = out / f"episode-{entry.index:02d}.latency.jsonl"
        scorer = CanEpisodeScorer()
        policy = VLAEpisodePolicy(
            entry.scene,
            source,
            adapter,
            instruction="Pick up the can and place it at the target location.",
            max_steps=120,
            max_chunk_actions=vla.max_chunk_actions,
            home_tolerance_rad=0.01,
            arm_names=('arm_1_joint','arm_2_joint','arm_3_joint','arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint'),
            trace=trace,
            score=scorer,
            begin_metrics=scorer.begin,
            recovery_v2=False,
            endpoint_gripper=False,
        )
        with launch_webots_session(webots=webots, world=DEFAULT_WORLD, trace_dir=out, trace_prefix=f"v3-diag-{entry.index:02d}", headless=True) as session:
            started = perf_counter()
            result = policy.execute(session.client, entry.seed)
            result['wall_s'] = perf_counter() - started
            result['scene'] = entry.scene.name
            result['index'] = entry.index
            rows.append(result)
            write_json(out / f"episode-{entry.index:02d}.result.json", result)
    summary = {'rows': rows, 'n': len(rows), 'successes': sum(1 for r in rows if r.get('success') is True)}
    write_json(out / 'diagnostic-summary.json', summary)
    return summary
