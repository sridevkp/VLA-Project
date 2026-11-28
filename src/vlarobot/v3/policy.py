"""Compact V3 waypoint policy prototype (CPU-only, deterministic).

This is a bounded prototype for offline feasibility: it predicts a waypoint
(xyz) and gripper-close logit from causally available inputs. It does not claim
holdout performance and is not a VLA training run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyConfig:
    input_dim: int
    hidden_dim: int = 256
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    seed: int = 0
    gripper_threshold_m: float = 0.02


def _mlp(in_dim: int, hidden: int, out_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class CompactWaypointPolicy:
    def __init__(self, cfg: PolicyConfig):
        import torch

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.net = _mlp(cfg.input_dim, cfg.hidden_dim, 4)  # xyz + gripper_logit
        self.device = torch.device("cpu")
        self.net.to(self.device)

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        self.net.eval()
        with torch.inference_mode():
            t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
            out = self.net(t).cpu().numpy()
        return {"xyz": out[:, :3], "gripper_logit": out[:, 3]}

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "state_dict": self.net.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> CompactWaypointPolicy:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        policy = CompactWaypointPolicy(PolicyConfig(**payload["config"]))
        policy.net.load_state_dict(payload["state_dict"])
        return policy


def train_policy(
    x_train: np.ndarray,
    y_xyz_train: np.ndarray,
    y_grip_train: np.ndarray,
    x_val: np.ndarray,
    y_xyz_val: np.ndarray,
    y_grip_val: np.ndarray,
    cfg: PolicyConfig,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    torch.manual_seed(cfg.seed)
    model = CompactWaypointPolicy(cfg)
    opt = torch.optim.AdamW(model.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    def batches(x, yx, yg):
        idx = np.arange(len(x))
        for i in range(0, len(idx), cfg.batch_size):
            j = idx[i : i + cfg.batch_size]
            yield (
                torch.from_numpy(x[j].astype(np.float32)),
                torch.from_numpy(yx[j].astype(np.float32)),
                torch.from_numpy(yg[j].astype(np.float32)),
            )

    history = []
    for epoch in range(cfg.epochs):
        model.net.train()
        losses = []
        for xb, yxb, ygb in batches(x_train, y_xyz_train, y_grip_train):
            opt.zero_grad(set_to_none=True)
            out = model.net(xb)
            loss = mse(out[:, :3], yxb) + bce(out[:, 3], ygb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_policy(model, x_val, y_xyz_val, y_grip_val, cfg.gripper_threshold_m)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val})
    return {"model": model, "history": history, "final": history[-1] if history else {}}


def evaluate_policy(
    model: CompactWaypointPolicy,
    x: np.ndarray,
    y_xyz: np.ndarray,
    y_grip: np.ndarray,
    threshold_m: float = 0.02,
) -> dict[str, Any]:
    pred = model.predict(x)
    xyz_err = np.abs(pred["xyz"] - y_xyz)
    per_axis = xyz_err.mean(axis=0)
    grip_true = (y_grip > threshold_m).astype(int)
    grip_pred = (1 / (1 + np.exp(-pred["gripper_logit"])) > 0.5).astype(int)
    tp = int(np.sum((grip_pred == 1) & (grip_true == 1)))
    fp = int(np.sum((grip_pred == 1) & (grip_true == 0)))
    fn = int(np.sum((grip_pred == 0) & (grip_true == 1)))
    tn = int(np.sum((grip_pred == 0) & (grip_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "xyz_mae_m": round(float(per_axis.mean()), 6),
        "xyz_mae_x_m": round(float(per_axis[0]), 6),
        "xyz_mae_y_m": round(float(per_axis[1]), 6),
        "xyz_mae_z_m": round(float(per_axis[2]), 6),
        "gripper_accuracy": round(float(np.mean(grip_pred == grip_true)), 6),
        "gripper_precision": round(float(precision), 6),
        "gripper_recall": round(float(recall), 6),
        "gripper_tp": tp,
        "gripper_fp": fp,
        "gripper_fn": fn,
        "gripper_tn": tn,
    }
