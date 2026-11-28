"""V3 ChunkSource: residual ridge waypoint + logistic gripper, no runtime phase labels."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from vlarobot.v3.train import _pca_fit_transform


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


class V3ChunkSource:
    def __init__(self, *, ridge_base, ridge_resid, grip_w, grip_b, grip_mu, grip_sd, pca_mu, pca_comp, dino, feature_mode="no_prev_grip"):
        self.ridge_base = ridge_base
        self.ridge_resid = ridge_resid
        self.grip_w = grip_w
        self.grip_b = grip_b
        self.grip_mu = grip_mu
        self.grip_sd = grip_sd
        self.pca_mu = pca_mu
        self.pca_comp = pca_comp
        self.dino = dino
        self.feature_mode = feature_mode
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.last_timing = {}

    def reset(self, seed: int) -> None:
        self.prev_action = np.zeros(7, dtype=np.float32)

    def _state(self, obs) -> np.ndarray:
        arm = ['arm_1_joint','arm_2_joint','arm_3_joint','arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint']
        return np.array([obs.robot.joint_pos[n] for n in arm] + [float(obs.robot.gripper_opening)], dtype=np.float32)

    def _vis(self, obs) -> np.ndarray:
        t0 = perf_counter()
        rgb = getattr(obs, "rgb", None)
        if self.dino is None or rgb is None:
            return np.zeros(self.pca_comp.shape[0], dtype=np.float32), 0.0
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(np.asarray(rgb)).save(buf, format="JPEG", quality=95)
        pooled, spatial4 = self.dino(buf.getvalue())
        pca = ((spatial4 - self.pca_mu) @ self.pca_comp.T).astype(np.float32)
        return pca.reshape(-1), perf_counter() - t0

    def predict(self, observation, instruction: str) -> tuple[np.ndarray, dict[str, float]]:
        t0 = perf_counter()
        state = self._state(observation)
        vis, dino_s = self._vis(observation)
        prev = self.prev_action.copy()
        if self.feature_mode == "no_prev_grip":
            prev[6] = 0.0
        elif self.feature_mode == "no_prev":
            prev = np.zeros(7, dtype=np.float32)
        x = np.concatenate([state, prev, vis]).reshape(1, -1)
        base_x = x[:, :15]
        pred_xyz = (predict_ridge(self.ridge_base, base_x) + predict_ridge(self.ridge_resid, x[:, 15:])).reshape(3)
        gx = (x - self.grip_mu) / self.grip_sd
        grip_logit = float(gx @ self.grip_w + self.grip_b)
        grip_open = grip_logit > 0.0
        grip_target = 0.045 if grip_open else 0.0
        # delta from current EE
        ee = observation.robot.ee_pose.position.as_array().astype(np.float32)
        delta = (pred_xyz - ee).astype(np.float32)
        # Bounded feasible step: preserve direction, cap norm for IK feasibility.
        norm = float(np.linalg.norm(delta))
        max_step = 0.01
        if norm > max_step:
            delta = delta * (max_step / norm)
        action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip_target], dtype=np.float32)
        self.prev_action = action.copy()  # policy-generated prev action
        timing = {"infer_s": perf_counter() - t0, "dino_s": dino_s}
        self.last_timing = timing
        return action.reshape(1, 7), timing


def predict_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    normalized = (x - model['mean']) / model['scale']
    design = np.column_stack([np.ones(len(normalized)), normalized])
    return design @ model['coefficients']
