"""Closed-loop V3 runtime: residual ridge waypoint + balanced logistic gripper.

No runtime expert phase labels. Uses existing backend IK/safety/interpolation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.execution.scripted_can import DEFAULT_HOME_JOINTS_RAD, GRIPPER_OPEN_TOLERANCE_M


@dataclass
class V3RuntimeConfig:
    max_steps: int = 120
    waypoint_tol_m: float = 0.01
    hold_steps: int = 4
    gripper_threshold_m: float = 0.02


class V3ResidualPolicy:
    """Minimal autonomous policy for 3-scene TRAIN diagnostic.

    This is a bounded prototype. It uses learned waypoint+gripper predictions but
    relies on existing backend safety, IK and interpolation. It does not use runtime
    expert phase labels or oracle grasp decisions.
    """

    def __init__(self, scene, *, ridge_base, ridge_resid, grip_model, pca_mu, pca_comp, dino=None, config: V3RuntimeConfig | None = None, trace_path: Path | None = None):
        self.scene = scene
        self.ridge_base = ridge_base
        self.ridge_resid = ridge_resid
        self.grip_model = grip_model
        self.pca_mu = pca_mu
        self.pca_comp = pca_comp
        self.dino = dino  # callable(image_bytes)->(pooled, spatial4x4_flat) or None
        self.config = config or V3RuntimeConfig()
        self.trace_path = trace_path

    def _features(self, obs, prev_action: np.ndarray) -> np.ndarray:
        # state: 8 dims from observation.robot.joint_pos in arm order + gripper
        arm = ['arm_1_joint','arm_2_joint','arm_3_joint','arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint']
        state = np.array([obs.robot.joint_pos[n] for n in arm] + [float(obs.robot.gripper_opening)], dtype=np.float32)
        if self.dino is not None and obs.camera_rgb is not None:
            pooled, spatial4 = self.dino(obs.camera_rgb)
            pca = ((spatial4 - self.pca_mu) @ self.pca_comp.T).astype(np.float32)
            vis = pca
        else:
            vis = np.zeros(self.pca_comp.shape[0], dtype=np.float32)
        return np.concatenate([state, prev_action.astype(np.float32), vis])

    def execute(self, backend: Any, seed: int) -> dict[str, Any]:
        obs = backend.reset(self.scene, seed)
        prev_action = np.zeros(7, dtype=np.float32)
        trace = []
        success = False
        failure_reason = None
        for step in range(self.config.max_steps):
            x = self._features(obs, prev_action)[None, :]
            base_x = x[:, :15]
            pred_xyz = (predict_ridge(self.ridge_base, base_x) + predict_ridge(self.ridge_resid, x[:, 15:])).reshape(3)
            grip_logit = float(self.grip_model['predict_logit'](x[:, :])[0])
            grip_open = grip_logit > 0.0
            grip_target = 0.045 if grip_open else 0.0
            # simple action: delta toward predicted xyz as absolute ee target via backend IK if available
            action = {'target_ee_base_xyz_m': pred_xyz.tolist(), 'gripper_target_m': grip_target}
            try:
                obs = backend.apply(action)
                for _ in range(self.config.hold_steps - 1):
                    obs = backend.step(1)
            except Exception as e:
                failure_reason = str(e)
                break
            prev_action = np.array([0, 0, 0, 0, 0, 0, grip_target], dtype=np.float32)
            trace.append({'step': step, 't': float(obs.t), 'pred_xyz': pred_xyz.tolist(), 'grip_target': grip_target})
            # termination: independent check if available
            if bool(getattr(obs, 'success', False)):
                success = True
                break
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text(json.dumps(trace, indent=2), encoding='utf-8')
        return {
            'success': bool(success),
            'steps': step + 1 if 'step' in locals() else 0,
            'failure_reason': failure_reason,
            'trace_path': str(self.trace_path) if self.trace_path else None,
        }


def predict_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    normalized = (x - model['mean']) / model['scale']
    design = np.column_stack([np.ones(len(normalized)), normalized])
    return design @ model['coefficients']
