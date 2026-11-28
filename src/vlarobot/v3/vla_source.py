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
    def __init__(
        self,
        *,
        ridge_base,
        ridge_resid,
        grip_w,
        grip_b,
        grip_mu,
        grip_sd,
        pca_mu,
        pca_comp,
        dino,
        feature_mode="no_prev_grip",
        can_model=None,
        waypoint_mode="absolute",  # absolute | object_relative
        waypoint_model=None,
    ):
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
        self.can_model = can_model
        self.waypoint_mode = waypoint_mode
        self.waypoint_model = waypoint_model

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
        can_pred = None
        if self.can_model is not None:
            X_loc = np.concatenate([state, vis]).reshape(1, -1)
            can_pred = predict_ridge(self.can_model, X_loc).reshape(3)
        if self.waypoint_mode == "object_relative" and self.waypoint_model is not None and can_pred is not None:
            X_wp = np.concatenate([state, prev, can_pred, vis]).reshape(1, -1)
            rel = predict_ridge(self.waypoint_model, X_wp).reshape(3)
            pred_xyz = (can_pred + rel).reshape(3)
        else:
            base_x = x[:, :15]
            pred_xyz = (
                predict_ridge(self.ridge_base, base_x) + predict_ridge(self.ridge_resid, x[:, 15:])
            ).reshape(3)
        gx_in = np.concatenate([state, prev, vis, can_pred]).reshape(1, -1) if can_pred is not None else x
        if gx_in.shape[1] == self.grip_mu.shape[0]:
            gx = (gx_in - self.grip_mu) / self.grip_sd
            grip_logit = float(gx @ self.grip_w + self.grip_b)
        else:
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


class HierarchicalChunkSource:
    """Runtime hierarchical policy: learned intent -> intent-conditioned waypoint/orientation/gripper/termination.

    Decision boundary: retain a learned waypoint across bounded feasible incremental
    movements; refresh at completion/interruption/replan boundaries. No expert phase
    labels, no privileged GT, no hardcoded task transitions, no scripted fallback.
    """

    def __init__(self, *, hier_models: dict, can_model, pca_mu, pca_comp, dino,
                 feature_mode="no_prev_grip", can_mode="vis", waypoint_hold_s=2.0,
                 waypoint_reach_m=0.015):
        self.m = hier_models
        self.can_model = can_model
        self.pca_mu = pca_mu
        self.pca_comp = pca_comp
        self.dino = dino
        self.feature_mode = feature_mode
        self.can_mode = can_mode  # vis|fused|state
        self.waypoint_hold_s = waypoint_hold_s
        self.waypoint_reach_m = waypoint_reach_m
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.last_timing = {}
        self.last_intent = None
        self.last_intent_probs = None
        self._active_waypoint = None
        self._active_rpy = None
        self._active_grip = 0.0
        self._active_intent = None
        self._active_since_t = None
        self._active_since_step = 0
        self._step = 0

    def reset(self, seed: int) -> None:
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.last_intent = None
        self.last_intent_probs = None
        self._active_waypoint = None
        self._active_rpy = None
        self._active_grip = 0.0
        self._active_intent = None
        self._active_since_t = None
        self._active_since_step = 0
        self._step = 0

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

    @staticmethod
    def _softmax1(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def _should_refresh(self, ee, intent, term_prob, now_t):
        if self._active_waypoint is None:
            return True, 'init'
        dist = float(np.linalg.norm(self._active_waypoint - ee))
        if dist <= self.waypoint_reach_m:
            return True, 'reached'
        if term_prob > 0.8 and intent == 'DONE':
            return True, 'term'
        if intent != self._active_intent:
            # refresh on learned intent transition
            return True, 'intent_transition'
        if self._active_since_t is not None and (now_t - self._active_since_t) > self.waypoint_hold_s:
            return True, 'hold_timeout'
        if dist > 0.25:
            return True, 'deviation'
        return False, 'hold'

    def predict(self, observation, instruction: str) -> tuple[np.ndarray, dict[str, float]]:
        t0 = perf_counter()
        state = self._state(observation)
        vis, dino_s = self._vis(observation)
        prev = self.prev_action.copy()
        if self.feature_mode == "no_prev_grip":
            prev[6] = 0.0
        elif self.feature_mode == "no_prev":
            prev = np.zeros(7, dtype=np.float32)

        # predicted CAN localization (learned, no GT). Default vis-only for robustness.
        if self.can_mode == 'state':
            X_loc = state.reshape(1, -1)
        elif self.can_mode == 'fused':
            X_loc = np.concatenate([state, vis]).reshape(1, -1)
        else:
            X_loc = vis.reshape(1, -1)
        can_pred = predict_ridge(self.can_model, X_loc).reshape(3)

        X_base = np.concatenate([state, prev, can_pred, vis]).reshape(1, -1).astype(np.float32)
        im = self.m['intent']
        Xn = (X_base - im['mu']) / im['sd']
        logits = (Xn @ im['W'] + im['b']).reshape(-1)
        probs = self._softmax1(logits)
        intent_id = int(np.argmax(probs))
        phases = self.m['phases']
        intent = phases[intent_id]

        n_classes = len(phases)
        oh = np.zeros((1, n_classes), dtype=np.float32)
        oh[0, intent_id] = 1.0
        Xh = np.concatenate([X_base, oh, probs.reshape(1, -1).astype(np.float32)], axis=1)

        def rr(key, x):
            mm = self.m[key]
            xn = (x - mm['mu']) / mm['sd']
            A = np.column_stack([np.ones(len(xn)), xn])
            return (A @ mm['W']).reshape(-1)

        pred_xyz = rr('wp', Xh).astype(np.float32)
        pred_rpy = rr('rpy', Xh).astype(np.float32)
        gm = self.m['grip']
        gxn = (Xh - gm['mu']) / gm['sd']
        grip_logit = float((gxn @ gm['W'] + gm['b']).reshape(-1)[0])
        grip_target = 0.045 if grip_logit > 0.0 else 0.0
        tm = self.m['term']
        txn = (Xh - tm['mu']) / tm['sd']
        term_logit = float((txn @ tm['W'] + tm['b']).reshape(-1)[0])
        term_prob = float(_sigmoid(term_logit))

        ee = observation.robot.ee_pose.position.as_array().astype(np.float32)
        now_t = float(getattr(observation, 't', self._step))
        refresh, refresh_reason = self._should_refresh(ee, intent, term_prob, now_t)
        if refresh:
            self._active_waypoint = pred_xyz.copy()
            self._active_rpy = pred_rpy.copy()
            self._active_grip = grip_target
            self._active_intent = intent
            self._active_since_t = now_t
            self._active_since_step = self._step
        target = self._active_waypoint.copy()
        active_rpy = self._active_rpy.copy()
        active_grip = float(self._active_grip)
        # gripper should follow learned gripper decision immediately (safety/task-critical)
        active_grip = grip_target

        delta = (target - ee).astype(np.float32)
        norm = float(np.linalg.norm(delta))
        max_step = 0.01
        if norm > max_step:
            delta = delta * (max_step / norm)
        rpy = np.clip(active_rpy, -0.05, 0.05).astype(np.float32)
        action = np.array([delta[0], delta[1], delta[2], rpy[0], rpy[1], rpy[2], active_grip], dtype=np.float32)
        self.prev_action = action.copy()
        timing = {"infer_s": perf_counter() - t0, "dino_s": dino_s}
        self.last_timing = timing
        self.last_intent = intent
        self.last_intent_probs = {phases[i]: float(probs[i]) for i in range(n_classes)}
        self.last_pred = {
            'intent': intent, 'intent_id': intent_id, 'intent_probs': self.last_intent_probs,
            'can_pred': can_pred.tolist(), 'pred_xyz': pred_xyz.tolist(), 'pred_rpy': pred_rpy.tolist(),
            'active_waypoint': target.tolist(), 'refresh': refresh, 'refresh_reason': refresh_reason,
            'grip_logit': grip_logit, 'grip_target': grip_target, 'term_prob': term_prob,
            'can_mode': self.can_mode,
        }
        self._step += 1
        return action.reshape(1, 7), timing
