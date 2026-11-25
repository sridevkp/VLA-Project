"""VLA-only lockstep control; ground truth is confined to the evaluation scorer."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from vlarobot.core.types import ApplyResult, GripperCmd, JointTarget, Observation, ScenarioSpec
from vlarobot.vla.action_adapter import ActionAdapter


class ChunkSource(Protocol):
    def reset(self, seed: int) -> None: ...
    def predict(
        self, observation: Observation, instruction: str
    ) -> tuple[np.ndarray, dict[str, float]]: ...


class _CountBackendVetoes:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.rejections = 0

    def apply(self, action: JointTarget | GripperCmd) -> ApplyResult:
        result = self.inner.apply(action)
        self.rejections += int(not result.accepted)
        return result

    def step(self, n_steps: int = 1) -> Observation:
        return self.inner.step(n_steps)


def project_recovery_chunk(actions: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Map bounded numerical extrapolation from the historical checkpoint into motor units."""
    values = np.asarray(actions)
    if (
        values.ndim != 2
        or values.shape[1] != 7
        or not len(values)
        or values.dtype.kind not in "fiu"
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("invalid model action chunk")
    result = values.astype(np.float64)
    opening = result[:, 6]
    lower, upper = limits
    # Historical valid labels are the endpoints. TRAIN observed up to 0.793 mm
    # of extrapolation. Values outside this distinct 1 mm proposal envelope fail closed.
    if np.any((opening < lower - 0.001) | (opening > upper + 0.001)):
        raise ValueError("learned gripper target outside V2 projection envelope")
    result[:, 6] = np.clip(opening, lower, upper)
    return result


class VLAEpisodePolicy:
    """Learned actions only. The independent scorer may confirm terminal success in V2."""

    def __init__(
        self,
        scene: ScenarioSpec,
        model: ChunkSource,
        adapter: ActionAdapter,
        *,
        instruction: str,
        max_steps: int,
        max_chunk_actions: int,
        home_tolerance_rad: float,
        arm_names: tuple[str, ...],
        trace: Path,
        score: Callable[[Any, Observation, bool, str | None, str | None], dict[str, Any]],
        begin_metrics: Callable[[Any], None] | None = None,
        recovery_v2: bool = False,
    ) -> None:
        if max_steps <= 0 or max_chunk_actions <= 0 or home_tolerance_rad <= 0:
            raise ValueError("VLA execution requires positive fixed bounds")
        self.scene, self.model, self.adapter = scene, model, adapter
        self.instruction, self.max_steps = instruction, max_steps
        self.max_chunk_actions = max_chunk_actions
        self.home_tolerance_rad, self.arm_names = home_tolerance_rad, arm_names
        self.trace, self.score = trace, score
        self.begin_metrics = begin_metrics
        self.recovery_v2 = recovery_v2

    def execute(self, backend: Any, seed: int) -> dict[str, Any]:
        import json

        fallback_at_start = getattr(self.adapter.solver, "fallback_solutions", 0)
        observation = backend.reset(self.scene, seed)
        if self.begin_metrics is not None:
            self.begin_metrics(backend)
        home = np.array([observation.robot.joint_pos[n] for n in self.arm_names])
        self.model.reset(seed)
        actuator = _CountBackendVetoes(backend)
        departed = home_reached = False
        verified = None
        home_candidates = 0
        failure_state = failure_reason = None
        executed = chunks = 0
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        inputs = self.trace.with_suffix(".inputs")
        inputs.mkdir(exist_ok=False)
        with self.trace.open("x", encoding="utf-8", buffering=1) as stream:
            while executed < self.max_steps:
                try:
                    actions, timing = self.model.predict(observation, self.instruction)
                    if (
                        actions.ndim != 2
                        or actions.shape[1] != 7
                        or not 0 < len(actions) <= self.max_chunk_actions
                        or actions.dtype.kind not in "fiu"
                        or not np.all(np.isfinite(actions))
                    ):
                        raise ValueError("invalid model action chunk")
                except (ValueError, RuntimeError, TimeoutError) as error:
                    failure_state, failure_reason = "MODEL", str(error)
                    break
                chunks += 1
                if not all(math.isfinite(v) and v >= 0 for v in timing.values()):
                    failure_state, failure_reason = "MODEL", "invalid inference timing telemetry"
                    break
                raw_actions = actions
                if self.recovery_v2:
                    try:
                        actions = project_recovery_chunk(
                            raw_actions, self.adapter.safety_monitor.limits.gripper_opening
                        )
                    except ValueError as error:
                        stream.write(
                            json.dumps(
                                {
                                    "chunk": chunks,
                                    "t": observation.t,
                                    "projection_error": str(error),
                                    "actions": raw_actions.tolist(),
                                }
                            )
                            + "\n"
                        )
                        failure_state, failure_reason = "MODEL", str(error)
                        break
                image_path = inputs / f"{chunks:06d}.png"
                Image.fromarray(observation.rgb).save(image_path)
                stream.write(
                    json.dumps(
                        {
                            "chunk": chunks,
                            "t": observation.t,
                            **timing,
                            "input_image": str(image_path),
                            "joint_pos": observation.robot.joint_pos,
                            "gripper_mean_m": observation.robot.gripper_opening,
                            "rgb_hwc_sha256": hashlib.sha256(observation.rgb.tobytes()).hexdigest(),
                            "actions": raw_actions.tolist(),
                            **(
                                {
                                    "projected_gripper_m": actions[:, 6].tolist(),
                                    "projection_delta_m": (
                                        actions[:, 6] - raw_actions[:, 6]
                                    ).tolist(),
                                    "projected_count": int(
                                        np.count_nonzero(actions[:, 6] != raw_actions[:, 6])
                                    ),
                                }
                                if self.recovery_v2
                                else {}
                            ),
                            "instruction": self.instruction,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                for values in actions[: self.max_steps - executed]:
                    result = self.adapter.apply(actuator, values, observation.robot)
                    if not result.accepted:
                        failure_state, failure_reason = "SAFETY", result.reason
                        break
                    observation = backend.step(1)
                    executed += 1
                    q = np.array([observation.robot.joint_pos[n] for n in self.arm_names])
                    error = float(np.max(np.abs(q - home)))
                    departed = departed or error > self.home_tolerance_rad
                    if departed and error <= self.home_tolerance_rad:
                        home_candidates += 1
                        if self.recovery_v2:
                            # Scorer output only gates episode termination. The model input
                            # and the next learned action do not use oracle observations.
                            candidate = self.score(backend, observation, True, None, None)
                            if candidate.get("success") is not True:
                                continue
                            verified = candidate
                        home_reached = True
                        break
                if failure_state or home_reached:
                    break
        if not home_reached and failure_state is None:
            failure_state, failure_reason = (
                "BUDGET",
                "fixed longest-demonstration horizon exhausted",
            )
        outcome = verified or self.score(
            backend, observation, home_reached, failure_state, failure_reason
        )
        return {
            **outcome,
            "model_chunks": chunks,
            **(
                {
                    "home_completion_candidates": home_candidates,
                    "ik_fallback_solutions": (
                        getattr(self.adapter.solver, "fallback_solutions", 0) - fallback_at_start
                    ),
                }
                if self.recovery_v2
                else {}
            ),
            "executed_actions": executed,
            "max_episode_steps": self.max_steps,
            "learned_input_state_source": "robot_rgb",
            "scripted_actions_sent": 0,
            "rejections": actuator.rejections,
            "adapter_vetoes": int(failure_state == "SAFETY" and actuator.rejections == 0),
        }
