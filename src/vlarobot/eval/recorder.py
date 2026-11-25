"""Episode recording for eval and demo collection (ADR-0029 week 1, ADR-0031).

One episode becomes:

- ``episode-XX.steps.npz`` — numeric streams: time, joint positions/velocities, gripper
  opening, EE position, base pose, and ground-truth object poses per DEF (position
  ``gt_xyz`` **and orientation ``gt_quat``** as (x, y, z, w); NaN rows where a step did
  not report one).
- ``episode-XX.actions.jsonl`` — every applied action with its safety verdict, plus
  outcome/predicted records kept separate (``episode-XX.meta.json``,
  ``episode-XX.predicted.jsonl``).
- ``episode-XX.contacts.jsonl`` — **actual-collision channel**: one record per step whose
  diagnostics reported non-robot contact nodes (empty file = no contact detected). Kept
  separate from the displacement/tilt collision *proxy* (ADR-0029), which is derived from
  ``gt_xyz``/``gt_quat`` by ``eval/metrics.py``.
- ``episode-XX.meta.json`` — scenario, split/family provenance, pipeline, state-source
  provenance flag (report 4.3), frame/step/contact counts, and the recorded outcome.
- ``frames/episode-XX/NNNNNN.jpg`` — RGB frames at ``frame_hz`` (optional; needs Pillow).

The layout is LeRobot-convertible (image directory + tabular arrays + per-episode
metadata); the conversion ships with ``collect-demos``. World-model predictions are
recorded through :meth:`EpisodeRecorder.record_predicted` and stay empty until M5.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vlarobot.core.types import Action, GroundTruth, Observation, ScenarioSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vlarobot.core.types import ApplyResult, RobotState

_JOINT_KEY_ERROR = "joint key set changed mid-episode"


def _json_default(value: Any) -> Any:
    """JSON encoder for dataclasses, NumPy scalars/arrays, and Path."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True)
class EpisodePaths:
    """Files belonging to one recorded episode."""

    meta: Path
    steps: Path
    actions: Path
    predicted: Path
    contacts: Path
    frames_dir: Path


class EpisodeRecorder:
    """Append-only recorder for one episode at a time (fail-loud when misused)."""

    def __init__(
        self,
        output_dir: Path,
        *,
        frame_hz: float = 1.0,
        record_frames: bool = True,
        jpeg_quality: int = 90,
    ) -> None:
        if frame_hz < 0.0:
            raise ValueError("frame_hz must be >= 0 (0 disables frame capture).")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100].")
        self.output_dir = Path(output_dir)
        self.frame_hz = float(frame_hz)
        self.record_frames = bool(record_frames) and frame_hz > 0.0
        self.jpeg_quality = int(jpeg_quality)
        self._episode: dict[str, Any] | None = None
        self._paths: EpisodePaths | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._episode is not None

    @property
    def paths(self) -> EpisodePaths | None:
        return self._paths

    def open_episode(self, index: int, seed: int, pipeline: str, provenance: str) -> EpisodePaths:
        """Start recording episode ``index`` (raises if one is already open)."""
        if self._episode is not None:
            raise RuntimeError("an episode is already open; close it first.")
        frames_dir = self.output_dir / "frames" / f"episode-{index:02d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.record_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
        self._paths = EpisodePaths(
            meta=self.output_dir / f"episode-{index:02d}.meta.json",
            steps=self.output_dir / f"episode-{index:02d}.steps.npz",
            actions=self.output_dir / f"episode-{index:02d}.actions.jsonl",
            predicted=self.output_dir / f"episode-{index:02d}.predicted.jsonl",
            contacts=self.output_dir / f"episode-{index:02d}.contacts.jsonl",
            frames_dir=frames_dir,
        )
        self._episode = {
            "index": int(index),
            "seed": int(seed),
            "pipeline": str(pipeline),
            "state_source": str(provenance),
            "scenario": None,
            "reset_seed": None,
            "joint_names": None,
            "times": [],
            "joint_pos": [],
            "joint_vel": [],
            "gripper": [],
            "ee_xyz": [],
            "ee_quat": [],
            "base_pose": [],
            "gt_names": set(),
            "gt_rows": [],
            "gt_quat_rows": [],
            "actions": self._paths.actions.open("w", encoding="utf-8"),
            "predicted": self._paths.predicted.open("w", encoding="utf-8"),
            "contacts_file": self._paths.contacts.open("w", encoding="utf-8"),
            "contact_steps": 0,
            "outcome": None,
            "frame_count": 0,
            "next_frame_t": 0.0,
        }
        return self._paths

    def on_reset(self, scenario: ScenarioSpec, seed: int) -> None:
        """Record the scenario actually passed to the backend at reset."""
        episode = self._require_open()
        if episode["scenario"] is not None:
            raise RuntimeError("reset seen twice within one recorded episode.")
        episode["scenario"] = scenario
        episode["reset_seed"] = int(seed)

    # -- data ---------------------------------------------------------------

    def record_step(self, observation: Observation, ground_truth: GroundTruth | None) -> None:
        """Append one post-step observation (and optional GT) to the episode streams."""
        episode = self._require_open()
        state: RobotState = observation.robot
        names = list(state.joint_pos)
        if episode["joint_names"] is None:
            episode["joint_names"] = names
        elif names != episode["joint_names"]:
            raise ValueError(_JOINT_KEY_ERROR)

        episode["times"].append(float(observation.t))
        episode["joint_pos"].append([float(state.joint_pos[name]) for name in names])
        episode["joint_vel"].append([float(state.joint_vel.get(name, 0.0)) for name in names])
        episode["gripper"].append(float(state.gripper_opening))
        episode["ee_xyz"].append(
            [
                float(state.ee_pose.position.x),
                float(state.ee_pose.position.y),
                float(state.ee_pose.position.z),
            ]
        )
        episode["ee_quat"].append([float(value) for value in state.ee_pose.orientation])
        episode["base_pose"].append(
            [float(state.base_pose.x), float(state.base_pose.y), float(state.base_pose.yaw)]
        )

        poses = ground_truth.object_poses if ground_truth is not None else {}
        episode["gt_names"].update(poses)
        episode["gt_rows"].append({name: pose.position for name, pose in poses.items()})
        episode["gt_quat_rows"].append({name: pose.orientation for name, pose in poses.items()})

        # Actual-collision channel (kept separate from the displacement/tilt proxy).
        diagnostics = state.diagnostics
        contact_nodes = diagnostics.get("contact_nodes") if isinstance(diagnostics, dict) else None
        if contact_nodes:
            labels = [
                item.get("node_name", "") if isinstance(item, dict) else str(item)
                for item in cast(Iterable[object], contact_nodes)
            ]
            episode["contacts_file"].write(
                json.dumps({"t": float(observation.t), "nodes": labels}) + "\n"
            )
            episode["contact_steps"] = int(episode["contact_steps"]) + 1

        if self.record_frames:
            t = float(observation.t)
            if t + 1e-9 >= float(episode["next_frame_t"]):
                self._write_frame(observation.rgb, int(episode["frame_count"]))
                episode["frame_count"] = int(episode["frame_count"]) + 1
                episode["next_frame_t"] = t + 1.0 / self.frame_hz

    def record_action(self, t: float, action: Action, result: ApplyResult | None) -> None:
        """Append one applied action with its safety verdict."""
        episode = self._require_open()
        record: dict[str, Any] = {"t": float(t), "action": action}
        if result is not None:
            record["accepted"] = bool(result.accepted)
            record["reason"] = result.reason
        episode["actions"].write(json.dumps(record, default=_json_default) + "\n")

    def record_predicted(self, t: float, payload: Mapping[str, Any]) -> None:
        """Append a world-model prediction (empty until M5)."""
        episode = self._require_open()
        record = {"t": float(t), "predicted": dict(payload)}
        episode["predicted"].write(json.dumps(record, default=_json_default) + "\n")

    def record_outcome(self, outcome: Mapping[str, Any]) -> None:
        """Attach the episode outcome (stored in the meta file at close)."""
        episode = self._require_open()
        episode["outcome"] = dict(outcome)

    def close_episode(self) -> EpisodePaths:
        """Flush arrays and metadata; returns the episode's file paths."""
        episode = self._require_open()
        paths = self._require_paths()
        assert episode["actions"] is not None and episode["predicted"] is not None
        episode["actions"].close()
        episode["predicted"].close()
        episode["contacts_file"].close()

        joint_names = episode["joint_names"] or []
        gt_names = sorted(episode["gt_names"])
        gt_array = np.full((len(episode["times"]), len(gt_names), 3), np.nan, dtype=np.float64)
        gt_quat = np.full((len(episode["times"]), len(gt_names), 4), np.nan, dtype=np.float64)
        for row_index, (position_row, quat_row) in enumerate(
            zip(episode["gt_rows"], episode["gt_quat_rows"], strict=True)
        ):
            for column, name in enumerate(gt_names):
                position = position_row.get(name)
                if position is not None:
                    gt_array[row_index, column] = (
                        float(position.x),
                        float(position.y),
                        float(position.z),
                    )
                orientation = quat_row.get(name)
                if orientation is not None:
                    gt_quat[row_index, column] = [float(value) for value in orientation]
        np.savez_compressed(
            paths.steps,
            t=np.asarray(episode["times"], dtype=np.float64),
            joint_names=np.asarray(joint_names),
            joint_pos=np.asarray(episode["joint_pos"], dtype=np.float64),
            joint_vel=np.asarray(episode["joint_vel"], dtype=np.float64),
            gripper=np.asarray(episode["gripper"], dtype=np.float64),
            ee_xyz=np.asarray(episode["ee_xyz"], dtype=np.float64),
            ee_quat=np.asarray(episode["ee_quat"], dtype=np.float64),
            base_pose=np.asarray(episode["base_pose"], dtype=np.float64),
            gt_names=np.asarray(gt_names),
            gt_xyz=gt_array,
            gt_quat=gt_quat,
        )

        scenario: ScenarioSpec | None = episode["scenario"]
        meta = {
            "index": episode["index"],
            "seed": episode["seed"],
            "reset_seed": episode["reset_seed"],
            "pipeline": episode["pipeline"],
            "state_source": episode["state_source"],
            "scenario": None if scenario is None else asdict(scenario),
            "step_count": len(episode["times"]),
            "frame_count": int(episode["frame_count"]),
            "frame_hz": self.frame_hz if self.record_frames else 0.0,
            "contact_steps": int(episode["contact_steps"]),
            "gt_names": gt_names,
            "outcome": episode["outcome"],
        }
        paths.meta.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
        self._episode = None
        return paths

    # -- helpers -------------------------------------------------------------

    def _require_open(self) -> dict[str, Any]:
        if self._episode is None:
            raise RuntimeError("no episode is open; call open_episode() first.")
        return self._episode

    def _require_paths(self) -> EpisodePaths:
        if self._paths is None:
            raise RuntimeError("no episode paths recorded.")
        return self._paths

    def _write_frame(self, rgb: np.ndarray, frame_index: int) -> None:
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - depends on eval extra
            raise RuntimeError(
                "frame recording requires Pillow: `uv sync --extra eval` "
                "(or disable frames with record_frames=False / frame_hz=0)."
            ) from error
        paths = self._require_paths()
        assert paths is not None
        image = Image.fromarray(np.ascontiguousarray(rgb))
        image.save(paths.frames_dir / f"{frame_index:06d}.jpg", quality=self.jpeg_quality)


class RecordingBackend:
    """Backend decorator that feeds an :class:`EpisodeRecorder` without policy changes.

    Every ``step()`` captures the returned observation plus one ground-truth read;
    every ``apply()`` captures the action and its safety verdict. All other attribute
    access delegates to the wrapped backend, so the scripted policy is untouched.
    """

    def __init__(self, inner: Any, recorder: EpisodeRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self._last_observation_t = 0.0

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        observation = self._inner.reset(scenario, seed)
        self._last_observation_t = float(observation.t)
        self._recorder.on_reset(scenario, seed)
        self._recorder.record_step(observation, self._read_ground_truth())
        return observation

    def step(self, n_steps: int = 1) -> Observation:
        observation = self._inner.step(n_steps)
        self._last_observation_t = float(observation.t)
        self._recorder.record_step(observation, self._read_ground_truth())
        return observation

    def apply(self, action: Action) -> ApplyResult:
        result = self._inner.apply(action)
        self._recorder.record_action(self._last_observation_t, action, result)
        return result

    def diagnostic_apply(self, action: Action) -> ApplyResult:
        return self._inner.apply(action)

    def diagnostic_step(self) -> dict[str, object]:
        executor = getattr(self._inner, "diagnostic_step", None)
        if callable(executor):
            return executor()
        return {"confirmed_ticks": 1, "observation": self._inner.step(1)}

    def record_diagnostic_apply(self, action: Action, result: ApplyResult) -> None:
        self._recorder.record_action(self._last_observation_t, action, result)

    def record_diagnostic_step(self, observation: Observation, ground_truth: GroundTruth) -> None:
        self._last_observation_t = float(observation.t)
        self._recorder.record_step(observation, ground_truth)

    def get_ground_truth(self) -> GroundTruth:
        return self._inner.get_ground_truth()

    def stop(self) -> None:
        self._inner.stop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _read_ground_truth(self) -> GroundTruth | None:
        getter = getattr(self._inner, "get_ground_truth", None)
        if getter is None:
            return None
        return getter()
