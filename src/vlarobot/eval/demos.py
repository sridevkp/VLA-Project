"""Scripted-expert demo collection for VLA fine-tuning (ADR-0029, M8 prep).

Produces per-episode recordings in the existing LeRobot-convertible EpisodeRecorder layout
(npz/jsonl/jpg/meta) plus a dataset manifest. Action representation for training (e.g.
7-D EE-delta + gripper through the IK/safety layer) is derived/post-processed separately as
needed; this module records the applied actions and state streams as executed.

Rendering/observation optimizations are not applied here by default; equivalence proofs are
required before using ``observe_images=false``/no-rendering for non-demo runs (ADR-0029).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from vlarobot.core.types import ScenarioSpec
from vlarobot.eval.recorder import EpisodeRecorder, RecordingBackend
from vlarobot.eval.runner import EvalEpisode, PolicyFactory, _plan_scene_id
from vlarobot.eval.scenarios import SPLITS, generate_scenes
from vlarobot.eval.settings import EvalSettings, load_eval_settings

DEFAULT_INSTRUCTION = "Pick up the can and place it at the target location."


@dataclass(frozen=True)
class DemoPlanEntry:
    index: int
    scene: ScenarioSpec
    seed: int
    instruction: str


def build_demo_plan(
    n: int,
    *,
    base_seed: int = 0,
    split: str = "train",
    instruction: str = DEFAULT_INSTRUCTION,
    settings: EvalSettings | None = None,
    allow_test: bool = False,
    max_seed_blocks: int = 50,
) -> tuple[DemoPlanEntry, ...]:
    """Plan ``n`` demo episodes by iterating seed blocks over ``split`` scenes.

    ``test`` is fail-closed unless ``allow_test=True`` (ADR-0029). Scenes repeat across seed
    blocks with different (seed, geometry) draws, which provides perturbed poses.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; available: {', '.join(SPLITS)}")
    resolved = settings or load_eval_settings()
    plan: list[DemoPlanEntry] = []
    for block in range(max_seed_blocks):
        if len(plan) >= n:
            break
        seed = base_seed + block
        scenes = generate_scenes(split, seed, allow_test=allow_test, sizes=resolved.sizes)
        for scene in scenes:
            if len(plan) >= n:
                break
            plan.append(
                DemoPlanEntry(index=len(plan), scene=scene, seed=seed, instruction=instruction)
            )
    if len(plan) < n:
        raise ValueError(
            f"could not build {n} demo episodes from split {split!r} "
            f"(built {len(plan)} over {max_seed_blocks} seed blocks)."
        )
    return tuple(plan)


@dataclass(frozen=True)
class DemoResult:
    manifest_path: Path
    episodes: int
    successes: int
    failures: int
    total_wall_s: float


def collect_demos(
    plan: Sequence[DemoPlanEntry],
    backend: Any,
    out_dir: Path,
    *,
    policy_factory: PolicyFactory,
    state_source: str = "oracle",
    pipeline: str = "scripted",
    record_frames: bool = True,
    frame_hz: float = 10.0,
    resume: bool = False,
) -> DemoResult:
    """Run scripted demos and write a manifest alongside EpisodeRecorder outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = EpisodeRecorder(out_dir, frame_hz=frame_hz, record_frames=record_frames)
    manifest_path = out_dir / "manifest.jsonl"
    mode = "a" if resume and manifest_path.exists() else "w"
    successes = 0
    failures = 0
    run_started = perf_counter()

    # Build set of completed indices when resuming.
    completed: set[int] = set()
    if resume and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("status") == "completed":
                completed.add(int(row.get("index", -1)))

    with manifest_path.open(mode, encoding="utf-8") as manifest:
        for entry in plan:
            if resume and entry.index in completed:
                continue
            started = perf_counter()
            paths = recorder.open_episode(entry.index, entry.seed, pipeline, state_source)
            status = "completed"
            error: str | None = None
            outcome: dict[str, Any] | None = None
            try:
                policy = policy_factory(entry.scene)
                result = policy.execute(RecordingBackend(backend, recorder), entry.seed)
                if not isinstance(result, Mapping) or "success" not in result:
                    raise ValueError("policy outcome must be a mapping with a 'success' key.")
                outcome = dict(result)
                recorder.record_outcome(outcome)
                recorder.close_episode()
                if outcome.get("success"):
                    successes += 1
                else:
                    failures += 1
            except Exception as exc:  # noqa: BLE001 - one demo must not lose the run
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                if recorder.is_open:
                    recorder.record_outcome(
                        {"success": False, "failure_state": "EXCEPTION", "failure_reason": error}
                    )
                    recorder.close_episode()
                failures += 1
            row = {
                "index": entry.index,
                "seed": entry.seed,
                "scene": asdict(entry.scene),
                "scene_id": _plan_scene_id(EvalEpisode(entry.index, entry.scene, entry.seed)),
                "instruction": entry.instruction,
                "pipeline": pipeline,
                "state_source": state_source,
                "status": status,
                "outcome": outcome,
                "error": error,
                "wall_s": perf_counter() - started,
                "meta": str(paths.meta),
                "steps": str(paths.steps),
                "actions": str(paths.actions),
                "frames_dir": str(paths.frames_dir),
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()

    return DemoResult(
        manifest_path=manifest_path,
        episodes=len(plan),
        successes=successes,
        failures=failures,
        total_wall_s=perf_counter() - run_started,
    )
