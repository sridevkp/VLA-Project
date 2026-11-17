"""Fresh-session successful TRAIN collection with physical admission and immutable provenance."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from vlarobot.backends.webots.config import SETTINGS, SETTINGS_PATH
from vlarobot.core.types import Observation, ScenarioSpec
from vlarobot.eval.demo_audit import DemoAuditSpec, audit_demo
from vlarobot.eval.demos import DEFAULT_INSTRUCTION, DemoPlanEntry, build_demo_plan
from vlarobot.eval.recorder import EpisodeRecorder, RecordingBackend
from vlarobot.execution.episode_runner import DEFAULT_WORLD, launch_webots_session
from vlarobot.execution.scripted_baseline import ScriptedBaselinePolicy
from vlarobot.execution.scripted_can import GRIPPER_OPEN_TOLERANCE_M
from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES
from vlarobot.vla.settings import load_vla_settings


def write_json(path: Path, value: Any) -> None:
    """Replace a small resumable status document; recordings themselves stay immutable."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_spec(sample_steps: int) -> DemoAuditSpec:
    """Use existing physical criteria/geometry; these checks do not change eval success."""
    return DemoAuditSpec(
        SETTINGS.place_world,
        SETTINGS.place_tolerance_m,
        SETTINGS.can_place_z_tolerance_m,
        SETTINGS.can_tilt_limit_deg,
        SETTINGS.lift_min,
        SETTINGS.can_radius_m,
        SETTINGS.close_align_tol,
        SETTINGS.gripper_opening_limits[1],
        GRIPPER_OPEN_TOLERANCE_M,
        sample_steps * SETTINGS.time_step / 1000,
    )


class PhysicalRecordingBackend(RecordingBackend):
    """Append motor/pad measurements for admission, never as privileged policy features."""

    def __init__(self, inner: Any, recorder: EpisodeRecorder, physical_path: Path) -> None:
        super().__init__(inner, recorder)
        self.stream = physical_path.open("x", encoding="utf-8", buffering=1)

    def capture(self, observation: Observation) -> None:
        diagnostics = observation.robot.diagnostics
        fingers = diagnostics.get("finger_joint_sensor_rad", {})
        pads = diagnostics.get("finger_tip_world_xyz_m", {})
        if not isinstance(fingers, dict) or not isinstance(pads, dict):
            raise ValueError("physical finger and pad telemetry must be mappings")
        names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
        row = {
            "t": observation.t,
            "finger_positions_m": [fingers.get(n) for n in names],
            "pad_positions_world_m": [pads.get(n) for n in names],
        }
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        observation = super().reset(scenario, seed)
        self.capture(observation)
        return observation

    def step(self, n_steps: int = 1) -> Observation:
        observation = super().step(n_steps)
        self.capture(observation)
        return observation


def _attempt(
    entry: DemoPlanEntry,
    out: Path,
    *,
    webots: str,
    world: Path,
    sample_steps: int,
    startup_timeout_s: float,
) -> dict[str, Any]:
    destination = out / "attempts" / f"attempt-{entry.index:04d}"
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    recorder = EpisodeRecorder(
        destination, frame_hz=1000 / (SETTINGS.time_step * sample_steps), record_frames=True
    )
    paths = recorder.open_episode(0, entry.seed, "scripted", "oracle")
    wrapped = None
    outcome = None
    attempt_error = None
    try:
        with launch_webots_session(
            webots=webots,
            world=world,
            trace_dir=destination,
            trace_prefix="episode",
            headless=False,
            startup_timeout_s=startup_timeout_s,
        ) as session:
            wrapped = PhysicalRecordingBackend(
                session.client, recorder, destination / "episode-00.physical.jsonl"
            )
            policy = ScriptedBaselinePolicy(entry.scene, trace_path=destination / "policy.jsonl")
            outcome = dict(policy.execute(wrapped, entry.seed))
            outcome["instruction"] = entry.instruction
            recorder.record_outcome(outcome)
            recorder.close_episode()
    except Exception as error:  # one failed attempt must not lose the collection
        attempt_error = f"{type(error).__name__}: {error}"
        failure_outcome = {
            "success": False,
            "failure_state": "EXCEPTION",
            "failure_reason": f"{type(error).__name__}: {error}",
        }
        if recorder.is_open:
            outcome = failure_outcome
            recorder.record_outcome(outcome)
            recorder.close_episode()
    finally:
        if wrapped is not None:
            wrapped.stream.close()
    audit: dict[str, Any]
    try:
        audit = audit_demo(paths.meta, audit_spec(sample_steps))
        if (
            len(list(paths.frames_dir.glob("*.jpg")))
            != json.loads(paths.meta.read_text(encoding="utf-8"))["frame_count"]
        ):
            audit["accepted"] = False
            audit["reasons"].append("missing RGB files")
    except Exception as error:
        audit = {
            "accepted": False,
            "reasons": [f"incomplete audit: {type(error).__name__}: {error}"],
        }
    if attempt_error is not None:
        audit["accepted"] = False
        audit["reasons"].append(attempt_error)
    write_json(destination / "audit.json", audit)
    result = {
        "attempt": entry.index,
        "seed": entry.seed,
        "scene": asdict(entry.scene),
        "instruction": entry.instruction,
        "status": "accepted" if audit["accepted"] else "quarantined",
        "raw_dir": str(destination.resolve()),
        "outcome": outcome,
        "audit": audit,
        "wall_s": perf_counter() - started,
    }
    write_json(destination / "result.json", result)
    return result


def collect_verified_training(
    out: Path,
    baseline: Path,
    *,
    webots: str,
    world: Path = DEFAULT_WORLD,
    target: int | None = None,
    seed: int = 100,
    max_attempts: int | None = None,
    resume: bool = False,
    instruction: str = DEFAULT_INSTRUCTION,
    startup_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Collect successful TRAIN episodes; quarantine every failed/partial physical attempt."""
    settings = load_vla_settings()
    target = settings.successful_demo_target if target is None else target
    max_attempts = settings.max_collection_attempts if max_attempts is None else max_attempts
    if target <= 0 or max_attempts < target:
        raise ValueError("collection needs a positive target and enough allowed attempts")
    if settings.sample_steps != 1:
        raise ValueError("admission currently requires full-step RGB/sensor recording")
    baseline_status = json.loads((baseline / "campaign-status.json").read_text(encoding="utf-8"))
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((baseline / "provenance.json").read_text(encoding="utf-8"))
    if baseline_status["status"] != "completed" or summary["n_episodes"] != 24:
        raise ValueError("the repaired frozen 24-episode baseline must finish before collection")
    if summary["measurement"] != "measured" or summary["split"] != "train":
        raise ValueError(
            "collection needs a measured TRAIN baseline, not fake or held-out evidence"
        )
    if summary["success"]["successes"] <= 0:
        raise ValueError("zero-success repaired baseline: no successful-demonstration collection")
    project = SETTINGS_PATH.parent.parent
    if world.resolve() != DEFAULT_WORLD.resolve():
        raise ValueError("collection must use the repaired baseline's frozen world")
    registration = os.environ.get(
        "VLAROBOT_URDF_REGISTRATION_FILE", SETTINGS.urdf_base_registration_file
    )
    if registration is None:
        raise ValueError("collection requires the repaired static registration")
    registration_hash = hashlib.sha256(
        (SETTINGS_PATH.parent / registration).read_bytes()
    ).hexdigest()
    if (
        registration_hash
        != provenance["files_sha256"]["configs/calibration/urdf-base-train-v1.yaml"]
    ):
        raise ValueError("runtime registration differs from the completed repaired baseline")
    protected = [
        p
        for p in provenance["files_sha256"]
        if p.startswith("configs/")
        or p.startswith("src/vlarobot/backends/")
        or p.endswith("scripted_can.py")
        or p.endswith("scripted_baseline.py")
        or p.startswith("worlds/")
    ]
    for name in protected:
        actual = hashlib.sha256((project / name).read_bytes()).hexdigest()
        if actual != provenance["files_sha256"][name]:
            raise ValueError(f"collection source changed since repaired baseline: {name}")
    plan = build_demo_plan(max_attempts, base_seed=seed, split="train", instruction=instruction)
    plan_value = json.loads(json.dumps([asdict(e) for e in plan]))
    identity = {
        "target_successes": target,
        "seed_start": seed,
        "attempt_plan": plan_value,
        "baseline": str(baseline.resolve()),
        "baseline_success": summary["success"],
        "baseline_provenance_sha256": hashlib.sha256(
            (baseline / "provenance.json").read_bytes()
        ).hexdigest(),
        "collection_settings": asdict(settings),
        "collection_code_sha256": {
            name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in [
                "src/vlarobot/execution/demo_collection.py",
                "src/vlarobot/eval/demo_audit.py",
                "src/vlarobot/vla/schema.py",
            ]
        },
        "camera_key": CAMERA_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "frame_hz": 1000 / SETTINGS.time_step,
        "sample_dt_s": SETTINGS.time_step / 1000,
        "rgb_decoder_sha256": provenance["files_sha256"]["src/vlarobot/backends/webots/images.py"],
        "registration_sha256": provenance["files_sha256"][
            "configs/calibration/urdf-base-train-v1.yaml"
        ],
        "measurement": "measured",
        "source_state": "oracle scripted expert; learned inputs exclude GT",
        "backend": "webots",
        "instruction": instruction,
        "baseline_and_dataset_are_separate": True,
    }
    identity = json.loads(json.dumps(identity))
    if out.exists():
        if (
            not resume
            or json.loads((out / "provenance.json").read_text(encoding="utf-8")) != identity
        ):
            raise FileExistsError(
                "existing collection requires resume with identical provenance/plan"
            )
    else:
        out.mkdir(parents=True)
        write_json(out / "provenance.json", identity)
    rows = []
    manifest = out / "attempts.jsonl"
    if manifest.exists():
        rows = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
        ]
    completed = {r["attempt"] for r in rows}
    pending = [e for e in plan if e.index not in completed]
    started = perf_counter()

    def checkpoint(status: str) -> dict[str, Any]:
        accepted = sorted(
            (r for r in rows if r["status"] == "accepted"), key=lambda r: r["attempt"]
        )[:target]
        write_json(
            out / "accepted.json",
            {
                "episodes": accepted,
                "count": len(accepted),
                "target": target,
                "training_approved": False,
            },
        )
        state = {
            "status": status,
            "attempted": len(rows),
            "successful_episodes": len(accepted),
            "quarantined_attempts": sum(r["status"] != "accepted" for r in rows),
            "target": target,
            "wall_s_this_invocation": perf_counter() - started,
            "training_approved": False,
        }
        write_json(out / "collection-status.json", state)
        return state

    with ThreadPoolExecutor(max_workers=settings.collection_workers) as pool:
        for start in range(0, len(pending), settings.collection_workers):
            if checkpoint("running")["successful_episodes"] >= target:
                break
            batch = pending[start : start + settings.collection_workers]
            jobs = []
            for entry in batch:
                destination = out / "attempts" / f"attempt-{entry.index:04d}"
                if destination.exists():
                    if (destination / "result.json").exists():
                        recovered = json.loads(
                            (destination / "result.json").read_text(encoding="utf-8")
                        )
                        if recovered["scene"] != json.loads(json.dumps(asdict(entry.scene))):
                            raise ValueError(
                                "recovered attempt does not match the frozen TRAIN plan"
                            )
                        rows.append(recovered)
                        continue
                    rows.append(
                        {
                            "attempt": entry.index,
                            "status": "interrupted",
                            "raw_dir": str(destination.resolve()),
                            "seed": entry.seed,
                            "reason": "preserved incomplete attempt found while resuming",
                        }
                    )
                else:
                    jobs.append(
                        pool.submit(
                            _attempt,
                            entry,
                            out,
                            webots=webots,
                            world=world,
                            sample_steps=settings.sample_steps,
                            startup_timeout_s=startup_timeout_s,
                        )
                    )
            # Consume planned order: dataset membership does not depend on wall-clock completion.
            for job in jobs:
                rows.append(job.result())
            rows.sort(key=lambda r: r["attempt"])
            temporary_manifest = manifest.with_suffix(".tmp")
            with temporary_manifest.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            temporary_manifest.replace(manifest)
            checkpoint("running")
    accepted = sum(r["status"] == "accepted" for r in rows)
    return checkpoint("completed" if accepted >= target else "attempt_limit")
