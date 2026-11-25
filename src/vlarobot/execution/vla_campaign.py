"""Actual frozen VLA-only Webots campaign, isolated model process and physical scoring."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import ARM_JOINTS, SETTINGS, SETTINGS_PATH
from vlarobot.core.types import Observation, Pose3D, ScenarioSpec
from vlarobot.eval.runner import (
    EvalRunConfig,
    _completed_recording,
    build_episode_plan,
    run_evaluation,
)
from vlarobot.eval.vla_metrics import final_can_outcome, paired_vla_gate
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.episode_runner import (
    DEFAULT_WORLD,
    _reserve_free_port,
    launch_webots_session,
)
from vlarobot.execution.vla_episode import VLAEpisodePolicy
from vlarobot.vla.action_adapter import ActionAdapter
from vlarobot.vla.checkpoint_export import require_trained_export
from vlarobot.vla.model_service import ModelClient
from vlarobot.vla.settings import load_vla_settings


def make_vla_adapter(
    expected_registration_sha256: str | None = None, *, recovery_ik: bool = False
) -> ActionAdapter:
    """Use registered bounded full-pose IK, with the backend retaining its live motor veto."""
    import xml.etree.ElementTree as ET

    from vlarobot.backends.webots.config import SAFETY_PROFILE, URDF_PATH
    from vlarobot.backends.webots.kinematics import Kinematics
    from vlarobot.backends.webots.pose_solver import BoundedPoseSolver, GateNormalizedPoseSolver
    from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor

    kin = Kinematics()
    if kin.registration_sha256 is None:
        raise ValueError("VLA evaluation requires accepted static registration")
    if (
        expected_registration_sha256 is not None
        and kin.registration_sha256 != expected_registration_sha256
    ):
        raise ValueError("VLA runtime registration differs from the accepted repaired baseline")
    tree = ET.parse(URDF_PATH)
    velocities = {}
    for name in ARM_JOINTS:
        limit = tree.find(f'./joint[@name="{name}"]/limit')
        if limit is None:
            raise ValueError(f"URDF safety limits unavailable: {name}")
        velocities[name] = float(limit.attrib["velocity"])
    limits = SafetyLimits(
        joint_position={n: kin.bounds(n) for n in ARM_JOINTS},
        joint_velocity=velocities,
        workspace_min=SAFETY_PROFILE.workspace_min,
        workspace_max=SAFETY_PROFILE.workspace_max,
        gripper_opening=SETTINGS.gripper_opening_limits,
        command_horizon=SETTINGS.safety_command_horizon,
    )
    settings = load_vla_settings()
    solver = (
        GateNormalizedPoseSolver(
            kin,
            max_joint_step_rad=SETTINGS.max_joint_step_rad,
            position_tolerance_m=settings.ik_position_tolerance_m,
            orientation_tolerance_rad=settings.ik_orientation_tolerance_rad,
        )
        if recovery_ik
        else BoundedPoseSolver(kin, max_joint_step_rad=SETTINGS.max_joint_step_rad)
    )
    return ActionAdapter(
        solver,
        SafetyMonitor(limits),
        position_tolerance_m=settings.ik_position_tolerance_m,
        orientation_tolerance_rad=settings.ik_orientation_tolerance_rad,
    )


class CanEpisodeScorer:
    """Ground-truth access is exclusively initial/final evaluation, never proposer input."""

    def __init__(self) -> None:
        self.initial: Pose3D | None = None

    def begin(self, backend: Any) -> None:
        self.initial = backend.get_ground_truth().object_poses["CAN"]

    def __call__(
        self,
        backend: Any,
        observation: Observation,
        home: bool,
        state: str | None,
        reason: str | None,
    ) -> dict[str, Any]:
        if self.initial is None:
            raise RuntimeError("initial CAN metric reference missing")
        return final_can_outcome(
            self.initial,
            backend.get_ground_truth().object_poses["CAN"],
            place_world_m=SETTINGS.place_world,
            distance_tolerance_m=SETTINGS.place_tolerance_m,
            height_tolerance_m=SETTINGS.can_place_z_tolerance_m,
            tilt_limit_deg=SETTINGS.can_tilt_limit_deg,
            home_reached=home,
            failure_state=state,
            failure_reason=reason,
        )


def run_vla_campaign(
    export: Path,
    out: Path,
    baseline: Path,
    *,
    model_python: Path,
    webots: str,
    resume: bool = False,
    recovery_v2: bool = False,
    diagnostic_indices: tuple[int, ...] = (),
    recovery_ik: bool = False,
    endpoint_gripper: bool = False,
) -> dict[str, Any]:
    """Run all TRAIN 8×3 pairs; refuse untrained fixtures and preserve every completed episode."""
    if (recovery_ik or endpoint_gripper) and not recovery_v2:
        raise ValueError("recovery IK requires V2 execution")
    approval = require_trained_export(export)
    baseline_provenance = json.loads((baseline / "provenance.json").read_text(encoding="utf-8"))
    project = SETTINGS_PATH.parent.parent
    if diagnostic_indices and (not recovery_v2 or resume):
        raise ValueError("TRAIN subset requires the explicitly versioned V2 diagnostic")
    evaluated_backend = "b1db6494ef0cb086c0c832622034b423932f3e3984df6d4d35355112984047cf"
    evaluated_expert = "921da033d497b86bb1a4f3dd9b3005dfaf733a7ac3fef245df3cc367bf8d0857"
    v1_source_pins = {
        "src/vlarobot/backends/webots/backend.py": evaluated_backend,
        "src/vlarobot/execution/scripted_can.py": evaluated_expert,
    }
    for name, expected_hash in baseline_provenance["files_sha256"].items():
        current_hash = hashlib.sha256((project / name).read_bytes()).hexdigest()
        if current_hash != expected_hash and (
            not recovery_v2 or current_hash != v1_source_pins.get(name)
        ):
            raise ValueError(f"frozen baseline source changed before VLA evaluation: {name}")
    adapter = make_vla_adapter(
        baseline_provenance["files_sha256"]["configs/calibration/urdf-base-train-v1.yaml"],
        recovery_ik=recovery_ik,
    )
    original = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    baseline_plan = json.loads((baseline / "plan.json").read_text(encoding="utf-8"))
    plan = build_episode_plan("train", 0)
    expected = json.loads(
        json.dumps([{"index": e.index, "seed": e.seed, "scene": asdict(e.scene)} for e in plan])
    )
    if baseline_plan["episodes"] != expected or original["success"]["successes"] == 0:
        raise ValueError("paired repaired baseline required before VLA gate")
    if diagnostic_indices and (
        len(set(diagnostic_indices)) != len(diagnostic_indices)
        or not set(diagnostic_indices) <= {e.index for e in plan}
    ):
        raise ValueError("diagnostic indices must be distinct frozen TRAIN trial indices")
    active_plan = (
        tuple(e for e in plan if e.index in diagnostic_indices) if diagnostic_indices else plan
    )
    manifest_sha = hashlib.sha256((export / "export-manifest.json").read_bytes()).hexdigest()
    horizon = max(approval["episode_frame_counts"]) - 1
    identity = {
        "pipeline": "vla_only",
        **(
            {
                "execution_profile": "v2_bounded_gripper_verified_home",
                "diagnostic_indices": list(diagnostic_indices),
                "gate_normalized_ik_fallback": recovery_ik,
                "endpoint_gripper": endpoint_gripper,
            }
            if recovery_v2
            else {}
        ),
        "model_manifest_sha256": manifest_sha,
        "plan": expected,
        "max_episode_steps": horizon,
        "horizon_source": "longest approved TRAIN demonstration",
        "home_tolerance_rad": SETTINGS.home_settle_tolerance_rad,
        "action_dt_s": SETTINGS.time_step / 1000,
        "controller_sha256": {
            name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in [
                "src/vlarobot/execution/vla_episode.py",
                "src/vlarobot/execution/vla_campaign.py",
                "src/vlarobot/vla/action_adapter.py",
                "src/vlarobot/vla/model_service.py",
                "src/vlarobot/backends/webots/pose_solver.py",
            ]
        },
        "baseline_summary_sha256": hashlib.sha256(
            (baseline / "summary.json").read_bytes()
        ).hexdigest(),
    }
    if out.exists():
        if (
            not resume
            or json.loads((out / "provenance.json").read_text(encoding="utf-8")) != identity
        ):
            raise FileExistsError("preserve existing VLA campaign; resume needs identical identity")
    else:
        out.mkdir(parents=True)
        write_json(out / "provenance.json", identity)
    endpoint = f"tcp://127.0.0.1:{_reserve_free_port()}"
    command = [
        str(model_python),
        "-u",
        "-m",
        "vlarobot.cli",
        "model-serve",
        "--export",
        str(export.resolve()),
        "--endpoint",
        endpoint,
    ]
    started = perf_counter()
    client = None
    with (out / "model-server.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while client is None:
                if process.poll() is not None or perf_counter() - started > 180:
                    raise RuntimeError("local model service failed to start; preserve logs")
                try:
                    client = ModelClient(endpoint, timeout_ms=1000)
                except Exception:
                    sleep(0.5)
            # Inference gets its full timeout after the short startup probes.
            client.close()
            client = ModelClient(endpoint)
            if client.metadata["export_manifest_sha256"] != manifest_sha:
                raise ValueError("model service loaded a different checkpoint")
            write_json(out / "model-metadata.json", client.metadata)
            settings = load_vla_settings()
            config = EvalRunConfig(
                out_dir=out,
                pipeline="vla_only",
                split="train",
                base_seed=0,
                backend="webots",
                measurement="measured",
                state_source="robot_rgb",
                record_frames=True,
                frame_hz=1.0,
            )
            diagnostic_rows = []
            for entry in active_plan:
                if resume and _completed_recording(out, entry, "vla_only") is not None:
                    continue
                trace = out / f"episode-{entry.index:02d}.latency.jsonl"
                if trace.exists():
                    raise FileExistsError("preserve partial VLA trace before explicit recovery")
                scorer = CanEpisodeScorer()

                def factory(
                    scene: ScenarioSpec, trace: Path = trace, scorer: CanEpisodeScorer = scorer
                ) -> VLAEpisodePolicy:
                    assert client is not None
                    return VLAEpisodePolicy(
                        scene,
                        client,
                        adapter,
                        instruction="Pick up the can and place it at the target location.",
                        max_steps=horizon,
                        max_chunk_actions=settings.max_chunk_actions,
                        home_tolerance_rad=SETTINGS.home_settle_tolerance_rad,
                        arm_names=tuple(ARM_JOINTS),
                        trace=trace,
                        score=scorer,
                        begin_metrics=scorer.begin,
                        recovery_v2=recovery_v2,
                        endpoint_gripper=endpoint_gripper,
                    )

                with launch_webots_session(
                    webots=webots,
                    world=DEFAULT_WORLD,
                    trace_dir=out,
                    trace_prefix=f"vla-ep{entry.index:02d}",
                    headless=False,
                ) as session:
                    report = run_evaluation([entry], session.client, config, policy_factory=factory)
                    if diagnostic_indices:
                        diagnostic_rows.extend(report.rows)
                        write_json(
                            out / "diagnostic-results.json",
                            {
                                "scope": "V2 TRAIN diagnostic, not a gate result",
                                "rows": diagnostic_rows,
                            },
                        )
            if diagnostic_indices:
                return {
                    "status": "diagnostic_completed",
                    "trials": len(diagnostic_rows),
                    "successes": sum(row["success"] is True for row in diagnostic_rows),
                }

            if any(_completed_recording(out, e, "vla_only") is None for e in plan):
                raise ValueError("no gate decision without all 24 physical recordings")

            def aggregation_only(scene: ScenarioSpec) -> VLAEpisodePolicy:
                raise RuntimeError("refusing to replace missing physical VLA recordings")

            assembled = run_evaluation(
                plan,
                FakeBackend(),
                replace(config, resume=True),
                policy_factory=aggregation_only,
            )
            gate = paired_vla_gate(original, assembled.summary)
            assembled.summary["week2_gate"] = {**gate, "baseline_rate": original["success"]["rate"]}
            write_json(out / "summary.json", assembled.summary)
            write_json(out / "proposer-role.json", gate)
            latencies = [
                json.loads(line)
                for p in out.glob("*.latency.jsonl")
                for line in p.read_text(encoding="utf-8").splitlines()
            ]
            times = np.asarray([r["inference_wall_s"] for r in latencies])
            result = {
                "status": "completed",
                "success": assembled.summary["success"],
                "gate": gate,
                "inference_chunks": len(times),
                "inference_latency_s": {
                    "mean": float(times.mean()),
                    "p50": float(np.median(times)),
                    "p95": float(np.percentile(times, 95)),
                }
                if len(times)
                else None,
                "campaign_wall_s": perf_counter() - started,
            }
            write_json(out / "vla-gate.json", result)
            return result
        finally:
            if client is not None:
                try:
                    client.request("vla_shutdown", {})
                except Exception:
                    pass
                client.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=30)
