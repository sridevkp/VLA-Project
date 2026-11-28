"""Command-line entry point for project tools."""

import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, overload

import numpy as np
import typer

from vlarobot.backends.webots.config import ARM_JOINTS, SIM_KINEMATICS_PATH, TIME_STEP, TORSO_JOINT
from vlarobot.backends.webots.sim_kinematics import (
    EXPORT_POSTURE_FIT_TOL_M,
    build_chain,
    finalize_verified_snapshot,
    verify_export_posture,
)
from vlarobot.backends.webots.smoke import create_webots_robot, run_smoke
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import DEFAULT_ENDPOINT
from vlarobot.core.types import JointTarget, Observation, ScenarioSpec
from vlarobot.eval.scenarios import SCENARIOS, get_scenarios
from vlarobot.execution.episode_runner import (
    DEFAULT_WORLD as RUNNER_DEFAULT_WORLD,
)
from vlarobot.execution.episode_runner import (
    RunnerSettings,
    determinism_report,
    format_determinism_report,
    format_summary_table,
    launch_webots_session,
    resolve_webots,
)
from vlarobot.execution.episode_runner import (
    run_episodes as run_episode_batch,
)
from vlarobot.execution.kinematics_validation import (
    DEFAULT_HORIZON_S,
    DEFAULT_VELOCITY_LIMIT_RAD_S,
    LINK_MAX_ACCEPT_M,
    LINK_MEAN_ACCEPT_M,
    SETTLE_DELTA_RAD,
    SETTLE_STEPS,
    SETTLE_SUBMOVE_CAP,
    JointSample,
    estimate_step_budget,
    evaluate_sweep,
    feasible_pose_plan,
    format_report,
    plan_submoves,
    pose_limits,
    sample_is_safe,
    submove_bound,
)
from vlarobot.execution.scripted_can import ResetProbeResult, ScriptedCanPolicy
from vlarobot.execution.trace_compare import compare_traces, joint_diagnostics

RUN_EPISODES_DEFAULT_WORLD = Path("worlds/factory.wbt")

app = typer.Typer(no_args_is_help=True)


@app.command("eval-vla")
def eval_vla(
    export: Annotated[Path, typer.Option(help="Trained portable export with optimizer evidence.")],
    out: Annotated[Path, typer.Option(help="New versioned VLA-only evaluation directory.")],
    model_python: Annotated[Path, typer.Option(help="Isolated installed ML Python executable.")],
    baseline: Path = Path("runs/eval/baseline-train-s0-fresh-v2-registered"),
    webots: str | None = None,
    resume: bool = False,
    recovery_v2: bool = False,
    recovery_ik: bool = False,
    endpoint_gripper: bool = False,
    diagnostic_index: Annotated[list[int] | None, typer.Option()] = None,
) -> None:
    """Run actual frozen TRAIN 8×3 VLA-only evaluation; never use compatibility fixtures."""
    from vlarobot.execution.vla_campaign import run_vla_campaign

    previous = os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE")
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = "calibration/urdf-base-train-v1.yaml"
    try:
        result = run_vla_campaign(
            export,
            out,
            baseline,
            model_python=model_python,
            webots=resolve_webots(webots),
            resume=resume,
            recovery_v2=recovery_v2,
            recovery_ik=recovery_ik,
            endpoint_gripper=endpoint_gripper,
            diagnostic_indices=tuple(diagnostic_index or ()),
        )
    finally:
        if previous is None:
            os.environ.pop("VLAROBOT_URDF_REGISTRATION_FILE", None)
        else:
            os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = previous
    typer.echo(json.dumps(result, indent=2))


@app.command("model-serve")
def model_serve(
    export: Annotated[Path, typer.Option(help="Portable local policy export.")],
    endpoint: str = "tcp://127.0.0.1:5560",
    device: str = "cpu",
) -> None:
    """Serve local SmolVLA inference in the isolated model process; never actuate a robot."""
    from vlarobot.vla.model_service import PortableChunkModel, serve_model

    serve_model(endpoint, PortableChunkModel(export, device=device))


@app.command("train-vla")
def train_vla(
    dataset: Annotated[Path, typer.Option(help="Fully approved successful TRAIN dataset.")],
    out: Annotated[Path, typer.Option(help="Persistent versioned training output.")],
    steps: int = 20000,
    batch_size: int = 8,
    seed: int = 0,
    save_freq: int = 1000,
) -> None:
    """Train approved data on authorized CUDA; record actual optimizer post-step evidence."""
    from vlarobot.vla.training import launch_approved_training

    launch_approved_training(
        dataset, out, steps=steps, batch_size=batch_size, seed=seed, save_freq=save_freq
    )


@app.command("vla-smoke")
def vla_smoke(
    out: Annotated[Path, typer.Option(help="New installed-policy evidence directory.")],
    recording: Annotated[
        Path, typer.Option(help="Correctly decoded measured recording directory.")
    ],
    seed: int = 0,
) -> None:
    """Run actual installed pretrained SmolVLA inference with TIAGo features (no robot)."""
    from vlarobot.vla.smoke import run_installed_forward

    report = run_installed_forward(out, recording, seed=seed)
    typer.echo(f"VLA installed forward: chunk={report['chunk_shape']} out={out}")


@app.command("export-demos")
def export_demos(
    collection: Annotated[Path, typer.Option(help="Audited successful TRAIN collection.")],
    out: Annotated[Path, typer.Option(help="New versioned native LeRobot dataset directory.")],
    repo_id: Annotated[str, typer.Option(help="Local dataset identity, e.g. tiago/train-v1.")],
) -> None:
    """Re-audit and export successful measured episodes using installed LeRobot."""
    from vlarobot.execution.dataset_validation import export_collection

    report = export_collection(collection, out, repo_id=repo_id)
    typer.echo(f"Exported {report['episodes']} successful episodes; training approval pending.")


@app.command("vla-dataset-check")
def vla_dataset_check(
    dataset: Annotated[Path, typer.Option(help="Native successful TRAIN dataset.")],
    smoke: Annotated[Path, typer.Option(help="Completed pinned installed-forward evidence.")],
    seed: int = 0,
) -> None:
    """Exercise actual dataset statistics in an installed policy forward pass (no robot)."""
    from vlarobot.execution.dataset_validation import validate_native_forward

    report = validate_native_forward(dataset, smoke, seed=seed)
    typer.echo(
        f"Native forward verified: {report['chunk_shape']}; episodes={len(report['episodes'])}"
    )


@app.command("grasp-report")
def grasp_report(
    diag: Annotated[Path, typer.Option(help="Completed physical diagnostic JSONL.")],
    trace: Annotated[Path, typer.Option(help="Policy state-boundary JSONL.")],
    out: Annotated[Path, typer.Option(help="New JSON report path (must not exist).")],
) -> None:
    """Measure recorded pad/can geometry and lift (privileged diagnostic, read-only)."""
    from vlarobot.eval.grasp_diagnostics import summarize_grasp

    report = summarize_grasp(diag, trace)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    typer.echo(f"PRIVILEGED DIAGNOSTIC: {report['sample_count']} samples; out={out}")


@app.callback()
def main() -> None:
    """Project commands."""


@app.command()
def smoke() -> None:
    """Connect to Webots, read a camera frame, and advance ten simulation steps."""
    result = run_smoke(create_webots_robot())
    typer.echo(
        f"sim_time={result.sim_time:.3f}s "
        f"image_shape=({result.image_height}, {result.image_width}, 4)"
    )


def run_webots_sim_server(endpoint: str = DEFAULT_ENDPOINT) -> None:
    """Run the Webots-side server; also used by the Webots controller shim."""
    from vlarobot.backends.webots.server import run_webots_sim_server as run_server

    run_server(endpoint)


@app.command("sim-server")
def sim_server(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ REP bind endpoint."),
) -> None:
    """Serve lockstep Webots requests; run from the Webots extern controller."""
    run_webots_sim_server(endpoint)


@app.command()
def brain(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    steps: int = typer.Option(10, min=1, help="Number of lockstep simulation steps."),
) -> None:
    """Connect to the simulator, reset the default scenario, and step the brain-side client."""
    client = ZmqBackendClient(endpoint)
    try:
        client.ping()
        observation = client.reset(ScenarioSpec(), seed=0)
        for _ in range(steps):
            observation = client.step()
        typer.echo(f"sim_time={observation.t:.3f}s image_shape={observation.rgb.shape}")
    finally:
        client.close()


@app.command("can-regression")
def can_regression(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    seeds: int = typer.Option(10, min=1, help="Number of seeded CAN episodes to run."),
    trace_log: Annotated[
        Path | None, typer.Option(help="Append per-state trace records to this JSONL.")
    ] = None,
    pregrasp_route: Annotated[
        Literal["interpolated", "safe_via"] | None,
        typer.Option(help="PREGRASP route; defaults to configs/robot.yaml."),
    ] = None,
    pregrasp_realign: Annotated[
        bool | None,
        typer.Option(
            "--pregrasp-realign/--no-pregrasp-realign",
            help="Re-align PREGRASP after base tilt; defaults to configs/robot.yaml.",
        ),
    ] = None,
    closed_loop_waypoints: Annotated[
        bool | None,
        typer.Option(
            "--closed-loop-waypoints/--no-closed-loop-waypoints",
            help=(
                "Recompute DESCEND/LIFT targets per waypoint from the live base pose; "
                "defaults to configs/robot.yaml."
            ),
        ),
    ] = None,
    stop_after: Annotated[
        Literal["PREGRASP", "LIFT"] | None,
        typer.Option(help="Stop this watched diagnostic run after PREGRASP or LIFT."),
    ] = None,
    timings: bool = typer.Option(
        False,
        "--timings",
        help="Print opt-in wall-time counters for IK, transport, trace writes, and diagnostics.",
    ),
) -> None:
    """Run a brain-side scripted CAN policy over repeated seeds and summarize the outcome."""
    client = ZmqBackendClient(endpoint)
    policy = ScriptedCanPolicy(
        trace_path=trace_log,
        pregrasp_route=pregrasp_route,
        pregrasp_realign=pregrasp_realign,
        closed_loop_waypoints=closed_loop_waypoints,
        stop_after=stop_after,
        timing_enabled=timings,
    )
    results = []
    try:
        client.ping()
        for seed in range(seeds):
            result = policy.execute(client, seed)
            results.append(result)
            typer.echo(
                f"seed={seed} success={result.success} rejections={result.rejections} "
                f"final_distance={result.final_target_distance:.3f}m state={result.final_state} "
                f"failure_state={result.failure_state} failure_reason={result.failure_reason} "
                f"pregrasp_realign={result.pregrasp_realign} "
                f"closed_loop_waypoints={result.closed_loop_waypoints} "
                f"home_reached={result.home_reached} "
                f"place_distance_ok={result.place_distance_ok} "
                f"can_height_ok={result.can_height_ok} can_tilt_ok={result.can_tilt_ok} "
                f"can_height_error_m={_optional_float(result.can_height_error_m, 4)} "
                f"can_tilt_deg={_optional_float(result.can_tilt_deg, 2)} "
                f"max_base_drift_m={_optional_float(result.max_base_drift_m, 6)} "
                f"max_base_drift_yaw_deg={_optional_float(result.max_base_drift_yaw_deg, 3)} "
                f"max_base_tilt_deg={_optional_float(result.max_base_tilt_deg, 3)} "
                f"torso_start_rad={_optional_float(result.torso_start_rad, 6)} "
                f"torso_end_rad={_optional_float(result.torso_end_rad, 6)}"
            )
            for state, sim_seconds in policy.state_sim_durations.items():
                settle_seconds = policy.state_settle_durations.get(state, 0.0)
                wall_seconds = policy.state_wall_durations.get(state, 0.0)
                typer.echo(
                    f"  timing {state}: sim={sim_seconds:.3f}s "
                    f"settle={settle_seconds:.3f}s wall={wall_seconds:.3f}s"
                )
            if timings:
                _print_timing_summary(policy, client)
        successes = sum(1 for result in results if result.success)
        rejections = sum(result.rejections for result in results)
        typer.echo(f"summary success={successes}/{seeds} rejections={rejections}")
    finally:
        client.close()


@app.command()
def eval(  # noqa: A001 - CLI command name is fixed by ADR-0029
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            help="Arm to run: 'baseline'/'scripted' (week 1). Other arms fail loud.",
        ),
    ] = "baseline",
    split: Annotated[
        Literal["train", "validation"],
        typer.Option(help="Scene split. The test split is fail-closed until week 4 (ADR-0029)."),
    ] = "train",
    n: Annotated[
        int,
        typer.Option("--n", min=0, help="Episodes to run (0 = every planned episode)."),
    ] = 0,
    seed: Annotated[
        int, typer.Option(help="Base seed (baseline repeats use seed, seed+1, ...).")
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory (default: runs/eval/<pipeline>-<split>-s<seed>)."),
    ] = None,
    backend: Annotated[
        Literal["fake", "webots"],
        typer.Option(help="Episode backend: 'fake' (fast, CI) or 'webots' (measured)."),
    ] = "fake",
    webots: Annotated[
        str | None,
        typer.Option("--webots", help="Webots executable (defaults to WEBOTS_HOME, then PATH)."),
    ] = None,
    world: Annotated[
        Path | None,
        typer.Option(help="Webots world file (default: worlds/factory.wbt)."),
    ] = None,
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--no-headless", help="Run Webots headless (no-rendering is UNVERIFIED)."
        ),
    ] = False,
    frames: Annotated[
        bool,
        typer.Option("--frames/--no-frames", help="Record RGB frames (needs Pillow)."),
    ] = False,
    frame_hz: Annotated[
        float,
        typer.Option("--frame-hz", min=0.0, help="Frame capture rate in sim Hz (0 disables)."),
    ] = 1.0,
    startup_timeout_s: Annotated[
        float,
        typer.Option(min=1.0, help="Seconds to wait for the Webots server."),
    ] = 180.0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume", help="Skip episodes whose outcome is already recorded."
        ),
    ] = False,
) -> None:
    """Run the frozen baseline evaluation campaign and write a report (ADR-0029/0033)."""
    from vlarobot.backends.fake import FakeBackend
    from vlarobot.eval.runner import (
        EvalRunConfig,
        build_episode_plan,
        canonical_pipeline,
        run_evaluation,
    )
    from vlarobot.eval.settings import load_eval_settings
    from vlarobot.execution.scripted_baseline import scripted_policy_factory

    try:
        canonical = canonical_pipeline(pipeline)
        if canonical == "vla_only":
            raise ValueError("Use eval-vla with a trained export; baseline CLI cannot run VLA.")
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    settings = load_eval_settings()
    try:
        plan = build_episode_plan(split, seed, n=n, settings=settings)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    out_dir = out or Path("runs") / "eval" / f"{canonical}-{split}-s{seed}"
    measurement = "measured" if backend == "webots" else "fake"
    config = EvalRunConfig(
        out_dir=out_dir,
        pipeline=canonical,
        split=split,
        base_seed=seed,
        backend=backend,
        measurement=measurement,
        state_source="oracle",
        record_frames=frames,
        frame_hz=frame_hz,
        resume=resume,
    )

    started = time.perf_counter()
    startup_s = 0.0
    if backend == "fake":
        report = run_evaluation(
            plan,
            FakeBackend(),
            config,
            policy_factory=scripted_policy_factory,
            settings=settings,
        )
    else:
        world_path = world or RUNNER_DEFAULT_WORLD
        # Fresh session per episode for reset independence (shared-session reset integrity
        # was not established; MEASURED evidence required). Each episode is run in its own
        # Webots process and recorded; the final report is assembled from all recordings.
        try:
            from vlarobot.eval.runner import _completed_recording

            total = len(plan)
            for i, entry in enumerate(plan, 1):
                if resume and _completed_recording(out_dir, entry, canonical) is not None:
                    typer.echo(f"eval episode {i}/{total} index={entry.index}: resumed")
                    continue
                typer.echo(
                    f"eval episode {i}/{total} index={entry.index} seed={entry.seed}: starting"
                )
                ep_prefix = f"eval-{canonical}-{split}-s{seed}-ep{entry.index:02d}"
                with launch_webots_session(
                    webots=webots,
                    world=world_path,
                    trace_dir=out_dir,
                    trace_prefix=ep_prefix,
                    headless=headless,
                    startup_timeout_s=startup_timeout_s,
                ) as session:
                    startup_s += session.startup_s
                    run_evaluation(
                        [entry],
                        session.client,
                        config,
                        policy_factory=scripted_policy_factory,
                        settings=settings,
                        startup_s=session.startup_s,
                    )
            # Assemble final report from all completed recordings (no new episodes run).
            report = run_evaluation(
                plan,
                FakeBackend(),
                EvalRunConfig(
                    out_dir=out_dir,
                    pipeline=canonical,
                    split=split,
                    base_seed=seed,
                    backend=backend,
                    measurement=measurement,
                    state_source="oracle",
                    record_frames=frames,
                    frame_hz=frame_hz,
                    resume=True,
                ),
                policy_factory=scripted_policy_factory,
                settings=settings,
                startup_s=startup_s,
            )
        except Exception as error:  # noqa: BLE001 - surface a single CLI failure
            typer.echo(f"eval failed: {type(error).__name__}: {error}", err=True)
            raise typer.Exit(code=1) from error

    wall_s = time.perf_counter() - started
    success = report.summary["success"]
    gate = report.summary["week2_gate"]
    typer.echo(
        f"eval pipeline={canonical} split={split} backend={backend} measurement={measurement} "
        f"episodes={report.summary['n_episodes']} seeds={report.summary['seeds']}"
    )
    typer.echo(
        f"  success={success['successes']}/{success['total']} rate={success['rate']:.3f} "
        f"wilson95=[{success['ci_low']:.3f}, {success['ci_high']:.3f}]"
    )
    typer.echo(
        f"  gate_threshold={gate['threshold']:.3f} "
        f"(rule: {gate['rule']}, baseline_rate={gate['baseline_rate']:.3f})"
    )
    typer.echo(f"  out={out_dir} episodes_csv={report.episodes_csv.name} wall_s={wall_s:.1f}")


@app.command("eval-worldmind")
def eval_worldmind(
    config: Annotated[
        Path,
        typer.Option(help="Owner-reviewed, ready WorldMind production config (VALIDATION only)."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="New versioned output directory; prior evidence is never overwritten."),
    ],
    backend: Annotated[
        Literal["fake", "webots"],
        typer.Option(help="Declared backend: 'fake' (inert) or 'webots' (existing sim server)."),
    ] = "webots",
    state_source: Annotated[
        str, typer.Option(help="Declared state source; only 'oracle' is permitted.")
    ] = "oracle",
    split: Annotated[
        str, typer.Option(help="Only 'validation' is permitted; TEST and TRAIN are sealed.")
    ] = "validation",
) -> None:
    """Run the production WorldMind composition with fail-closed VALIDATION preflight."""
    from vlarobot.eval.worldmind_preflight import run_worldmind_preflight
    from vlarobot.execution.worldmind_pipeline import run_worldmind_evaluation

    try:
        from vlarobot.backends.worldmind_bindings import (
            production_worldmind_bindings,
            runtime_preflight_probe,
        )

        bindings = production_worldmind_bindings()
    except Exception as error:  # noqa: BLE001 - registration must fail closed with one CLI error
        typer.echo(
            f"eval-worldmind preflight failed: binding registration: "
            f"{type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    preflight = run_worldmind_preflight(
        config,
        out,
        split=split,
        backend=backend,
        state_source=state_source,
        bindings=bindings,
        runtime_probe=runtime_preflight_probe,
    )
    if not preflight.ready or preflight.config is None:
        typer.echo("eval-worldmind preflight failed:", err=True)
        for blocker in preflight.blockers or ("config: no validated configuration",):
            typer.echo(f"  - {blocker}", err=True)
        raise typer.Exit(code=1)
    resolved, plan = preflight.config, preflight.plan

    if backend == "fake":
        from vlarobot.backends.fake import FakeBackend

        client: Any = FakeBackend()
    else:
        # Connect to an already running sim server; this command never launches Webots.
        client = ZmqBackendClient(DEFAULT_ENDPOINT)
    try:
        report, sidecars = run_worldmind_evaluation(
            plan,
            client,
            resolved,
            out,
            bindings,
        )
    except Exception as error:  # noqa: BLE001 - surface one CLI failure after evidence finalize
        typer.echo(f"eval-worldmind failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    typer.echo(
        f"eval-worldmind split={split} backend={backend} state_source={state_source} "
        f"episodes={report.summary['n_episodes']} seeds={report.summary['seeds']}"
    )
    typer.echo(f"  out={out} sidecars={sidecars['decisions'].parent}")


@app.command("worldmind-diagnostic")
def worldmind_diagnostic(
    config: Annotated[
        Path,
        typer.Option(help="Stage-1 uncalibrated VALIDATION diagnostic config."),
    ],
    out: Annotated[Path, typer.Option(help="New versioned output directory; never overwritten.")],
    backend: Annotated[Literal["fake", "webots"], typer.Option()] = "webots",
    state_source: Annotated[str, typer.Option()] = "oracle",
    split: Annotated[str, typer.Option()] = "validation",
) -> None:
    """Preflight an uncalibrated diagnostic config. This command never scores or approves."""
    from vlarobot.backends.worldmind_bindings import (
        production_worldmind_bindings,
        runtime_preflight_probe,
    )
    from vlarobot.eval.worldmind_preflight import run_worldmind_diagnostic_preflight

    result = run_worldmind_diagnostic_preflight(
        config,
        out,
        split=split,
        backend=backend,
        state_source=state_source,
        bindings=production_worldmind_bindings(),
        runtime_probe=runtime_preflight_probe,
    )
    for check in result.checks:
        typer.echo(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    if not result.ready:
        typer.echo("worldmind-diagnostic preflight failed:", err=True)
        for blocker in result.blockers:
            typer.echo(f"  - {blocker}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        "worldmind-diagnostic: uncalibrated; scored_evaluation_allowed=False; "
        "no parameters selected or approved"
    )


@app.command("worldmind-stage1-evidence")
def worldmind_stage1_evidence(
    config: Annotated[
        Path,
        typer.Option(help="Stage-1 uncalibrated VALIDATION diagnostic config."),
    ],
    out: Annotated[Path, typer.Option(help="New versioned output directory; never overwritten.")],
    arm_mode: Annotated[
        Literal["full_system_veto", "scripted_proposer"],
        typer.Option(help="Explicit owner-authorized diagnostic mode; there is no default."),
    ],
    execution_mode: Annotated[
        Literal[
            "receding_horizon_tick",
            "diagnostic_open_loop_chunk",
            "diagnostic_observed_state_feedback_chunk",
        ],
        typer.Option(
            help=(
                "Explicit diagnostic execution path; observed-state feedback requires "
                "scripted_proposer and emits no prediction pairing."
            )
        ),
    ] = "receding_horizon_tick",
    backend: Annotated[Literal["fake", "webots"], typer.Option()] = "webots",
    state_source: Annotated[str, typer.Option()] = "oracle",
    split: Annotated[str, typer.Option()] = "validation",
    resume: Annotated[
        bool, typer.Option(help="Resume an interrupted run in the same output directory.")
    ] = False,
) -> None:
    """Collect non-scored Stage 1 evidence; never score, approve or launch Webots."""
    from vlarobot.backends.worldmind_bindings import (
        production_worldmind_bindings,
        runtime_preflight_probe,
    )
    from vlarobot.eval.worldmind_preflight import run_worldmind_diagnostic_preflight
    from vlarobot.execution.worldmind_stage1_pipeline import (
        STAGE1_EVIDENCE_CLASS,
        STAGE1_EXECUTION_MODE_LABELS,
        run_stage1_evidence,
    )

    try:
        bindings = production_worldmind_bindings()
    except Exception as error:  # noqa: BLE001 - registration must fail closed with one CLI error
        typer.echo(
            f"worldmind-stage1-evidence preflight failed: binding registration: "
            f"{type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    try:
        preflight = run_worldmind_diagnostic_preflight(
            config,
            out,
            split=split,
            backend=backend,
            state_source=state_source,
            bindings=bindings,
            runtime_probe=runtime_preflight_probe,
            allow_existing=resume,
        )
    except Exception as error:  # noqa: BLE001 - surface one CLI error for an invalid command
        typer.echo(
            f"worldmind-stage1-evidence preflight failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error
    if not preflight.ready or preflight.config is None:
        typer.echo("worldmind-stage1-evidence preflight failed:", err=True)
        for blocker in preflight.blockers:
            typer.echo(f"  - {blocker}", err=True)
        raise typer.Exit(code=1)
    if preflight.config.diagnostic.scored_evaluation_allowed:
        typer.echo("worldmind-stage1-evidence refuses a scored config", err=True)
        raise typer.Exit(code=1)

    resolved = preflight.config
    runtime_values = runtime_preflight_probe(resolved, bindings)
    try:
        from vlarobot.backends.fake import FakeBackend

        client: Any = FakeBackend() if backend == "fake" else ZmqBackendClient(DEFAULT_ENDPOINT)
        report, sidecars = run_stage1_evidence(
            preflight.plan,
            client,
            resolved,
            out,
            bindings,
            arm_mode=arm_mode,
            execution_mode=execution_mode,
            runtime_values=runtime_values,
            resume=resume,
        )
    except Exception as error:  # noqa: BLE001 - surface one CLI failure after evidence finalize
        typer.echo(f"worldmind-stage1-evidence failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        close = getattr(locals().get("client"), "close", None)
        if close is not None:
            close()
    typer.echo(
        f"worldmind-stage1-evidence arm_mode={arm_mode} "
        f"evidence_class={STAGE1_EVIDENCE_CLASS[arm_mode]} "
        f"execution_mode={execution_mode} "
        f"execution_mode_label={STAGE1_EXECUTION_MODE_LABELS[execution_mode]} "
        f"scored_evaluation_allowed=False episodes={report.summary['n_episodes']} "
        f"resume={resume}"
    )
    typer.echo(f"  out={out} sidecars={sidecars['decisions'].parent}")


@app.command("worldmind-webots-diagnostic")
def worldmind_webots_diagnostic(
    config: Annotated[
        Path,
        typer.Option(help="Reviewed no-actuation Webots diagnostic config (VALIDATION only)."),
    ],
    out: Annotated[
        Path,
        typer.Option(
            help="New exclusive output directory; prior diagnostic evidence is never overwritten."
        ),
    ],
) -> None:
    """Inspect a real Webots backend read-only. Never applies, steps, resets or scores."""
    from vlarobot.backends.worldmind_webots_diagnostic import (
        collect_diagnostic_snapshot,
        dependency_availability,
    )
    from vlarobot.eval.worldmind_webots_diagnostic import (
        ensure_fresh_output,
        failed_webots_diagnostic,
        finalize_webots_diagnostic,
        load_webots_diagnostic_config,
        preflight_webots_diagnostic,
        write_diagnostic_evidence,
    )

    try:
        resolved = load_webots_diagnostic_config(config)
    except Exception as error:  # noqa: BLE001 - config failures must fail closed with one error
        typer.echo(
            f"worldmind-webots-diagnostic config failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    try:
        ensure_fresh_output(out)
        out.mkdir(parents=True, exist_ok=False)
    except Exception as error:  # noqa: BLE001 - output exclusivity is a hard gate
        typer.echo(
            f"worldmind-webots-diagnostic output failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    preflight = preflight_webots_diagnostic(resolved)
    if preflight.blockers:
        report = failed_webots_diagnostic(resolved, preflight, "config_preflight_failed")
        write_diagnostic_evidence(out, report)
        typer.echo("worldmind-webots-diagnostic preflight failed:", err=True)
        for blocker in preflight.blockers:
            typer.echo(f"  - {blocker}", err=True)
        raise typer.Exit(code=1)

    try:
        snapshot = collect_diagnostic_snapshot(resolved, out)
    except Exception as error:  # noqa: BLE001 - surface one failure and preserve a fail-closed report
        try:
            availability = dependency_availability(resolved)
        except Exception:  # noqa: BLE001 - unavailable dependencies are recorded as empty
            availability = {}
        report = failed_webots_diagnostic(
            resolved, preflight, f"{type(error).__name__}: {error}", availability
        )
        write_diagnostic_evidence(out, report)
        typer.echo(f"worldmind-webots-diagnostic failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error

    report, state, provenance = finalize_webots_diagnostic(resolved, preflight, snapshot)
    files = write_diagnostic_evidence(out, report, state, provenance)
    outcome = str(report["outcome"])
    typer.echo(
        f"worldmind-webots-diagnostic outcome={outcome} mode=read_only_inspection "
        f"apply=0 step=0 reset=0 scored_evaluation_allowed=False"
    )
    blockers = report.get("blockers", [])
    if isinstance(blockers, (list, tuple)):
        for blocker in blockers:
            typer.echo(f"  - {blocker}", err=True)
    typer.echo(f"  out={out} evidence={','.join(path.name for path in files)}")
    if outcome != "PASS":
        raise typer.Exit(code=1)


@app.command("worldmind-webots-geometry-inspection")
def worldmind_webots_geometry_inspection(
    config: Annotated[
        Path,
        typer.Option(help="Reviewed read-only Webots geometry config (VALIDATION only)."),
    ],
    out: Annotated[
        Path,
        typer.Option(
            help="New exclusive output directory; prior geometry evidence is never overwritten."
        ),
    ],
) -> None:
    """Export Webots boundingObject geometry read-only. Never actuates, steps, IKs or scores."""
    from vlarobot.backends.worldmind_webots_geometry_inspection import (
        collect_geometry_inspection_snapshot,
    )
    from vlarobot.eval.worldmind_webots_diagnostic import ensure_fresh_output
    from vlarobot.eval.worldmind_webots_geometry_inspection import (
        failed_geometry_inspection,
        finalize_geometry_inspection,
        load_geometry_inspection_config,
        preflight_geometry_inspection,
        write_geometry_inspection_evidence,
    )

    try:
        resolved = load_geometry_inspection_config(config)
    except Exception as error:  # noqa: BLE001 - config failures must fail closed with one error
        typer.echo(
            f"worldmind-webots-geometry-inspection config failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    try:
        ensure_fresh_output(out)
        out.mkdir(parents=True, exist_ok=False)
    except Exception as error:  # noqa: BLE001 - output exclusivity is a hard gate
        typer.echo(
            f"worldmind-webots-geometry-inspection output failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    preflight = preflight_geometry_inspection(resolved)
    if preflight.blockers:
        report = failed_geometry_inspection(resolved, preflight, "config_preflight_failed")
        write_geometry_inspection_evidence(out, report, None)
        typer.echo("worldmind-webots-geometry-inspection preflight failed:", err=True)
        for blocker in preflight.blockers:
            typer.echo(f"  - {blocker}", err=True)
        raise typer.Exit(code=1)

    try:
        snapshot = collect_geometry_inspection_snapshot(resolved, out)
    except Exception as error:  # noqa: BLE001 - preserve one fail-closed report
        report = failed_geometry_inspection(resolved, preflight, f"{type(error).__name__}: {error}")
        write_geometry_inspection_evidence(out, report, None)
        typer.echo(
            f"worldmind-webots-geometry-inspection failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    report, state, provenance, geometry = finalize_geometry_inspection(
        resolved, preflight, snapshot
    )
    files = write_geometry_inspection_evidence(out, report, geometry, state, provenance)
    typer.echo(
        f"worldmind-webots-geometry-inspection outcome={report['outcome']} "
        f"mode=live_read_only_bounding_object_export apply=0 step=0 reset=0 "
        f"scored_evaluation_allowed=False motion_allowed=False"
    )
    blockers = report.get("blockers", [])
    if isinstance(blockers, (list, tuple)):
        for blocker in blockers:
            typer.echo(f"  - {blocker}", err=True)
    typer.echo(f"  out={out} evidence={','.join(path.name for path in files)}")
    if report["outcome"] != "PASS":
        raise typer.Exit(code=1)


@app.command("worldmind-geometry-diagnostic")
def worldmind_geometry_diagnostic(
    config: Annotated[
        Path,
        typer.Option(help="Reviewed local-only geometry diagnostic config (VALIDATION only)."),
    ],
    out: Annotated[
        Path,
        typer.Option(
            help="New exclusive output directory; prior diagnostic evidence is never overwritten."
        ),
    ],
) -> None:
    """Inspect pinned geometry files locally. Never launch Webots, actuate, solve IK or score."""
    from vlarobot.backends.worldmind_geometry_diagnostic import collect_geometry_evidence
    from vlarobot.eval.worldmind_geometry_diagnostic import (
        ensure_fresh_output,
        failed_geometry_diagnostic,
        finalize_geometry_diagnostic,
        load_geometry_diagnostic_config,
        preflight_geometry_diagnostic,
        write_geometry_evidence,
    )

    try:
        resolved = load_geometry_diagnostic_config(config)
    except Exception as error:  # noqa: BLE001 - config failures must fail closed with one error
        typer.echo(
            f"worldmind-geometry-diagnostic config failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    try:
        ensure_fresh_output(out)
        out.mkdir(parents=True, exist_ok=False)
    except Exception as error:  # noqa: BLE001 - output exclusivity is a hard gate
        typer.echo(
            f"worldmind-geometry-diagnostic output failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    preflight = preflight_geometry_diagnostic(resolved)
    if preflight.blockers:
        report = failed_geometry_diagnostic(resolved, preflight, "config_preflight_failed")
        write_geometry_evidence(out, report)
        typer.echo("worldmind-geometry-diagnostic preflight failed:", err=True)
        for blocker in preflight.blockers:
            typer.echo(f"  - {blocker}", err=True)
        raise typer.Exit(code=1)

    try:
        evidence = collect_geometry_evidence(resolved)
    except Exception as error:  # noqa: BLE001 - preserve a fail-closed report on collection failure
        report = failed_geometry_diagnostic(resolved, preflight, f"{type(error).__name__}: {error}")
        write_geometry_evidence(out, report)
        typer.echo(
            f"worldmind-geometry-diagnostic failed: {type(error).__name__}: {error}", err=True
        )
        raise typer.Exit(code=1) from error

    report, state, provenance = finalize_geometry_diagnostic(resolved, preflight, evidence)
    files = write_geometry_evidence(out, report, state, provenance)
    typer.echo(
        f"worldmind-geometry-diagnostic outcome={report['outcome']} mode=local_file_inspection "
        f"webots_launch=0 apply=0 step=0 reset=0 scored_evaluation_allowed=False "
        f"motion_allowed=False"
    )
    gaps = report.get("coverage_gaps")
    if isinstance(gaps, (list, tuple)):
        for gap in gaps:
            typer.echo(f"  gap: {gap}", err=True)
    blockers = report.get("blockers", [])
    if isinstance(blockers, (list, tuple)):
        for blocker in blockers:
            typer.echo(f"  - {blocker}", err=True)
    typer.echo(f"  out={out} evidence={','.join(path.name for path in files)}")
    if report["outcome"] != "PASS":
        raise typer.Exit(code=1)


@app.command("worldmind-motion-readiness-dossier")
def worldmind_motion_readiness_dossier(
    output: Annotated[
        Path,
        typer.Option(help="New JSON dossier path; the command never approves motion."),
    ],
) -> None:
    """Write the inert PASS/FAIL/UNVERIFIED dossier. Never launch Webots, move or score."""
    from vlarobot.eval.worldmind_motion_readiness_dossier import build_motion_readiness_dossier

    try:
        report = build_motion_readiness_dossier()
        resolved = output.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    except Exception as error:  # noqa: BLE001 - dossier generation must fail closed with one error
        typer.echo(
            f"worldmind-motion-readiness-dossier failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error
    typer.echo(
        f"worldmind-motion-readiness-dossier overall={report['overall_gate']} "
        f"motion_readiness={report['motion_readiness']} approved=False "
        f"webots_launch=0 apply=0 step=0 scored_evaluation_allowed=False"
    )
    for gate in report["gates"]:
        typer.echo(f"  {gate['status']}: {gate['id']} - {gate['summary']}")
    typer.echo(f"  out={resolved}")
    raise typer.Exit(code=1)


@app.command("worldmind-calibration-proposal")
def worldmind_calibration_proposal(
    draft: Annotated[Path, typer.Option(help="Unapproved, reviewable calibration draft JSON.")],
    evaluator: Annotated[str, typer.Option(help="Registered evaluator binding identity.")],
) -> None:
    """Print a deterministic VALIDATION selection statistic; never write or approve parameters."""
    from vlarobot.eval.worldmind_calibration import (
        load_calibration_draft,
        rank_calibration_candidates,
    )

    try:
        from vlarobot.backends.worldmind_bindings import production_worldmind_bindings

        registered = production_worldmind_bindings().calibration_evaluators.get(evaluator)
    except Exception as error:  # noqa: BLE001 - registration must fail closed with one error
        typer.echo(f"worldmind-calibration-proposal failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not callable(registered):
        typer.echo(
            f"worldmind-calibration-proposal failed: no registered evaluator: {evaluator}",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        proposal = rank_calibration_candidates(load_calibration_draft(draft), registered)
    except Exception as error:  # noqa: BLE001 - surface one CLI failure
        typer.echo(
            f"worldmind-calibration-proposal failed: {type(error).__name__}: {error}", err=True
        )
        raise typer.Exit(code=1) from error
    typer.echo(f"calibration_id={proposal.calibration_id} evidence_id={proposal.evidence_id}")
    typer.echo(f"evaluation_class={proposal.evaluation_class}")
    typer.echo(
        f"recommended_candidate_id={proposal.recommended_candidate_id} "
        f"scores={json.dumps(dict(sorted(proposal.scores.items())), sort_keys=True)}"
    )
    typer.echo("not an approval: owner must write an approved candidate artifact before scoring")


@app.command("derive-demo-actions")
def derive_demo_actions(
    steps: Annotated[Path, typer.Argument(help="Path to episode-XX.steps.npz")],
    out: Annotated[
        Path | None,
        typer.Option(help="Output JSONL path (default: <steps>.derived-actions.jsonl)."),
    ] = None,
) -> None:
    """Derive per-transition EE-delta + gripper actions from recorded steps (dataset prep)."""
    from vlarobot.eval.demos import write_derived_actions_jsonl

    if not steps.is_file():
        typer.echo(f"steps file not found: {steps}", err=True)
        raise typer.Exit(code=1)
    out_path = out or steps.with_suffix("").with_suffix(".derived-actions.jsonl")
    written = write_derived_actions_jsonl(steps, out_path)
    # count rows
    n_rows = sum(1 for _ in written.read_text(encoding="utf-8").splitlines() if _.strip())
    typer.echo(f"derived_actions={n_rows} out={written}")


@app.command("reset-repro-check")
def reset_repro_check(
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory (default: runs/eval/reset-repro-<split>-s<seed>)."),
    ] = None,
    split: Annotated[
        Literal["train", "validation"],
        typer.Option(help="Scene split (test fail-closed by default)."),
    ] = "train",
    seed: Annotated[int, typer.Option(help="Seed selecting target/other scenarios.")] = 0,
    webots: Annotated[
        str | None,
        typer.Option("--webots", help="Webots executable (defaults to WEBOTS_HOME, then PATH)."),
    ] = None,
    world: Annotated[
        Path | None,
        typer.Option(help="Webots world file (default: worlds/factory.wbt)."),
    ] = None,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="Run Webots headless."),
    ] = False,
    startup_timeout_s: Annotated[
        float,
        typer.Option(min=1.0, help="Seconds to wait for the Webots server."),
    ] = 180.0,
) -> None:
    """MEASURED (Webots): prove reset independence (fresh vs reused session) (ADR-0029)."""
    from vlarobot.eval.reset_repro import run_reset_repro

    out_dir = out or Path("runs") / "eval" / f"reset-repro-{split}-s{seed}"
    try:
        report = run_reset_repro(
            webots=webots,
            world=world,
            out_dir=out_dir,
            split=split,
            seed=seed,
            headless=headless,
            startup_timeout_s=startup_timeout_s,
        )
    except Exception as error:  # noqa: BLE001
        typer.echo(f"reset-repro-check failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    comp = report["comparison"]
    typer.echo(
        f"reset-repro ok={comp['ok']} max_abs_diff={comp['max_abs_diff']:.6g} "
        f"target={report['target_scenario']} other={report['other_scenario']}"
    )
    if comp["notes"]:
        typer.echo(f"  notes={comp['notes']}")
    typer.echo(f"  out={out_dir / 'reset_repro.json'}")


@app.command("collect-demos")
def collect_demos(
    n: Annotated[int, typer.Option("--n", min=1, help="Number of demo episodes to collect.")] = 80,
    seed: Annotated[
        int, typer.Option(help="Base seed (iterated by seed blocks for coverage).")
    ] = 100,
    split: Annotated[
        Literal["train", "validation"],
        typer.Option(help="Scene split. The test split is fail-closed until week 4 (ADR-0029)."),
    ] = "train",
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory (default: runs/demos/<split>-s<seed>-n<n>)."),
    ] = None,
    instruction: Annotated[
        str,
        typer.Option(help="Task instruction recorded per episode."),
    ] = "Pick up the can and place it at the target location.",
    backend: Annotated[
        Literal["fake", "webots"],
        typer.Option(help="Episode backend: 'fake' (plumbing test) or 'webots' (measured demos)."),
    ] = "webots",
    webots: Annotated[
        str | None,
        typer.Option("--webots", help="Webots executable (defaults to WEBOTS_HOME, then PATH)."),
    ] = None,
    world: Annotated[
        Path | None,
        typer.Option(help="Webots world file (default: worlds/factory.wbt)."),
    ] = None,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="Run Webots headless."),
    ] = False,
    frames: Annotated[
        bool,
        typer.Option("--frames/--no-frames", help="Record RGB frames (needed for vision demos)."),
    ] = True,
    frame_hz: Annotated[
        float,
        typer.Option("--frame-hz", min=0.0, help="Frame capture rate in sim Hz."),
    ] = 1000.0 / TIME_STEP,
    baseline: Annotated[
        Path,
        typer.Option(help="Completed repaired baseline used to authorize physical collection."),
    ] = Path("runs/eval/baseline-train-s0-fresh-v2-registered"),
    registration_file: Annotated[
        Path, typer.Option(help="Frozen static registration, relative to configs or absolute.")
    ] = Path("calibration/urdf-base-train-v1.yaml"),
    max_attempts: Annotated[
        int, typer.Option(min=0, help="Attempt budget; 0 uses configs/vla.yaml.")
    ] = 0,
    startup_timeout_s: Annotated[
        float,
        typer.Option(min=1.0, help="Seconds to wait for the Webots server."),
    ] = 180.0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume", help="Skip episodes already marked completed in the manifest."
        ),
    ] = False,
) -> None:
    """Collect scripted-expert demos in a LeRobot-convertible layout (ADR-0029/M8 prep)."""
    from vlarobot.backends.fake import FakeBackend
    from vlarobot.eval.demos import build_demo_plan
    from vlarobot.eval.demos import collect_demos as collect_demo_episodes
    from vlarobot.eval.settings import load_eval_settings
    from vlarobot.execution.scripted_baseline import scripted_policy_factory

    if backend == "webots" and (split != "train" or headless or not frames):
        raise typer.BadParameter("Successful demonstrations require TRAIN, rendering and RGB.")
    settings = load_eval_settings()
    try:
        plan = build_demo_plan(
            n, base_seed=seed, split=split, instruction=instruction, settings=settings
        )
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    out_dir = out or Path("runs") / "demos" / f"{split}-s{seed}-n{n}"
    state_source = "oracle"
    measurement = "measured" if backend == "webots" else "fake"
    started = time.perf_counter()

    if backend == "fake":
        result = collect_demo_episodes(
            plan,
            FakeBackend(),
            out_dir,
            policy_factory=scripted_policy_factory,
            state_source=state_source,
            pipeline="scripted",
            record_frames=frames,
            frame_hz=frame_hz,
            resume=resume,
        )
    else:
        from vlarobot.execution.demo_collection import collect_verified_training

        if abs(frame_hz - 1000.0 / TIME_STEP) > 1e-9:
            raise typer.BadParameter("Measured demo admission requires full lockstep RGB cadence.")
        previous_registration = os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE")
        os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = str(registration_file)
        try:
            state = collect_verified_training(
                out_dir,
                baseline,
                webots=resolve_webots(webots),
                world=world or RUNNER_DEFAULT_WORLD,
                target=n,
                seed=seed,
                max_attempts=max_attempts or None,
                startup_timeout_s=startup_timeout_s,
                resume=resume,
                instruction=instruction,
            )
        except Exception as error:  # noqa: BLE001
            typer.echo(f"collect-demos failed: {type(error).__name__}: {error}", err=True)
            raise typer.Exit(code=1) from error
        finally:
            if previous_registration is None:
                os.environ.pop("VLAROBOT_URDF_REGISTRATION_FILE", None)
            else:
                os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = previous_registration
        typer.echo(
            f"collect-demos status={state['status']} accepted={state['successful_episodes']}/{n} "
            f"attempted={state['attempted']} quarantined={state['quarantined_attempts']} "
            f"out={out_dir}"
        )
        return

    wall_s = time.perf_counter() - started
    typer.echo(
        f"collect-demos split={split} backend={backend} measurement={measurement} "
        f"episodes={result.episodes} successes={result.successes} failures={result.failures}"
    )
    typer.echo(f"  out={out_dir} manifest={result.manifest_path.name} wall_s={wall_s:.1f}")


@app.command("run-episodes")
def run_episodes(
    n: int = typer.Option(1, "--n", min=1, help="Number of episodes to run."),
    fresh_world: bool = typer.Option(
        True,
        "--fresh-world/--no-fresh-world",
        help="Launch a fresh Webots subprocess per episode so every run starts from a clean world.",
    ),
    headless: bool = typer.Option(
        False,
        "--headless/--no-headless",
        help=(
            "Use the batch flags (--batch --mode=fast --no-rendering --stdout --stderr). "
            "--no-rendering is UNVERIFIED and can blank camera frames; keep rendering for now."
        ),
    ),
    determinism_check: int = typer.Option(
        0,
        "--determinism-check",
        min=0,
        help=(
            "Repeat one episode K times in fresh worlds and report whether outcomes and "
            "traces match. Overrides --n."
        ),
    ),
    world: Annotated[
        Path | None,
        typer.Option(help="Path to the Webots world file; defaults to worlds/factory.wbt."),
    ] = None,
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    seed: int = typer.Option(0, help="PRNG seed for the first episode; later episodes increment."),
    webots: str | None = typer.Option(
        None, "--webots", help="Webots executable; defaults to WEBOTS_HOME, then PATH."
    ),
    trace_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for per-episode traces, summaries, and Webots logs."),
    ] = None,
    pregrasp_route: Annotated[
        Literal["interpolated", "safe_via"] | None,
        typer.Option(help="PREGRASP route; defaults to configs/robot.yaml."),
    ] = None,
    pregrasp_realign: Annotated[
        bool | None,
        typer.Option(
            "--pregrasp-realign/--no-pregrasp-realign",
            help="Re-align PREGRASP after base tilt; defaults to configs/robot.yaml.",
        ),
    ] = None,
    closed_loop_waypoints: Annotated[
        bool | None,
        typer.Option(
            "--closed-loop-waypoints/--no-closed-loop-waypoints",
            help=(
                "Recompute DESCEND/LIFT targets per waypoint from the live base pose; "
                "defaults to configs/robot.yaml."
            ),
        ),
    ] = None,
    kinematic_model: Annotated[
        Literal["urdf", "sim"] | None,
        typer.Option(
            "--kinematic-model",
            help=(
                "Kinematic model for this run (recorded in every episode summary). "
                "sim requires a valid snapshot verifiable against the running session."
            ),
        ),
    ] = None,
    stop_after: Annotated[
        Literal["PREGRASP", "LIFT"] | None,
        typer.Option(help="Stop each episode after PREGRASP or LIFT."),
    ] = None,
    grasp_servo: Annotated[
        Literal["none", "gt_pad"],
        typer.Option(
            "--grasp-servo",
            help=(
                "Grasp servo mode (ADR-0023): 'none' keeps legacy open-loop motion; "
                "'gt_pad' servos PREGRASP XY, DESCEND stop height, and CLOSE alignment "
                "on the measured arm_7 pad reference."
            ),
        ),
    ] = "none",
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            help=(
                "Named scene set from vlarobot.eval.scenarios (e.g. 'mini-grid10'); "
                "episode i runs scene i, cycling when --n exceeds the set (ADR-0025)."
            ),
        ),
    ] = None,
    startup_timeout_s: float = typer.Option(
        180.0, min=1.0, help="Seconds to wait for the Webots ZeroMQ server before failing."
    ),
) -> None:
    """Run Webots episodes with clean process isolation and summarize outcomes."""
    scene_set: tuple[ScenarioSpec, ...] = ()
    if scenario is not None:
        try:
            scene_set = get_scenarios(scenario)
        except KeyError:
            raise typer.BadParameter(
                f"unknown scenario {scenario!r}; available: {', '.join(sorted(SCENARIOS))}",
                param_hint="--scenario",
            ) from None
    if headless:
        typer.echo(
            "Note: headless mode disables Webots rendering; the scripted policy reads RGB every "
            "step, so camera frames may be blank and episodes may fail for that reason only."
        )
    resolved_webots = resolve_webots(webots)
    settings = RunnerSettings(
        webots=resolved_webots,
        world=(world or RUN_EPISODES_DEFAULT_WORLD).expanduser().resolve(),
        endpoint=endpoint,
        trace_dir=(
            trace_dir.expanduser().resolve()
            if trace_dir is not None
            else Path("runs") / "episode-runner"
        ),
        trace_prefix="episode",
        headless=headless,
        startup_timeout_s=startup_timeout_s,
        seed=seed,
        pregrasp_realign=pregrasp_realign,
        closed_loop_waypoints=closed_loop_waypoints,
        stop_after=stop_after,
        kinematic_model=kinematic_model,
        grasp_servo=grasp_servo,
        scenario=scene_set,
    )
    outcomes = run_episode_batch(
        settings,
        n=n,
        fresh_world=fresh_world,
        determinism_check=determinism_check,
    )
    # Always persist the full outcomes (incl. failure reasons) for crash forensics.
    summary_path = settings.trace_dir / f"{settings.trace_prefix}-summary.json"
    try:
        summary_path.write_text(
            json.dumps([row.to_dict() for row in outcomes], indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    typer.echo(format_summary_table(outcomes))
    if determinism_check > 0:
        typer.echo(format_determinism_report(determinism_report(outcomes)))
        if not determinism_report(outcomes).get("match"):
            raise typer.Exit(code=1)


@app.command("repeat-check")
def repeat_check(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    episodes: int = typer.Option(2, min=2, help="Consecutive episodes in one server session."),
    seed: int = typer.Option(0, help="Same seed for every episode."),
    stop_after: Annotated[
        Literal["PREGRASP", "LIFT"] | None,
        typer.Option(help="Stop each episode after PREGRASP or LIFT."),
    ] = "PREGRASP",
    pregrasp_route: Annotated[
        Literal["interpolated", "safe_via"] | None,
        typer.Option(help="PREGRASP route; defaults to configs/robot.yaml."),
    ] = None,
    pregrasp_realign: Annotated[
        bool | None,
        typer.Option(
            "--pregrasp-realign/--no-pregrasp-realign",
            help="Re-align PREGRASP after base tilt; defaults to configs/robot.yaml.",
        ),
    ] = None,
    trace_log: Annotated[
        Path | None, typer.Option(help="Append per-state trace records to this JSONL.")
    ] = None,
    measured_tolerance: float = typer.Option(
        1e-3,
        min=0.0,
        help="Allowed change in measured sensors/poses between episodes (m, rad, or matrix units).",
    ),
    timings: bool = typer.Option(
        False,
        "--timings",
        help="Print opt-in wall-time counters for each episode.",
    ),
) -> None:
    """Repeat the same seeded episode in one server session and diff reset fingerprints."""
    client = ZmqBackendClient(endpoint)
    policy = ScriptedCanPolicy(
        trace_path=trace_log,
        pregrasp_route=pregrasp_route,
        pregrasp_realign=pregrasp_realign,
        stop_after=stop_after,
        timing_enabled=timings,
    )
    results = []
    try:
        client.ping()
        for episode in range(episodes):
            result = policy.execute(client, seed)
            results.append(result)
            typer.echo(
                f"episode={episode} seed={seed} final_state={result.final_state} "
                f"success={result.success} pregrasp_realign={result.pregrasp_realign} "
                f"max_base_drift_m={_optional_float(result.max_base_drift_m, 6)} "
                f"max_base_drift_yaw_deg={_optional_float(result.max_base_drift_yaw_deg, 3)} "
                f"max_base_tilt_deg={_optional_float(result.max_base_tilt_deg, 3)} "
                f"torso_start_rad={_optional_float(result.torso_start_rad, 6)} "
                f"torso_end_rad={_optional_float(result.torso_end_rad, 6)} "
                f"fingerprint={'present' if result.reset_fingerprint is not None else 'missing'}"
            )
            if timings:
                _print_timing_summary(policy, client)

        first = results[0]
        comparisons = []
        for index, result in enumerate(results[1:], start=1):
            comparison = _fingerprint_comparison(
                first.reset_fingerprint,
                result.reset_fingerprint,
                measured_tolerance,
            )
            comparisons.append(comparison)
            typer.echo(
                f"fingerprint_diff episode=0->{index} "
                f"commanded={len(comparison.commanded)} "
                f"measured={len(comparison.measured)} "
                f"transient={len(comparison.transient)} "
                f"max_measured_delta={_optional_float(comparison.max_measured_delta, 9)} "
                f"max_transient_delta={_optional_float(comparison.max_transient_delta, 9)}"
            )
            for difference in comparison.commanded:
                typer.echo(f"  commanded {difference}")
            for difference in comparison.measured:
                typer.echo(f"  measured {difference}")
            for difference in comparison.transient:
                typer.echo(f"  transient {difference}")

        passed = all(comparison.passed for comparison in comparisons)
        typer.echo(f"repeat-check result={'PASS' if passed else 'FAIL'} episodes={episodes}")
        if not passed:
            raise typer.Exit(code=1)
    finally:
        client.close()


@app.command("can-reset-check")
def can_reset_check(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    episodes: int = typer.Option(3, min=3, help="Consecutive reset episodes to compare."),
    seed: int = typer.Option(0, help="Repeat the same seeded policy on every reset."),
    warmup_steps: int = typer.Option(5, min=0, help="Camera warm-up steps before PREGRASP."),
    trace_log: Annotated[
        Path | None, typer.Option(help="Append reset probe records to this JSONL.")
    ] = None,
) -> None:
    """Compare RGB and the INIT/PREGRASP command sequence over consecutive resets."""
    client = ZmqBackendClient(endpoint)
    policy = ScriptedCanPolicy(trace_path=trace_log)
    probes: list[ResetProbeResult] = []
    try:
        client.ping()
        for episode in range(episodes):
            probe = policy.probe_initial_pregrasp(
                client,
                seed,
                warmup_steps,
                replay=probes[0] if probes else None,
            )
            probes.append(probe)
            typer.echo(
                f"episode={episode} seed={probe.seed} reset_s={probe.reset_duration_s:.6f} "
                f"probe_s={probe.probe_duration_s:.6f} states={probe.state_sequence} "
                f"pregrasp_base_m={probe.commanded_pregrasp} "
                f"achieved_base_m={probe.achieved_position} "
                f"gripper_m={probe.gripper_opening:.6f} "
                f"waypoints={len(probe.waypoints)} failure={probe.failure_reason}"
            )

        passed = True
        for index, (first, second) in enumerate(zip(probes, probes[1:], strict=False), start=1):
            behavior = _compare_reset_probes(first, second)
            image_comparisons = [
                (
                    "reset",
                    *_image_similarity(first.rgb_at_reset, second.rgb_at_reset),
                )
            ]
            for step, image_a in first.rgb_warmup_samples:
                image_b = dict(second.rgb_warmup_samples).get(step)
                if image_b is not None:
                    image_comparisons.append((f"step{step}", *_image_similarity(image_a, image_b)))
            image_output = " ".join(
                f"rgb_{label}_mad={mad:.6f} rgb_{label}_ssim={ssim:.9f} "
                f"rgb_{label}_max_delta={maximum}"
                for label, mad, ssim, maximum in image_comparisons
            )
            typer.echo(
                f"comparison={index - 1}->{index} behavior={behavior['result']} "
                f"states_equal={behavior['states_equal']} "
                f"reset_max_joint_delta_rad={behavior['reset_max_joint_delta_rad']:.9g} "
                f"reset_base_delta_m={behavior['reset_base_delta_m']:.9g} "
                f"reset_base_orientation_delta_rad="
                f"{behavior['reset_base_orientation_delta_rad']:.9g} "
                f"reset_ee_delta_m={behavior['reset_ee_delta_m']:.9g} "
                f"reset_ee_orientation_delta_rad="
                f"{behavior['reset_ee_orientation_delta_rad']:.9g} "
                f"reset_can_delta_m={behavior['reset_can_delta_m']:.9g} "
                f"reset_can_orientation_delta_rad="
                f"{behavior['reset_can_orientation_delta_rad']:.9g} "
                f"pregrasp_delta_m={behavior['pregrasp_delta_m']:.9g} "
                f"max_joint_command_delta_rad={behavior['max_joint_command_delta_rad']:.9g} "
                f"max_waypoint_pose_delta_m={behavior['max_waypoint_pose_delta_m']:.9g} "
                f"max_waypoint_orientation_delta_rad="
                f"{behavior['max_waypoint_orientation_delta_rad']:.9g} "
                f"max_waypoint_gripper_delta_m={behavior['max_waypoint_gripper_delta_m']:.9g} "
                f"achieved_position_delta_m={behavior['achieved_position_delta_m']:.9g} "
                f"achieved_orientation_delta_rad={behavior['achieved_orientation_delta_rad']:.9g} "
                f"gripper_delta_m={behavior['gripper_delta_m']:.9g} "
                f"{image_output}"
            )
            if behavior["result"] != "PASS":
                passed = False

        typer.echo(
            "reset-check "
            f"behavior={'PASS' if passed else 'FAIL'} "
            f"pose_joint_can_tolerance=1e-3 "
            f"policy_seed={seed} command_sequence='episode 0 plan replayed in later episodes' "
            f"rgb_warmup_steps={warmup_steps} "
            f"reset_observation_timing='after one Webots step, episode t=0'"
        )
        if not passed:
            raise typer.Exit(code=1)
    finally:
        client.close()


@app.command("export-sim-kinematics")
def export_sim_kinematics(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    output: Annotated[
        Path | None,
        typer.Option(help="Where to write the JSON snapshot; defaults to configs/robot.yaml."),
    ] = None,
    seed: int = typer.Option(0, help="Seed for the reset that sets the export posture."),
    poll_timeout_ms: int = typer.Option(
        120_000,
        min=1,
        help="Per-poll request timeout (ms); each poll probes one joint chunk.",
    ),
) -> None:
    """Export live Supervisor kinematics via chunked polling and verify the chain (ADR-0020)."""
    client = ZmqBackendClient(endpoint)
    try:
        ping = client.ping()
        session = ping.get("payload", {}).get("session_id")
        if session:
            typer.echo(f"session_id={session}")
        # Fast reset to the settled export posture (no probes).
        client.reset(ScenarioSpec(), seed=seed)
        typer.echo("export: start (chunked polling, one joint per poll)")
        status = client.export_kinematics("start", timeout_ms=poll_timeout_ms)
        while not status.get("done"):
            status = client.export_kinematics("poll", timeout_ms=poll_timeout_ms)
            per_joint = status.get("per_joint") or []
            current = per_joint[-1] if per_joint else None
            if isinstance(current, dict):
                typer.echo(
                    f"  progress {status.get('current_index')}/{status.get('total_joints')} "
                    f"joint={current.get('name')} axis={current.get('axis_source')} "
                    f"steps={current.get('steps_used')} "
                    f"moved={current.get('measured_angle_moved_rad')} "
                    f"reason={current.get('reason')} "
                    f"total_steps={status.get('steps_used_total')}"
                )
            else:
                typer.echo(
                    f"  progress {status.get('current_index')}/{status.get('total_joints')} "
                    f"state={status.get('state')} total_steps={status.get('steps_used_total')}"
                )
        snapshot = status.get("snapshot")
        if not isinstance(snapshot, dict):
            typer.echo(f"error: export did not produce a snapshot: {status}")
            raise typer.Exit(code=1)
        path = (output or SIM_KINEMATICS_PATH).expanduser()
        model = finalize_verified_snapshot(
            snapshot,
            path,
            session_id=session if isinstance(session, dict) else None,
        )
        chain = build_chain(model)
        residuals = verify_export_posture(model, chain)
        # Read back for reporting (includes valid/hash/session written on success).
        saved = json.loads(path.read_text(encoding="utf-8"))
        typer.echo(
            f"export path={path} bytes={path.stat().st_size} schema={saved.get('schema_version')} "
            f"valid={saved.get('valid')} hash={str(saved.get('content_hash', ''))[:12]} "
            f"joints={len(model.joints)} torso_reference_m={model.torso_reference:.6f} "
            f"tool_source={model.tool_source}"
        )
        readability = model.metadata.get("readability_report")
        if isinstance(readability, dict):
            typer.echo(
                f"readability jointParameters={readability.get('jointParameters')} "
                f"posture_probe={readability.get('posture_probe')} "
                f"unreadable={readability.get('unreadable')}"
            )
        for joint in model.joints:
            cross_check = _optional_float(joint.axis_cross_check_deg, 3)
            residual = _optional_float(joint.pose_residual_m, 9)
            typer.echo(
                f"  joint={joint.name} source={joint.axis_source} "
                f"axis={np.array2string(joint.axis, precision=6)} "
                f"limits={joint.limits} probe_residual_m={residual} "
                f"jointParameters_cross_check_deg={cross_check}"
            )
        progress = snapshot.get("export_progress") or {}
        if isinstance(progress, dict):
            for item in progress.get("per_joint") or []:
                if isinstance(item, dict):
                    typer.echo(
                        f"  probe_status joint={item.get('name')} steps={item.get('steps_used')} "
                        f"measured_angle_moved_rad={item.get('measured_angle_moved_rad')} "
                        f"settled={item.get('settled')} reason={item.get('reason')}"
                    )
        typer.echo(
            "export-posture FK residual max_m="
            f"{max(residuals.values(), default=0.0):.3e} "
            f"(tolerance {EXPORT_POSTURE_FIT_TOL_M:.0e} m)"
        )
        typer.echo(
            "next: to validate, set export_sim_kinematics: true in configs/robot.yaml, "
            "restart the Webots sim-server (controller process), then run "
            "uv run python -m vlarobot.cli validate-kinematics"
        )
    finally:
        client.close()


@dataclass
class _PoseOutcome:
    """One sweep pose attempt: the sample if scored, else the abort reason."""

    sample: JointSample | None
    reason: str | None


def _max_joint_error(observation: Observation, command: dict[str, float]) -> float:
    """Largest |commanded - measured| over the arm joints of one observation (rad)."""
    errors = [
        abs(command[name] - observation.robot.joint_pos[name])
        for name in ARM_JOINTS
        if name in observation.robot.joint_pos
    ]
    return max(errors, default=float("inf"))


def _stationary_settled(
    previous: Observation, current: Observation, *, delta_rad: float = SETTLE_DELTA_RAD
) -> bool:
    """Stationarity: per-joint change below threshold between consecutive observations."""
    for name in ARM_JOINTS:
        p = previous.robot.joint_pos.get(name)
        c = current.robot.joint_pos.get(name)
        if p is None or c is None:
            return False
        if abs(float(c) - float(p)) >= delta_rad:
            return False
    return True


@overload
def _drive_pose(
    client: ZmqBackendClient,
    target: dict[str, float],
    torso_target: float | None = None,
    *,
    max_steps: int,
    settle_tolerance: float,
    observation: None = None,
    max_delta_rad: float = 0.05,
) -> _PoseOutcome: ...


@overload
def _drive_pose(
    client: ZmqBackendClient,
    target: dict[str, float],
    torso_target: float | None = None,
    *,
    max_steps: int,
    settle_tolerance: float,
    observation: Observation,
    max_delta_rad: float = 0.05,
) -> tuple[_PoseOutcome, Observation]: ...


def _drive_pose(
    client: ZmqBackendClient,
    target: dict[str, float],
    torso_target: float | None = None,
    *,
    max_steps: int,
    settle_tolerance: float,
    observation: Observation | None = None,
    max_delta_rad: float = 0.05,
) -> _PoseOutcome | tuple[_PoseOutcome, Observation]:
    """Apply a joint target, settle, and pre-check the pose for table safety.

    Legacy: ``(client, target, torso_target, *, max_steps, settle_tolerance)`` returns a
    :class:`_PoseOutcome` and applies the full command once (including torso) for
    Webots-free unit tests. Live sweep: pass ``observation=`` and no ``torso_target`` to
    interpolate arm targets and return ``(_PoseOutcome, Observation)``.
    """
    # Legacy/single-shot path (tests): no current observation supplied.
    if observation is None:
        command = dict(target)
        if torso_target is not None:
            command[TORSO_JOINT] = float(torso_target)
        response = client.apply(JointTarget(command))
        if not response.accepted:
            return _PoseOutcome(None, f"safety rejected the target: {response.reason}")
        obs = client.step(1)
        goal = {name: command[name] for name in ARM_JOINTS if name in command}
        for _ in range(max_steps - 1):
            if _max_joint_error(obs, goal) < settle_tolerance:
                break
            obs = client.step(1)
        if _max_joint_error(obs, goal) >= settle_tolerance:
            return _PoseOutcome(None, f"pose did not settle within {max_steps} steps")
        telemetry = obs.robot.diagnostics.get("kinematics_telemetry")
        if not isinstance(telemetry, dict):
            return _PoseOutcome(None, "observation carries no kinematics_telemetry")
        safe, reason = sample_is_safe(telemetry)
        if not safe:
            return _PoseOutcome(None, f"aborted by the table pre-check: {reason}")
        joints = {
            name: float(obs.robot.joint_pos[name])
            for name in ARM_JOINTS
            if name in obs.robot.joint_pos
        }
        if len(joints) != len(ARM_JOINTS):
            return _PoseOutcome(None, "observation lacks measured arm joint angles")
        return _PoseOutcome(JointSample(joints=joints, telemetry=telemetry), None)

    current = {name: float(observation.robot.joint_pos[name]) for name in ARM_JOINTS}
    goal = {name: float(target[name]) for name in ARM_JOINTS}
    max_err = max(abs(goal[n] - current[n]) for n in ARM_JOINTS)
    segments = max(1, int(np.ceil(max_err / max(1e-6, max_delta_rad))))
    for seg in range(1, segments + 1):
        alpha = seg / segments
        command = {
            name: current[name] + alpha * (goal[name] - current[name]) for name in ARM_JOINTS
        }
        response = client.apply(JointTarget(command))
        if not response.accepted:
            return (
                _PoseOutcome(None, f"safety rejected the target: {response.reason}"),
                observation,
            )
        observation = client.step(1)
    for _ in range(max_steps):
        if _max_joint_error(observation, goal) < settle_tolerance:
            break
        observation = client.step(1)
    if _max_joint_error(observation, goal) >= settle_tolerance:
        return (
            _PoseOutcome(None, f"pose did not settle within {max_steps} steps"),
            observation,
        )
    telemetry = observation.robot.diagnostics.get("kinematics_telemetry")
    if not isinstance(telemetry, dict):
        return _PoseOutcome(None, "observation carries no kinematics_telemetry"), observation
    safe, reason = sample_is_safe(telemetry)
    if not safe:
        return _PoseOutcome(None, f"aborted by the table pre-check: {reason}"), observation
    joints = {
        name: float(observation.robot.joint_pos[name])
        for name in ARM_JOINTS
        if name in observation.robot.joint_pos
    }
    if len(joints) != len(ARM_JOINTS):
        return _PoseOutcome(None, "observation lacks measured arm joint angles"), observation
    return _PoseOutcome(JointSample(joints=joints, telemetry=telemetry), None), observation


def _drive_submove_to_target(
    client: ZmqBackendClient,
    observation: Observation,
    target: dict[str, float],
) -> tuple[bool, str | None, Observation]:
    """Apply one bounded sub-move and settle by stationarity (cap SETTLE_SUBMOVE_CAP)."""
    response = client.apply(JointTarget({name: float(target[name]) for name in ARM_JOINTS}))
    if not response.accepted:
        return False, f"safety rejected the sub-move: {response.reason}", observation
    previous = observation
    observation = client.step(1)
    stable = 1 if _stationary_settled(previous, observation) else 0
    for _ in range(1, SETTLE_SUBMOVE_CAP):
        if stable >= SETTLE_STEPS:
            return True, None, observation
        previous = observation
        observation = client.step(1)
        stable = stable + 1 if _stationary_settled(previous, observation) else 0
    if stable >= SETTLE_STEPS:
        return True, None, observation
    return (
        False,
        f"sub-move did not settle within {SETTLE_SUBMOVE_CAP} steps (stationarity)",
        observation,
    )


@app.command("validate-kinematics")
def validate_kinematics(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    n: int = typer.Option(
        12, min=1, help="Number of poses in the feasible sweep (ADR-0020 redesign: 12)."
    ),
    seed: int = typer.Option(0, help="Seed for pose sampling (deterministic sweep)."),
    report: Annotated[
        Path | None,
        typer.Option(help="Write the full report as JSON here (optional)."),
    ] = None,
    timeout_ms: int = typer.Option(
        300_000,
        min=1,
        help=(
            "Request timeout (ms); the initial reset may run axis probes when export is enabled."
        ),
    ),
    bench_ms_per_step: Annotated[
        float | None,
        typer.Option(
            "--bench-ms-per-step",
            help="Measured ms/step (e.g. from bench-steps) to print the total step budget.",
        ),
    ] = None,
    velocity_limit_rad_s: float = typer.Option(
        DEFAULT_VELOCITY_LIMIT_RAD_S,
        min=1e-6,
        help="Conservative joint velocity limit (rad/s) for sub-move bound 0.4*v*h.",
    ),
    command_horizon_s: float = typer.Option(
        DEFAULT_HORIZON_S,
        min=1e-6,
        help="Safety command horizon (s) for sub-move bound 0.4*v*h.",
    ),
) -> None:
    """Run the feasible sim-vs-URDF FK sweep against live GT Solids (ADR-0020 redesign)."""
    if n != 12:
        typer.echo(
            "warning: ADR-0020 redesigned acceptance sweep requires n=12 "
            "(6 spread + 6 near-task); results reported but not an acceptance run."
        )
    client = ZmqBackendClient(endpoint, timeout_ms=timeout_ms)
    samples: list[tuple[JointSample, str]] = []
    aborted: list[str] = []
    try:
        ping = client.ping()
        payload = ping.get("payload") if isinstance(ping, dict) else None
        telem_on = None
        if isinstance(payload, dict):
            telem_on = payload.get("kinematics_telemetry_enabled")
        if telem_on is False:
            typer.echo(
                "error: server reports kinematics_telemetry_enabled=false. Set "
                "export_sim_kinematics: true in configs/robot.yaml and restart the "
                "Webots sim-server/controller, then run this command again."
            )
            raise typer.Exit(code=1)
        observation = client.reset(ScenarioSpec(), seed=seed)
        if not isinstance(observation.robot.diagnostics.get("kinematics_telemetry"), dict):
            typer.echo(
                "error: observations carry no kinematics_telemetry. Set "
                "export_sim_kinematics: true in configs/robot.yaml (enables live GT telemetry), "
                "restart the Webots sim-server, and run this command again."
            )
            raise typer.Exit(code=1)
        torso_target = float(observation.robot.joint_pos.get(TORSO_JOINT, 0.0))
        snapshot_payload: dict[str, object] | None = None
        snapshot_path = SIM_KINEMATICS_PATH
        if snapshot_path.exists():
            try:
                loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    snapshot_payload = loaded
            except Exception:
                snapshot_payload = None
        limits = pose_limits(snapshot_payload)
        all_specs = feasible_pose_plan(seed=seed, limits=limits)
        specs = all_specs[:n]
        current = {name: float(observation.robot.joint_pos[name]) for name in ARM_JOINTS}
        bound = submove_bound(velocity_limit_rad_s, command_horizon_s)
        budget = estimate_step_budget(specs, current, bound, bench_ms_per_step)
        limit_text = ", ".join(
            f"{name}: ({low:.3f}, {high:.3f})" for name, (low, high) in limits.items()
        )
        typer.echo(
            f"sweep n={len(specs)} seed={seed} torso_rad={torso_target:.6f} limits={{{limit_text}}}"
        )
        typer.echo(
            "step budget: "
            f"poses={budget['poses']} submoves={budget['total_submoves']} "
            f"cap_per_submove={budget['steps_per_submove_cap']} "
            f"max_steps={budget['max_total_steps']}"
            + (
                f" bench_ms_per_step={budget['ms_per_step']:.1f} "
                f"est_s={budget.get('estimated_s', 0.0):.1f} "
                f"est_min={budget.get('estimated_min', 0.0):.2f}"
                if budget.get("ms_per_step") is not None
                else " (bench_ms_per_step not provided; ~447 ms/step recorded in docs)"
            )
        )
        typer.echo(
            f"submove_bound_rad={bound:.6f} "
            f"(0.4*velocity_limit_rad_s={velocity_limit_rad_s}*horizon_s={command_horizon_s})"
        )
        # Re-home to export posture before sweep if no samples yet; best-effort (current is current)
        for spec in specs:
            pose_label = f"pose={spec.index} group={spec.group}"
            pose_failed = False
            submoves = plan_submoves(current, spec.target, bound)
            for sub_index, sub_target in enumerate(submoves):
                ok, reason, observation = _drive_submove_to_target(client, observation, sub_target)
                if not ok:
                    aborted.append(f"{pose_label} submove={sub_index} {reason}")
                    typer.echo(f"  {pose_label} aborted (fail-fast): {reason}")
                    pose_failed = True
                    break
                telemetry = observation.robot.diagnostics.get("kinematics_telemetry")
                if isinstance(telemetry, dict):
                    safe, why = sample_is_safe(telemetry)
                    if not safe:
                        aborted.append(f"{pose_label} submove={sub_index} unsafe: {why}")
                        typer.echo(f"  {pose_label} aborted (fail-fast): table pre-check: {why}")
                        pose_failed = True
                        break
            if pose_failed:
                # try to continue from current measured pose
                current = {name: float(observation.robot.joint_pos[name]) for name in ARM_JOINTS}
                continue
            telemetry = observation.robot.diagnostics.get("kinematics_telemetry")
            if not isinstance(telemetry, dict):
                aborted.append(f"{pose_label} no kinematics_telemetry")
                typer.echo(f"  {pose_label} aborted: observation carries no kinematics_telemetry")
                current = {name: float(observation.robot.joint_pos[name]) for name in ARM_JOINTS}
                continue
            safe, why = sample_is_safe(telemetry)
            if not safe:
                aborted.append(f"{pose_label} unsafe: {why}")
                typer.echo(f"  {pose_label} aborted: table pre-check: {why}")
                current = {name: float(observation.robot.joint_pos[name]) for name in ARM_JOINTS}
                continue
            joints = {name: float(observation.robot.joint_pos[name]) for name in ARM_JOINTS}
            samples.append((JointSample(joints=joints, telemetry=telemetry), spec.group))
            current = dict(joints)
            typer.echo(f"  {pose_label} collected samples={len(samples)} submoves={len(submoves)}")
        if not samples:
            typer.echo("error: no usable sweep poses; nothing to score.")
            raise typer.Exit(code=1)
        sweep = evaluate_sweep(samples, tool_verified=False)
        for line in format_report(sweep):
            typer.echo(line)
        typer.echo(
            f"sweep samples={len(samples)} aborted={len(aborted)} "
            f"accept_mean_mm={LINK_MEAN_ACCEPT_M * 1e3:.1f} "
            f"accept_max_mm={LINK_MAX_ACCEPT_M * 1e3:.1f} "
            f"tool_frame=FALLBACK/UNVERIFIED"
        )
        if report is not None:
            path = report.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": {
                    "seed": seed,
                    "requested_poses": len(specs),
                    "scored_poses": len(samples),
                    "aborted": aborted,
                    "torso_rad": torso_target,
                    "limits": limits,
                    "velocity_limit_rad_s": velocity_limit_rad_s,
                    "command_horizon_s": command_horizon_s,
                    "submove_bound_rad": bound,
                    "step_budget": budget,
                    "tool_frame": "FALLBACK/UNVERIFIED",
                },
                "report": asdict(sweep),
                "samples": [
                    {"group": group, "joints": sample.joints, "telemetry": sample.telemetry}
                    for sample, group in samples
                ],
            }
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            typer.echo(f"report written path={path}")
        if sweep.accepted_model is None:
            raise typer.Exit(code=1)
    finally:
        client.close()


@app.command("compare-traces")
def compare_trace_files(
    trace_a: Path,
    trace_b: Path,
    prefix: str = typer.Option(
        "INIT,PREGRASP,DESCEND,CLOSE,LIFT", help="Comma-separated required state prefix."
    ),
    tol_pos: float = typer.Option(
        1e-3, min=0.0, help="Maximum commanded/achieved boundary and Cartesian EE delta in meters."
    ),
    max_tilt_deg: float = typer.Option(15.0, min=0.0, help="Maximum permitted tool tilt."),
) -> None:
    """Compare original-controller and brain waypoint traces."""
    result = compare_traces(
        trace_a,
        trace_b,
        tuple(state.strip() for state in prefix.split(",") if state.strip()),
        tol_pos,
        max_tilt_deg,
    )
    compared_states = tuple(state.strip() for state in prefix.split(",") if state.strip())
    for label, path in (("A", trace_a), ("B", trace_b)):
        diagnostics = joint_diagnostics(path, compared_states)
        max_step = max(diagnostics.max_step_by_joint_rad.values(), default=0.0)
        wrist = (
            "none"
            if not diagnostics.wrist_roll_excursions
            else "; ".join(diagnostics.wrist_roll_excursions)
        )
        margin = (
            "unavailable"
            if diagnostics.min_limit_margin_rad is None
            else f"{diagnostics.min_limit_margin_rad:.6g} rad"
        )
        typer.echo(
            f"trace {label} joint diagnostics: max_step={max_step:.6g} rad, "
            f"min_limit_margin={margin}, arm_7_step_excursions_over_1_rad={wrist}, "
            f"per_joint_max={diagnostics.max_step_by_joint_rad}"
        )
    if result.passed:
        typer.echo(f"PASS: {result.reason}; prefix={prefix}")
        return
    typer.echo(
        f"FAIL: first divergence state={result.state} waypoint={result.waypoint}: {result.reason}"
    )
    raise typer.Exit(code=1)


def _print_timing_summary(
    policy: ScriptedCanPolicy, client: ZmqBackendClient | None = None
) -> None:
    summary = policy.timing_summary()
    if not summary.get("enabled"):
        return
    totals = summary["totals"]
    total_wall = float(summary["total_wall_s"])
    if total_wall > 0.0:
        shares = " ".join(
            f"{component}={float(seconds):.6f}s ({100.0 * float(seconds) / total_wall:.1f}%)"
            for component, seconds in sorted(totals.items())
        )
        steps = int(summary.get("steps", 0) or 0)
        per_step = summary.get("wall_per_step_s")
        per_step_text = "n/a" if per_step is None else f"{float(per_step):.6f}s"
        typer.echo(
            f"  timings total_wall={total_wall:.6f}s steps={steps} "
            f"accounted_wall_per_step={per_step_text} {shares}"
        )
    for state, components in sorted(summary["by_state"].items()):
        state_total = sum(float(value) for value in components.values())
        detail = " ".join(
            f"{component}={float(seconds):.6f}s"
            for component, seconds in sorted(components.items())
        )
        typer.echo(f"  timings {state}: total={state_total:.6f}s {detail}")
    if client is not None:
        transport = client.timings()
        if transport:
            counted = {key: value for key, value in sorted(transport.items()) if key != "requests"}
            detail = " ".join(f"{key}={float(value):.6f}" for key, value in counted.items())
            typer.echo(f"  timings zmq: requests={int(transport.get('requests', 0.0))} {detail}")


def _optional_float(value: float | None, digits: int) -> str:
    return "None" if value is None else f"{value:.{digits}f}"


@dataclass(frozen=True)
class _FingerprintComparison:
    """Classified differences between two reset fingerprints."""

    commanded: list[str]
    measured: list[str]
    transient: list[str]
    max_measured_delta: float | None
    max_transient_delta: float | None

    @property
    def passed(self) -> bool:
        return not self.commanded and not self.measured


_TRANSIENT_FINGERPRINT_PREFIXES = (
    "base.velocity_world_xyz_m_s",
    "base.angular_velocity_world_xyz_rad_s",
)


def _fingerprint_leaves(value: object, path: str = "") -> dict[str, object]:
    leaves: dict[str, object] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{path}.{key}" if path else str(key)
            leaves.update(_fingerprint_leaves(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.update(_fingerprint_leaves(item, f"{path}[{index}]"))
    else:
        leaves[path] = value
    return leaves


def _numeric_delta(first: object, second: object) -> float | None:
    if (
        isinstance(first, (int, float))
        and isinstance(second, (int, float))
        and not isinstance(first, bool)
        and not isinstance(second, bool)
    ):
        return abs(float(first) - float(second))
    return None


def _is_transient_fingerprint_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
        for prefix in _TRANSIENT_FINGERPRINT_PREFIXES
    )


def _is_commanded_fingerprint_path(path: str) -> bool:
    return path.startswith("motors.")


def _fingerprint_comparison(
    first: object,
    second: object,
    measured_tolerance: float,
) -> _FingerprintComparison:
    """Classify fingerprint differences into commanded, measured, and transient state.

    Motor targets/velocities/control modes are the deterministic reset contract and must match
    exactly. Joint sensors and poses are physical measurements, so they may differ by up to
    ``measured_tolerance``. Base and CAN velocities captured at reset are world-load transients;
    they are reported but never fail the check.
    """
    if not isinstance(first, dict) or not isinstance(second, dict):
        return _FingerprintComparison(
            commanded=["fingerprint: missing in one run"],
            measured=[],
            transient=[],
            max_measured_delta=None,
            max_transient_delta=None,
        )
    first_leaves = _fingerprint_leaves(first)
    second_leaves = _fingerprint_leaves(second)
    commanded: list[str] = []
    measured: list[str] = []
    transient: list[str] = []
    max_measured_delta = 0.0
    max_transient_delta = 0.0
    for path in sorted(set(first_leaves) | set(second_leaves)):
        if path not in first_leaves or path not in second_leaves:
            description = f"{path}: missing in one run"
            if _is_transient_fingerprint_path(path):
                transient.append(description)
            else:
                commanded.append(description)
            continue
        first_value = first_leaves[path]
        second_value = second_leaves[path]
        delta = _numeric_delta(first_value, second_value)
        if delta is None:
            if first_value == second_value:
                continue
            description = f"{path}: {first_value!r} != {second_value!r}"
            if _is_transient_fingerprint_path(path):
                transient.append(description)
            else:
                commanded.append(description)
            continue
        if delta == 0.0:
            continue
        description = f"{path}: {first_value!r} != {second_value!r} (delta {delta:.6g})"
        if _is_transient_fingerprint_path(path):
            transient.append(description)
            max_transient_delta = max(max_transient_delta, delta)
        elif _is_commanded_fingerprint_path(path):
            commanded.append(description)
        else:
            max_measured_delta = max(max_measured_delta, delta)
            if delta > measured_tolerance:
                measured.append(description)
    return _FingerprintComparison(
        commanded=commanded,
        measured=measured,
        transient=transient,
        max_measured_delta=max_measured_delta if max_measured_delta > 0.0 else None,
        max_transient_delta=max_transient_delta if max_transient_delta > 0.0 else None,
    )


def _compare_reset_probes(
    first: ResetProbeResult, second: ResetProbeResult
) -> dict[str, float | str | bool]:
    states_equal = first.state_sequence == second.state_sequence
    first_reset_joints = dict(first.reset_joint_positions)
    second_reset_joints = dict(second.reset_joint_positions)
    if first_reset_joints.keys() != second_reset_joints.keys():
        reset_joint_delta = float("inf")
    else:
        reset_joint_delta = max(
            (
                abs(first_reset_joints[name] - second_reset_joints[name])
                for name in first_reset_joints
            ),
            default=0.0,
        )
    reset_base_delta = _vector_distance(first.reset_base_position, second.reset_base_position)
    reset_base_orientation_delta = _quaternion_distance(
        first.reset_base_orientation, second.reset_base_orientation
    )
    reset_ee_delta = _vector_distance(first.reset_ee_position, second.reset_ee_position)
    reset_ee_orientation_delta = _quaternion_distance(
        first.reset_ee_orientation, second.reset_ee_orientation
    )
    reset_can_delta = _vector_distance(first.reset_can_position, second.reset_can_position)
    reset_can_orientation_delta = _quaternion_distance(
        first.reset_can_orientation, second.reset_can_orientation
    )
    pregrasp_delta = _vector_distance(first.commanded_pregrasp, second.commanded_pregrasp)
    max_joint_delta = 0.0
    max_waypoint_pose_delta = 0.0
    max_waypoint_orientation_delta = 0.0
    max_waypoint_gripper_delta = 0.0
    if len(first.waypoints) != len(second.waypoints):
        max_joint_delta = float("inf")
        max_waypoint_pose_delta = float("inf")
    else:
        for first_waypoint, second_waypoint in zip(first.waypoints, second.waypoints, strict=True):
            first_targets = dict(first_waypoint.joint_targets)
            second_targets = dict(second_waypoint.joint_targets)
            if first_targets.keys() != second_targets.keys():
                max_joint_delta = float("inf")
            elif first_targets:
                max_joint_delta = max(
                    max_joint_delta,
                    max(abs(first_targets[name] - second_targets[name]) for name in first_targets),
                )
            max_waypoint_pose_delta = max(
                max_waypoint_pose_delta,
                _vector_distance(
                    first_waypoint.achieved_position, second_waypoint.achieved_position
                ),
            )
            max_waypoint_orientation_delta = max(
                max_waypoint_orientation_delta,
                _quaternion_distance(
                    first_waypoint.achieved_orientation,
                    second_waypoint.achieved_orientation,
                ),
            )
            max_waypoint_gripper_delta = max(
                max_waypoint_gripper_delta,
                abs(first_waypoint.gripper_opening - second_waypoint.gripper_opening),
            )
    achieved_position_delta = _vector_distance(first.achieved_position, second.achieved_position)
    achieved_orientation_delta = _quaternion_distance(
        first.achieved_orientation, second.achieved_orientation
    )
    gripper_delta = abs(first.gripper_opening - second.gripper_opening)
    passed = (
        states_equal
        and reset_joint_delta <= 1e-3
        and reset_base_delta <= 1e-3
        and reset_base_orientation_delta <= 1e-3
        and reset_ee_delta <= 1e-3
        and reset_ee_orientation_delta <= 1e-3
        and reset_can_delta <= 1e-3
        and reset_can_orientation_delta <= 1e-3
        and pregrasp_delta <= 1e-3
        and max_joint_delta <= 1e-3
        and max_waypoint_pose_delta <= 1e-3
        and max_waypoint_orientation_delta <= 1e-3
        and max_waypoint_gripper_delta <= 1e-3
        and achieved_position_delta <= 1e-3
        and achieved_orientation_delta <= 1e-3
        and gripper_delta <= 1e-3
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "states_equal": states_equal,
        "reset_max_joint_delta_rad": reset_joint_delta,
        "reset_base_delta_m": reset_base_delta,
        "reset_base_orientation_delta_rad": reset_base_orientation_delta,
        "reset_ee_delta_m": reset_ee_delta,
        "reset_ee_orientation_delta_rad": reset_ee_orientation_delta,
        "reset_can_delta_m": reset_can_delta,
        "reset_can_orientation_delta_rad": reset_can_orientation_delta,
        "pregrasp_delta_m": pregrasp_delta,
        "max_joint_command_delta_rad": max_joint_delta,
        "max_waypoint_pose_delta_m": max_waypoint_pose_delta,
        "max_waypoint_orientation_delta_rad": max_waypoint_orientation_delta,
        "max_waypoint_gripper_delta_m": max_waypoint_gripper_delta,
        "achieved_position_delta_m": achieved_position_delta,
        "achieved_orientation_delta_rad": achieved_orientation_delta,
        "gripper_delta_m": gripper_delta,
    }


def _vector_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def _quaternion_distance(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    dot = abs(sum(a * b for a, b in zip(first, second, strict=True)))
    return 2.0 * float(np.arccos(min(1.0, dot)))


def _image_similarity(first: np.ndarray, second: np.ndarray) -> tuple[float, float, int]:
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("RGB reset images must have matching (height, width, channels) shapes.")
    first_float = first.astype(np.float64)
    second_float = second.astype(np.float64)
    absolute_difference = np.abs(first_float - second_float)
    mean_absolute_difference = float(np.mean(absolute_difference))
    maximum_difference = int(np.max(absolute_difference, initial=0.0))
    ssim_scores = []
    height, width, _ = first.shape
    for row in range(0, height, 8):
        for column in range(0, width, 8):
            a = first_float[row : row + 8, column : column + 8]
            b = second_float[row : row + 8, column : column + 8]
            mean_a = np.mean(a, axis=(0, 1))
            mean_b = np.mean(b, axis=(0, 1))
            centered_a = a - mean_a
            centered_b = b - mean_b
            variance_a = np.mean(centered_a**2, axis=(0, 1))
            variance_b = np.mean(centered_b**2, axis=(0, 1))
            covariance = np.mean(centered_a * centered_b, axis=(0, 1))
            numerator = (2.0 * mean_a * mean_b + 6.5025) * (2.0 * covariance + 58.5225)
            denominator = (mean_a**2 + mean_b**2 + 6.5025) * (variance_a + variance_b + 58.5225)
            ssim_scores.extend((numerator / denominator).tolist())
    return mean_absolute_difference, float(np.mean(ssim_scores)), maximum_difference


def _port_owner(port: int) -> list[str]:
    """Best-effort owner lines for a TCP port (cross-platform where possible)."""
    lines: list[str] = []
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"], text=True, encoding="utf-8", errors="replace"
            )
            needle = f":{port}"
            for line in out.splitlines():
                if needle in line and "LISTENING" in line.upper():
                    lines.append(line.strip())
        except Exception as exc:
            lines.append(f"netstat failed: {exc}")
    else:
        try:
            out = subprocess.check_output(
                ["ss", "-ltnp"], text=True, encoding="utf-8", errors="replace"
            )
            needle = f":{port} "
            for line in out.splitlines():
                if needle in line or line.rstrip().endswith(f":{port}"):
                    lines.append(line.strip())
        except Exception:
            try:
                out = subprocess.check_output(
                    ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                lines.extend([ln.strip() for ln in out.splitlines() if ln.strip()])
            except Exception as exc:
                lines.append(f"port-owner lookup failed: {exc}")
    return lines


def _webots_process_lines() -> list[str]:
    lines: list[str] = []
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["tasklist", "/V"], text=True, encoding="utf-8", errors="replace"
            )
            for line in out.splitlines():
                low = line.lower()
                if "webots" in low or "tiago_controller" in low:
                    lines.append(line.strip())
        else:
            out = subprocess.check_output(
                ["ps", "-eo", "pid,ppid,etime,cmd"], text=True, encoding="utf-8", errors="replace"
            )
            for line in out.splitlines()[1:]:
                low = line.lower()
                if "webots" in low or "tiago_controller" in low or "sim-server" in low:
                    lines.append(line.strip())
    except Exception as exc:
        lines.append(f"process listing failed: {exc}")
    return lines


@app.command()
def doctor(
    port: int = typer.Option(5555, help="Simulator ZMQ port to inspect."),
    kill: bool = typer.Option(
        False, "--kill", help="Terminate listed stale processes after printing."
    ),
) -> None:
    """Preflight: list stale Webots/controller processes and the configured port owner."""
    typer.echo(f"doctor port={port}")
    port_free = True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    except OSError:
        port_free = False
    typer.echo(f"port_free={port_free}")
    owners = _port_owner(port)
    if owners:
        typer.echo("port_owner:")
        for line in owners:
            typer.echo(f"  {line}")
    else:
        typer.echo("port_owner: none")
    procs = _webots_process_lines()
    if procs:
        typer.echo("processes:")
        for line in procs:
            typer.echo(f"  {line}")
        webots_lines = [ln for ln in procs if "webots" in ln.lower()]
        if webots_lines:
            modes = ["batch" if "--batch" in ln else "gui/unknown" for ln in webots_lines]
            typer.echo(f"webots_mode={','.join(modes)}")
        else:
            typer.echo("webots_mode=none")
    else:
        typer.echo("processes: none")
        typer.echo("webots_mode=none")
    if not kill:
        return
    if not procs:
        typer.echo("kill: nothing to terminate")
        return
    typer.echo("kill: terminating listed processes")
    for line in procs:
        parts = line.split()
        if not parts:
            continue
        pid_str = parts[0]
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            if os.name == "nt":
                subprocess.check_call(["taskkill", "/PID", str(pid), "/T", "/F"])
            else:
                os.kill(pid, 15)
        except Exception as exc:
            typer.echo(f"  failed pid={pid}: {exc}")
        else:
            typer.echo(f"  signaled pid={pid}")


def _recent_run_episode_stats(trace_dir: Path | None = None) -> tuple[int, float]:
    """(episode_count, mean_step_rows) from the most recent run-episodes traces, if any."""
    root = trace_dir or (Path(__file__).resolve().parents[2] / "runs")
    candidates = [root / "episode-runner", root]
    files: list[Path] = []
    for base in candidates:
        if base.is_dir():
            files.extend(sorted(base.glob("episode-*.jsonl")))
    if not files:
        files = sorted(root.glob("**/episode-*.jsonl"))
    if not files:
        return 0, 0.0
    newest = max(files, key=lambda p: p.stat().st_mtime)
    batch = sorted(newest.parent.glob("episode-*.jsonl")) or [newest]
    counts: list[int] = []
    for path in batch:
        try:
            with path.open("r", encoding="utf-8") as handle:
                counts.append(sum(1 for line in handle if line.strip()))
        except OSError:
            continue
    if not counts:
        return 0, 0.0
    return len(counts), float(sum(counts)) / len(counts)


@app.command("bench-steps")
def bench_steps(
    endpoint: str = typer.Option(DEFAULT_ENDPOINT, help="ZeroMQ simulator endpoint."),
    n: int = typer.Option(200, min=1, help="Number of lockstep steps to measure."),
    seed: int = typer.Option(0, help="Reset seed."),
) -> None:
    """Measure step latency against a running simulator (no Webots launched)."""
    client = ZmqBackendClient(endpoint)
    try:
        ping = client.ping()
        typer.echo(f"session_id={ping.get('payload', {}).get('session_id')}")
        client.reset(ScenarioSpec(), seed=seed)
        # Warmup
        for _ in range(min(5, n)):
            client.step(1)
        t0 = time.perf_counter()
        for _ in range(n):
            client.step(1)
        elapsed = time.perf_counter() - t0
        ms_per = (elapsed / n) * 1000.0 if n else 0.0
        typer.echo(f"bench steps={n} elapsed_s={elapsed:.4f} ms_per_step={ms_per:.3f}")
        episodes, mean_steps = _recent_run_episode_stats()
        if episodes > 0 and mean_steps > 0.0:
            total_s = episodes * mean_steps * (ms_per / 1000.0)
            typer.echo(
                f"verdict: {episodes} episodes of ~{mean_steps:.0f} steps "
                f"would take ~{total_s:.1f}s (~{total_s / 60.0:.1f} min) at measured ms/step"
            )
        else:
            typer.echo(
                "verdict: no recent run-episodes trace found; cannot estimate episode batch time"
            )
        typer.echo(
            "note: each measured call is step+observation (state always; RGB controlled by "
            "server config observe_images; kinematics_telemetry attached when "
            "export_sim_kinematics is enabled). Compare runs with diagnostics/images toggled "
            "on the server to attribute cost. No server config is changed by this command."
        )
    finally:
        client.close()


@app.command("v4-skill-diagnostic")
def v4_skill_diagnostic(
    out: Path,
    webots: str | None = None,
    skill: str = "OPEN",
    *,
    empty_grasp: bool = False,
    imperfect_approach: bool = False,
    scene_index: int = 0,
    proxy: bool = False,
) -> None:
    """Check that one explicit V4 skill executes in isolation on TRAIN."""
    from vlarobot.execution.skill_diagnostic import run_skill_diagnostic
    from vlarobot.execution.skills import Skill

    result = run_skill_diagnostic(
        out,
        webots,
        Skill(skill),
        empty_grasp=empty_grasp,
        imperfect_approach=imperfect_approach,
        scene_index=scene_index,
        proxy=proxy,
    )
    typer.echo(json.dumps(result, indent=2))
    if not result["settled"] or not result["isolated"]:
        raise typer.Exit(1)


@app.command("v4-extract-demonstrations")
def v4_extract_demonstrations(out: Path, manifests: list[Path]) -> None:
    """Extract decision boundaries from accepted TRAIN demonstrations."""
    from vlarobot.execution.demo_collection import write_json
    from vlarobot.execution.skill_demonstrations import (
        demonstration_decisions,
        partition_demonstrations,
    )

    if out.exists():
        raise typer.BadParameter("Choose a fresh output file.")
    decisions = [row for manifest in manifests for row in demonstration_decisions(manifest)]
    if not decisions:
        raise typer.BadParameter("No decision boundaries found.")
    partition = partition_demonstrations(decisions)
    write_json(out, {"split": "train", "partition": partition, "decisions": decisions})
    episodes = {(row["scene"], row["seed"]) for row in decisions}
    typer.echo(f"{len(decisions)} decisions from {len(episodes)} TRAIN episodes.")


@app.command("v4-train-reference")
def v4_train_reference(dataset: Path, output: Path) -> None:
    """Fit a demonstration-supervised reference on TRAIN scene groups."""
    from vlarobot.execution.skill_reference import train_reference

    result = train_reference(dataset, output)
    typer.echo(
        f"Fit {result['fit_n']} decisions. Internal TRAIN accuracy "
        f"{result['internal_train_accuracy']:.3f} on {result['internal_train_evaluation_n']}."
    )


@app.command("v4-train-empirical")
def v4_train_empirical(dataset: Path, output: Path, measured: list[Path]) -> None:
    """Fit the initial utility baseline using observed TRAIN outcomes."""
    from vlarobot.execution.skill_empirical import train_empirical

    result = train_empirical(dataset, output, measured)
    typer.echo(
        f"Fit {result['demonstration_fit_n']} demonstrations and "
        f"{result['measured_fit_n']} measured outcomes."
    )


@app.command("v4-learned-smoke")
def v4_learned_smoke(
    out: Path,
    model: Path,
    webots: str | None = None,
    localizer: Path | None = None,
    context: str = "smoke",
) -> None:
    """Run the bounded oracle-assisted V4 clear TRAIN diagnostic."""
    from vlarobot.execution.skill_live import run_learned_smoke

    typer.echo(
        json.dumps(
            run_learned_smoke(out, model, webots, localizer_path=localizer, context=context),
            indent=2,
        )
    )


@app.command("v4-collect-approach-pairs")
def v4_collect_approach_pairs(
    out: Path, webots: str | None = None, seed: int = 1, *, internal: bool = False
) -> None:
    """Collect a bounded three-worker batch of TRAIN approach alternatives."""
    from vlarobot.execution.skill_collection import collect_approach_pairs

    typer.echo(json.dumps(collect_approach_pairs(out, webots, seed, internal=internal), indent=2))


@app.command("v4-fit-localizer")
def v4_fit_localizer(dataset: Path, sources: Path, cache: Path, output: Path) -> None:
    """Fit a CAN localizer on cached V3 frozen visual features."""
    from vlarobot.execution.skill_vision import fit_cached_localizer

    typer.echo(
        json.dumps(fit_cached_localizer(dataset, sources, cache, output)["internal_train_error"])
    )


@app.command("v4-collect-outcomes")
def v4_collect_outcomes(
    out: Path, webots: str | None = None, seed: int = 2, target: int = 150
) -> None:
    """Collect bounded TRAIN skill outcomes using three persistent sessions."""
    from vlarobot.execution.skill_collection import collect_outcomes

    typer.echo(json.dumps(collect_outcomes(out, webots, seed, target), indent=2))


@app.command("v4-update-outcome-models")
def v4_update_outcome_models(
    batch: Path,
    dataset: Path,
    out: Path,
    prior: list[Path],
    *,
    watch: bool = False,
) -> None:
    """Fit incremental empirical snapshots while the TRAIN batch runs."""
    from vlarobot.execution.skill_collection import update_outcome_models

    typer.echo(json.dumps(update_outcome_models(batch, dataset, out, prior, watch=watch), indent=2))


@app.command("v4-matched-clear")
def v4_matched_clear(out: Path, model: Path, webots: str | None = None) -> None:
    """Run all four matched prediction-context conditions on the clear TRAIN scene."""
    from vlarobot.execution.skill_live import run_matched_clear

    typer.echo(json.dumps(run_matched_clear(out, model, webots), indent=2))


if __name__ == "__main__":
    app()
