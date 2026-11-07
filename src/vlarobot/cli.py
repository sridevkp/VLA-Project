"""Command-line entry point for project tools."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from vlarobot.backends.webots.smoke import create_webots_robot, run_smoke
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import DEFAULT_ENDPOINT
from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.scripted_can import ResetProbeResult, ScriptedCanPolicy
from vlarobot.execution.trace_compare import compare_traces

app = typer.Typer(no_args_is_help=True)


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
) -> None:
    """Run a brain-side scripted CAN policy over repeated seeds and summarize the outcome."""
    client = ZmqBackendClient(endpoint)
    policy = ScriptedCanPolicy(trace_path=trace_log)
    results = []
    try:
        client.ping()
        for seed in range(seeds):
            result = policy.execute(client, seed)
            results.append(result)
            typer.echo(
                f"seed={seed} success={result.success} rejections={result.rejections} "
                f"final_distance={result.final_target_distance:.3f}m state={result.final_state} "
                f"failure_state={result.failure_state} failure_reason={result.failure_reason}"
            )
        successes = sum(1 for result in results if result.success)
        rejections = sum(result.rejections for result in results)
        typer.echo(f"summary success={successes}/{seeds} rejections={rejections}")
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


@app.command("compare-traces")
def compare_trace_files(
    trace_a: Path,
    trace_b: Path,
    prefix: str = typer.Option(
        "INIT,PREGRASP,DESCEND,CLOSE,LIFT", help="Comma-separated required state prefix."
    ),
    tol_joint: float = typer.Option(1e-3, min=0.0, help="Maximum joint target delta in radians."),
    tol_pos: float = typer.Option(1e-3, min=0.0, help="Maximum commanded EE delta in meters."),
) -> None:
    """Compare original-controller and brain waypoint traces."""
    result = compare_traces(
        trace_a,
        trace_b,
        tuple(state.strip() for state in prefix.split(",") if state.strip()),
        tol_joint,
        tol_pos,
    )
    if result.passed:
        typer.echo(f"PASS: traces match prefix={prefix}")
        return
    typer.echo(
        f"FAIL: first divergence state={result.state} waypoint={result.waypoint}: {result.reason}"
    )
    raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()
