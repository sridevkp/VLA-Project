"""Command-line entry point for project tools."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer

from vlarobot.backends.webots.smoke import create_webots_robot, run_smoke
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import DEFAULT_ENDPOINT
from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.scripted_can import ResetProbeResult, ScriptedCanPolicy
from vlarobot.execution.trace_compare import compare_traces, joint_diagnostics

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
                _print_timing_summary(policy)
        successes = sum(1 for result in results if result.success)
        rejections = sum(result.rejections for result in results)
        typer.echo(f"summary success={successes}/{seeds} rejections={rejections}")
    finally:
        client.close()


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
                _print_timing_summary(policy)

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


def _print_timing_summary(policy: ScriptedCanPolicy) -> None:
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
        typer.echo(f"  timings total_wall={total_wall:.6f}s {shares}")
    for state, components in sorted(summary["by_state"].items()):
        state_total = sum(float(value) for value in components.values())
        detail = " ".join(
            f"{component}={float(seconds):.6f}s"
            for component, seconds in sorted(components.items())
        )
        typer.echo(f"  timings {state}: total={state_total:.6f}s {detail}")


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


if __name__ == "__main__":
    app()
