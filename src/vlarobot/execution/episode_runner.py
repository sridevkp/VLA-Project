"""Cross-platform fresh-world Webots episode runner.

Launches ``webots <world>`` as a subprocess per episode (arg list, never ``shell=True``), waits
for the ZeroMQ lockstep server, runs exactly one brain-side scripted episode, terminates the
simulator, and collects one outcome row per episode. This module imports no simulator package;
it only uses the brain-side client, the scripted policy, and ``subprocess``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import IO

from vlarobot.backends.webots.config import SETTINGS, SIM_KINEMATICS_PATH
from vlarobot.backends.webots.sim_kinematics import (
    load_snapshot,
    resolve_snapshot_path,
)
from vlarobot.backends.zmq.client import REQUEST_TIMEOUTS_MS, ZmqBackendClient
from vlarobot.backends.zmq.server import endpoint_for_port
from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.scripted_can import ScriptedCanPolicy, ScriptedCanResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORLD = PROJECT_ROOT / "worlds" / "factory.wbt"
DEFAULT_TRACE_DIR = PROJECT_ROOT / "runs"
SHUTDOWN_TIMEOUT_S = 30.0

# EpisodeOutcome fields that legitimately differ between otherwise identical fresh-world runs:
# each episode gets its own index, trace/log paths, and wall-clock timings.
_OUTCOME_VOLATILE_KEYS = ("index", "trace_path", "log_path", "startup_s", "episode_s")

# Rows recorded by ``ScriptedCanPolicy`` that legitimately differ between otherwise identical
# fresh-world episodes: the reset fingerprint carries world-load transients, and
# ``controller_reset_state`` embeds the episode's own trace path and wall-clock durations.
_TRACE_VOLATILE_KEYS = (
    "reset_fingerprint",
    "controller_reset_state",
    "trace_path",
    "backend_diagnostics_compute_s",
)


@dataclass(frozen=True)
class RunnerSettings:
    """Pure brain-side configuration for the episode runner (no simulator imports)."""

    webots: str
    world: Path
    endpoint: str
    trace_dir: Path
    trace_prefix: str = "episode"
    headless: bool = False
    startup_timeout_s: float = 180.0
    shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S
    seed: int = 0
    pregrasp_realign: bool | None = None
    closed_loop_waypoints: bool | None = None
    stop_after: str | None = None
    kinematic_model: str | None = None
    grasp_servo: str | None = None
    # Named scene set (ADR-0025): episode i runs scene i (cycles when n exceeds the set).
    # Resolved by the CLI from vlarobot.eval.scenarios so this module stays eval-agnostic.
    scenario: tuple[ScenarioSpec, ...] = ()


@dataclass(frozen=True)
class EpisodeOutcome:
    """One row of the per-episode summary table."""

    index: int
    seed: int
    success: bool
    final_state: str
    failure_state: str | None
    failure_reason: str | None
    final_target_distance_m: float
    home_reached: bool
    place_distance_ok: bool
    can_height_ok: bool
    can_tilt_ok: bool
    can_height_error_m: float | None
    can_tilt_deg: float | None
    rejections: int
    startup_s: float
    episode_s: float
    trace_path: Path
    log_path: Path
    pregrasp_realign: bool | None = None
    closed_loop_waypoints: bool | None = None
    kinematic_model: str | None = None
    snapshot_hash: str | None = None
    snapshot_path: str | None = None
    grasp_servo: str | None = None
    scenario: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe version of this outcome row."""
        payload = asdict(self)
        payload["trace_path"] = str(self.trace_path)
        payload["log_path"] = str(self.log_path)
        return payload


def scenario_for_index(settings: RunnerSettings, index: int) -> ScenarioSpec:
    """Scene for episode ``index``: the configured set cycled by index, else the default scene.

    The default scene keeps the world-file pose, so batches without ``--scenario`` behave
    exactly as before ADR-0025.
    """
    if not settings.scenario:
        return ScenarioSpec()
    return settings.scenario[index % len(settings.scenario)]


def build_webots_command(webots: str, world: Path, *, headless: bool = False) -> list[str]:
    """Webots command line as a subprocess arg list (Webots R2025a, per docs/VERSIONS.md).

    ``headless=True`` maps to the documented batch flags
    (``webots --batch --mode=fast --no-rendering --stdout --stderr <world>``); these are not yet
    live-verified on any OS so they stay opt-in. The default keeps ``--batch --mode=fast`` with
    log redirection but leaves rendering on, because the scripted policy reads RGB every step.
    """
    flags = [webots, "--batch", "--mode=fast"]
    if headless:
        flags.append("--no-rendering")
    flags += ["--stdout", "--stderr", str(world)]
    return flags


def webots_environment(webots: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the Webots subprocess: inherit, and set ``WEBOTS_HOME`` if missing."""
    env = dict(os.environ if base is None else base)
    if not env.get("WEBOTS_HOME"):
        home = Path(webots).expanduser().resolve().parent
        env["WEBOTS_HOME"] = str(home)
    return env


def resolve_webots(webots: str | None = None) -> str:
    """Locate the Webots executable: explicit argument, ``WEBOTS_HOME``, then ``PATH``."""
    if webots:
        found = shutil.which(webots)
        if found is not None:
            return found
        candidate = Path(webots).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Webots executable {webots!r} was not found.")
    home = os.environ.get("WEBOTS_HOME")
    if home:
        executable = Path(home) / ("webots.exe" if os.name == "nt" else "webots")
        if executable.is_file():
            return str(executable)
    found = shutil.which("webots")
    if found is not None:
        return found
    raise FileNotFoundError(
        "Webots was not found. Install it, or pass --webots or set WEBOTS_HOME."
    )


def format_summary_table(outcomes: list[EpisodeOutcome]) -> str:
    """Fixed-width table: outcome, each success criterion, failure state, and timing."""
    header = (
        f"{'epi':>3s} {'seed':>4s} {'outcome':>7s} {'final':>6s} {'fail_state':>10s} "
        f"{'dist_m':>7s} {'home':>5s} {'place':>5s} {'height':>6s} {'tilt':>5s} "
        f"{'startup_s':>9s} {'episode_s':>9s} trace"
    )
    lines = [header, "-" * len(header)]
    for row in outcomes:
        lines.append(
            f"{row.index:>3d} {row.seed:>4d} {('ok' if row.success else 'no'):>7s} "
            f"{row.final_state:>6s} {(row.failure_state or '-'):>10s} "
            f"{row.final_target_distance_m:>7.3f} {str(row.home_reached):>5s} "
            f"{str(row.place_distance_ok):>5s} {str(row.can_height_ok):>6s} "
            f"{str(row.can_tilt_ok):>5s} {row.startup_s:>9.2f} {row.episode_s:>9.2f} "
            f"{row.trace_path.name}"
        )
    successes = sum(1 for row in outcomes if row.success)
    rows = len(outcomes)
    states = sorted({row.failure_state for row in outcomes if row.failure_state})
    lines.append("-" * len(header))
    lines.append(f"summary success={successes}/{rows} failure_states={states}")
    return "\n".join(lines)


def normalized_trace_rows(path: Path) -> list[dict[str, object]]:
    """Trace rows with wall-clock and session-specific keys removed, for determinism diffs."""
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in _TRACE_VOLATILE_KEYS:
            row.pop(key, None)
        rows.append(row)
    return rows


def trace_digest(path: Path) -> str | None:
    """SHA-256 over the normalized trace rows, or ``None`` when the file is missing."""
    if not path.is_file():
        return None
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in normalized_trace_rows(path))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trace_available(path: Path) -> bool:
    """True when a trace file exists and can be read for comparison."""
    try:
        return path.is_file()
    except OSError:
        return False


def trace_numeric_comparison(first: Path, second: Path) -> dict[str, object]:
    """Max row-count and pose/time deltas between two same-seed traces, in event order.

    If either trace file is missing/unreadable, comparison is unavailable and reported as such
    instead of raising.
    """
    if not _trace_available(first) or not _trace_available(second):
        return {
            "trace_comparison_available": False,
            "missing_trace": str(second if _trace_available(first) else first),
        }
    rows_first = [
        row for row in normalized_trace_rows(first) if row.get("event") == "state_boundary"
    ]
    rows_second = [
        row for row in normalized_trace_rows(second) if row.get("event") == "state_boundary"
    ]
    report: dict[str, object] = {
        "trace_comparison_available": True,
        "boundary_rows_equal": len(rows_first) == len(rows_second),
        "boundary_rows": min(len(rows_first), len(rows_second)),
    }
    limits = {
        "sim_time_max_delta_s": 0.0,
        "achieved_ee_max_delta_m": 0.0,
        "can_world_max_delta_m": 0.0,
        "commanded_ee_max_delta_m": 0.0,
    }
    for row_first, row_second in zip(rows_first, rows_second, strict=False):
        for key, limit_key in (
            ("sim_time_s", "sim_time_max_delta_s"),
            ("achieved_ee_base_xyz_m", "achieved_ee_max_delta_m"),
            ("can_world_xyz_m", "can_world_max_delta_m"),
            ("commanded_ee_base_xyz_m", "commanded_ee_max_delta_m"),
        ):
            value_first = row_first.get(key)
            value_second = row_second.get(key)
            if not isinstance(value_first, (list, tuple)) or not isinstance(
                value_second, (list, tuple)
            ):
                if isinstance(value_first, (int, float)) and isinstance(value_second, (int, float)):
                    limits[limit_key] = max(limits[limit_key], abs(value_first - value_second))
                continue
            if len(value_first) != len(value_second):
                delta = float("inf")
            else:
                delta = max(
                    abs(float(a) - float(b))
                    for a, b in zip(value_first, value_second, strict=False)
                )
            limits[limit_key] = max(limits[limit_key], delta)
    report.update(limits)
    return report


def normalized_outcome(row: EpisodeOutcome) -> dict[str, object]:
    """Outcome dictionary without the per-episode index, paths, and wall-clock timings."""
    payload = row.to_dict()
    for key in _OUTCOME_VOLATILE_KEYS:
        payload.pop(key, None)
    return payload


def determinism_report(outcomes: list[EpisodeOutcome]) -> dict[str, object]:
    """Outcomes, trace digests, and numeric deltas for repeated same-seed fresh-world episodes.

    Episodes without a produced trace are reported as unavailable and compared only against the
    first episode that produced a trace (when any). Missing traces make the determinism verdict
    non-matching so callers fail clearly instead of crashing.
    """
    digests = [trace_digest(row.trace_path) for row in outcomes]
    available = [
        (index, row, digest)
        for index, (row, digest) in enumerate(zip(outcomes, digests, strict=True))
        if _trace_available(row.trace_path) and digest is not None
    ]
    baseline_idx = available[0][0] if available else 0
    baseline_row = outcomes[baseline_idx] if outcomes else None
    baseline_outcome = normalized_outcome(baseline_row) if baseline_row is not None else None
    outcomes_match = (
        all(normalized_outcome(row) == baseline_outcome for row in outcomes[1:])
        if outcomes
        else False
    )
    available_digests = [digest for _, _, digest in available]
    digests_match = (
        len(available_digests) == len(outcomes)
        and len(outcomes) > 0
        and len(set(available_digests)) == 1
    )
    missing_traces = [
        {
            "index": index,
            "trace_path": str(row.trace_path),
            "log_path": str(row.log_path),
            "failure_state": row.failure_state,
            "failure_reason": row.failure_reason,
        }
        for index, row in enumerate(outcomes)
        if not _trace_available(row.trace_path)
    ]
    comparisons: list[dict[str, object]] = []
    for index, row in enumerate(outcomes):
        if index == baseline_idx:
            continue
        pair = f"{baseline_idx}->{index}"
        if not _trace_available(row.trace_path) or not _trace_available(
            outcomes[baseline_idx].trace_path
        ):
            comparisons.append(
                {
                    "pair": pair,
                    "outcome_match": normalized_outcome(row) == baseline_outcome,
                    "digest_match": False,
                    "trace_comparison_available": False,
                    "reason": row.failure_reason or "missing trace",
                    "log_path": str(row.log_path),
                }
            )
            continue
        comparisons.append(
            {
                "pair": pair,
                "outcome_match": normalized_outcome(row) == baseline_outcome,
                "digest_match": digests[baseline_idx] is not None
                and digests[index] == digests[baseline_idx],
                **trace_numeric_comparison(outcomes[baseline_idx].trace_path, row.trace_path),
            }
        )
    match = outcomes_match and digests_match and len(missing_traces) == 0
    return {
        "episodes": len(outcomes),
        "outcomes_match": outcomes_match,
        "trace_digests_match": digests_match,
        "match": match,
        "first_digest": digests[baseline_idx] if outcomes else None,
        "baseline_index": baseline_idx,
        "missing_traces": missing_traces,
        "comparisons": comparisons,
    }


def format_determinism_report(report: dict[str, object]) -> str:
    """Human-readable determinism verdict for the ``--determinism-check`` flag."""
    lines = [
        f"determinism episodes={report.get('episodes')} "
        f"outcomes_match={report.get('outcomes_match')} "
        f"trace_digests_match={report.get('trace_digests_match')} "
        f"match={report.get('match')}"
    ]
    comparisons = report.get("comparisons", [])
    assert isinstance(comparisons, list)
    for comparison in comparisons:
        assert isinstance(comparison, dict)
        fields = " ".join(f"{key}={comparison[key]}" for key in sorted(comparison) if key != "pair")
        lines.append(f"  pair={comparison.get('pair')} {fields}")
    return "\n".join(lines)


def _validate_sim_snapshot_for_session(
    settings: RunnerSettings, ping: dict[str, object]
) -> tuple[str, str]:
    """Return (resolved_snapshot_path, content_hash) or raise with clear available fields."""
    requested = str(settings.kinematic_model or SETTINGS.kinematic_model)
    if requested != "sim":
        resolved = resolve_snapshot_path(SIM_KINEMATICS_PATH)
        hash_value = None
        if resolved.is_file():
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    hash_value = payload.get("content_hash")
            except Exception:
                hash_value = None
        return str(resolved), str(hash_value) if hash_value is not None else ""
    try:
        load_snapshot(SIM_KINEMATICS_PATH)
    except Exception as exc:
        raise RuntimeError(
            "kinematic_model=sim refused: snapshot missing/invalid at "
            f"{SIM_KINEMATICS_PATH} ({exc})"
        ) from exc
    resolved = resolve_snapshot_path(SIM_KINEMATICS_PATH)
    payload = {}
    try:
        payload_obj = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload_obj, dict):
            payload = payload_obj
    except Exception:
        payload = {}
    snap_hash = str(payload.get("content_hash") or "")
    snap_webots = payload.get("webots_version")
    snap_session = payload.get("session_id")
    # ping/session fields available from running server
    ping_session = ping.get("session_id") if isinstance(ping, dict) else None
    ping_payload = ping.get("payload") if isinstance(ping, dict) else None
    if isinstance(ping_payload, dict) and ping_session is None:
        ping_session = ping_payload.get("session_id")
    session_webots = None
    session_world_hash = None
    if isinstance(ping_payload, dict):
        session_webots = ping_payload.get("webots_version")
        session_world_hash = ping_payload.get("world_hash") or ping_payload.get("webots_world_hash")
    # webots version check if both available and comparable
    if (
        isinstance(snap_webots, str)
        and snap_webots
        and isinstance(session_webots, str)
        and session_webots
    ):
        if snap_webots != session_webots:
            raise RuntimeError(
                "kinematic_model=sim refused: snapshot webots_version differs from running session "
                f"(snapshot={snap_webots!r}, session={session_webots!r}, "
                f"snapshot_hash={snap_hash!r}, snapshot_session={snap_session!r}, "
                f"session_id={ping_session!r})"
            )
    # world hash: refuse if different from running session; if unverifiable, fail clearly.
    # If session does not expose world hash we cannot prove equality/difference; fail clearly.
    snap_world_hash = payload.get("webots_world_hash") or payload.get("world_hash")
    if snap_world_hash is not None or session_world_hash is not None:
        if snap_world_hash != session_world_hash:
            raise RuntimeError(
                "kinematic_model=sim refused: snapshot world hash differs from running session "
                f"(snapshot_world_hash={snap_world_hash!r}, "
                f"session_world_hash={session_world_hash!r}, "
                f"snapshot_webots_version={snap_webots!r}, "
                f"session_webots_version={session_webots!r}, "
                f"snapshot_hash={snap_hash!r}, snapshot_session={snap_session!r}, "
                f"session_id={ping_session!r})"
            )
    else:
        # world hash unavailable on both sides -> cannot verify identity to running session
        raise RuntimeError(
            "kinematic_model=sim refused: cannot verify snapshot/world identity against "
            "running session (session world hash unavailable, snapshot world hash unavailable); "
            f"available fields: snapshot_hash={snap_hash!r}, "
            f"snapshot_webots_version={snap_webots!r}, snapshot_session={snap_session!r}, "
            f"session_id={ping_session!r}, session_webots_version={session_webots!r}"
        )
    return str(resolved), snap_hash


def run_episodes(
    settings: RunnerSettings,
    *,
    n: int,
    fresh_world: bool,
    determinism_check: int = 0,
) -> list[EpisodeOutcome]:
    """Run ``n`` episodes and return one outcome per episode.

    ``fresh_world=True`` launches a new Webots process for every episode, so each run starts from
    a reloaded world. ``determinism_check`` overrides the batch: the single seeded episode runs
    ``K`` times, each in its own fresh world, and the caller compares outcomes/traces.
    """
    if determinism_check > 0:
        # A determinism check repeats ONE scene: pin the scene set to its first entry so
        # scene cycling cannot masquerade as nondeterminism (ADR-0025).
        if settings.scenario:
            settings = replace(settings, scenario=settings.scenario[:1])
        outcomes: list[EpisodeOutcome] = []
        for index in range(determinism_check):
            try:
                outcomes.append(_run_fresh_episode(settings, index, settings.seed))
            except Exception as exc:
                suffix = f"{settings.trace_prefix}-{index:02d}"
                outcomes.append(
                    _error_outcome(
                        index,
                        settings.seed,
                        settings.trace_dir / f"{suffix}.jsonl",
                        settings.trace_dir / f"{suffix}.log",
                        0.0,
                        0.0,
                        failure_state="RUNNER",
                        failure_reason=f"{type(exc).__name__}: {exc}",
                        scenario=scenario_for_index(settings, index).name,
                    )
                )
        return outcomes
    if fresh_world:
        outcomes = []
        for index in range(n):
            try:
                outcomes.append(_run_fresh_episode(settings, index, settings.seed + index))
            except Exception as exc:
                suffix = f"{settings.trace_prefix}-{index:02d}"
                outcomes.append(
                    _error_outcome(
                        index,
                        settings.seed + index,
                        settings.trace_dir / f"{suffix}.jsonl",
                        settings.trace_dir / f"{suffix}.log",
                        0.0,
                        0.0,
                        failure_state="RUNNER",
                        failure_reason=f"{type(exc).__name__}: {exc}",
                        scenario=scenario_for_index(settings, index).name,
                    )
                )
        return outcomes
    return _run_shared_session(settings, n)


def _reserve_free_port() -> int:
    """Pick a free localhost TCP port (released immediately for the simulator to bind)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _episode_endpoint_port(settings: RunnerSettings) -> tuple[str, int]:
    """Endpoint and port for one episode: unique free port by default for hygiene."""
    # Respect an explicit non-default endpoint only when no per-episode isolation requested
    # is implied by run_episodes' fresh-world flow; callers pass settings.endpoint.
    # For hygiene we always allocate a free port unless caller pinned a specific tcp port
    # different from the default and wants to reuse it (shared session keeps one port).
    port = _reserve_free_port()
    return endpoint_for_port(port), port


def _run_shared_session(settings: RunnerSettings, n: int) -> list[EpisodeOutcome]:
    """One Webots process serves every episode (soft reset per episode) with process hygiene."""
    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    webots = resolve_webots(settings.webots)
    trace_path = settings.trace_dir / f"{settings.trace_prefix}-session.jsonl"
    log_path = settings.trace_dir / f"{settings.trace_prefix}-session.log"
    endpoint, port = _episode_endpoint_port(settings)
    pid_path = settings.trace_dir / f"{settings.trace_prefix}-session.pid"
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        command = build_webots_command(webots, settings.world, headless=settings.headless)
        env = webots_environment(webots)
        env["VLAROBOT_ZMQ_PORT"] = str(port)
        process = _popen(command, env, log_file)
        _write_pid_file(pid_path, process.pid)
        outcomes: list[EpisodeOutcome] = []
        policy = ScriptedCanPolicy(
            trace_path=trace_path,
            pregrasp_realign=settings.pregrasp_realign,
            closed_loop_waypoints=settings.closed_loop_waypoints,
            stop_after=settings.stop_after,
            grasp_servo=settings.grasp_servo,
        )
        try:
            startup_started = perf_counter()
            client = wait_for_simulator(endpoint, process, settings.startup_timeout_s)
            startup_s = perf_counter() - startup_started
            try:
                ping = client.ping()
                requested_model = str(settings.kinematic_model or SETTINGS.kinematic_model)
                snap_path, snap_hash = _validate_sim_snapshot_for_session(settings, ping)
                for index in range(n):
                    episode_started = perf_counter()
                    try:
                        # One policy serves the session; re-select the scene per episode.
                        policy.scenario = scenario_for_index(settings, index)
                        result = policy.execute(client, settings.seed + index)
                        outcomes.append(
                            _outcome(
                                index,
                                settings.seed + index,
                                result,
                                trace_path,
                                log_path,
                                startup_s,
                                perf_counter() - episode_started,
                                kinematic_model=requested_model,
                                snapshot_hash=snap_hash or None,
                                snapshot_path=snap_path or None,
                                grasp_servo=settings.grasp_servo,
                                scenario=scenario_for_index(settings, index).name,
                            )
                        )
                    except Exception as exc:
                        outcomes.append(
                            _error_outcome(
                                index,
                                settings.seed + index,
                                trace_path,
                                log_path,
                                startup_s,
                                perf_counter() - episode_started,
                                failure_state="EXCEPTION",
                                failure_reason=f"{type(exc).__name__}: {exc}",
                                kinematic_model=requested_model,
                                snapshot_hash=snap_hash or None,
                                snapshot_path=snap_path or None,
                                grasp_servo=settings.grasp_servo,
                                scenario=scenario_for_index(settings, index).name,
                            )
                        )
                        break
            finally:
                _stop_client(client)
            # Let Webots exit on its own after simulationQuit(0) before the runner sends
            # SIGTERM. SIGTERM can race with Webots' exit sequence and cause the runner's
            # 30s timeout to escalate to SIGKILL (-9) even though Webots would have exited
            # cleanly on its own. Waiting here avoids that race.
            for _ in range(10):
                if process.poll() is not None:
                    break
                time.sleep(1.0)
        finally:
            _terminate_process(process, settings.shutdown_timeout_s, pid_path=pid_path)
    return outcomes


@dataclass(frozen=True)
class WebotsSession:
    """One live Webots process plus its connected brain-side client."""

    client: ZmqBackendClient
    endpoint: str
    port: int
    process: subprocess.Popen[str]
    log_path: Path
    pid_path: Path
    startup_s: float


@contextmanager
def launch_webots_session(
    *,
    webots: str | None = None,
    world: Path = DEFAULT_WORLD,
    trace_dir: Path,
    trace_prefix: str = "session",
    headless: bool = False,
    startup_timeout_s: float = 180.0,
    shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
    grace_s: float = 10.0,
    extra_env: Mapping[str, str] | None = None,
) -> Generator[WebotsSession, None, None]:
    """Launch one Webots process that can serve many episodes (soft reset per episode).

    Yields a connected client once the lockstep server answers ``ping()``. On exit the client
    asks the simulator to quit, Webots gets ``grace_s`` to exit on its own, then the whole
    process tree is terminated (polite, then forced). Process hygiene matches
    :func:`run_episodes`; the log and pid file land in ``trace_dir``.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_webots(webots)
    log_path = trace_dir / f"{trace_prefix}.log"
    pid_path = trace_dir / f"{trace_prefix}.pid"
    port = _reserve_free_port()
    endpoint = endpoint_for_port(port)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        command = build_webots_command(resolved, world, headless=headless)
        env = webots_environment(resolved)
        if extra_env:
            env.update(extra_env)
        env["VLAROBOT_ZMQ_PORT"] = str(port)
        process = _popen(command, env, log_file)
        _write_pid_file(pid_path, process.pid)
        try:
            startup_started = perf_counter()
            client = wait_for_simulator(endpoint, process, startup_timeout_s)
            startup_s = perf_counter() - startup_started
            try:
                yield WebotsSession(
                    client=client,
                    endpoint=endpoint,
                    port=port,
                    process=process,
                    log_path=log_path,
                    pid_path=pid_path,
                    startup_s=startup_s,
                )
            finally:
                _stop_client(client)
                # Let Webots exit on its own after simulationQuit(0); SIGTERM can race with
                # its shutdown sequence and force an unnecessary SIGKILL.
                deadline = perf_counter() + max(0.0, grace_s)
                while perf_counter() < deadline and process.poll() is None:
                    time.sleep(0.1)
        finally:
            _terminate_process(process, shutdown_timeout_s, pid_path=pid_path)


def _run_fresh_episode(settings: RunnerSettings, index: int, seed: int) -> EpisodeOutcome:
    """One fresh Webots process, one client, one seed, one trace, one outcome."""
    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    webots = resolve_webots(settings.webots)
    suffix = f"{settings.trace_prefix}-{index:02d}"
    trace_path = settings.trace_dir / f"{suffix}.jsonl"
    if trace_path.is_file():
        trace_path.unlink()
    log_path = settings.trace_dir / f"{suffix}.log"
    pid_path = settings.trace_dir / f"{suffix}.pid"
    controller_log_path = settings.trace_dir / f"{suffix}.controller.log"
    exit_path = settings.trace_dir / f"{suffix}.exit.json"
    endpoint, port = _episode_endpoint_port(settings)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        command = build_webots_command(webots, settings.world, headless=settings.headless)
        env = webots_environment(webots)
        env["VLAROBOT_ZMQ_PORT"] = str(port)
        # Crash forensics (ADR-0022): controller logs + fatal-signal dumps per episode.
        env["VLAROBOT_CONTROLLER_STDERR_PATH"] = str(controller_log_path)
        process = _popen(command, env, log_file)
        _write_pid_file(pid_path, process.pid)
        startup_s = 0.0
        episode_started = perf_counter()
        requested_model = str(settings.kinematic_model or SETTINGS.kinematic_model)
        snap_path: str | None = None
        snap_hash: str | None = None
        try:
            startup_started = perf_counter()
            client = wait_for_simulator(endpoint, process, settings.startup_timeout_s)
            startup_s = perf_counter() - startup_started
            try:
                ping = client.ping()
                snap_path, snap_hash = _validate_sim_snapshot_for_session(settings, ping)
                policy = ScriptedCanPolicy(
                    trace_path=trace_path,
                    pregrasp_realign=settings.pregrasp_realign,
                    closed_loop_waypoints=settings.closed_loop_waypoints,
                    stop_after=settings.stop_after,
                    grasp_servo=settings.grasp_servo,
                    scenario=scenario_for_index(settings, index),
                )
                episode_started = perf_counter()
                result = policy.execute(client, seed)
                return _outcome(
                    index,
                    seed,
                    result,
                    trace_path,
                    log_path,
                    startup_s,
                    perf_counter() - episode_started,
                    kinematic_model=requested_model,
                    snapshot_hash=snap_hash or None,
                    snapshot_path=snap_path or None,
                    grasp_servo=settings.grasp_servo,
                    scenario=scenario_for_index(settings, index).name,
                )
            finally:
                _stop_client(client)
            # Let Webots exit on its own after simulationQuit(0) before the runner sends
            # SIGTERM. SIGTERM can race with Webots' exit sequence and cause the runner's
            # 30s timeout to escalate to SIGKILL (-9) even though Webots would have exited
            # cleanly on its own. Waiting here avoids that race.
            for _ in range(10):
                if process.poll() is not None:
                    break
                time.sleep(1.0)
        except Exception as exc:
            return _error_outcome(
                index,
                seed,
                trace_path,
                log_path,
                startup_s,
                perf_counter() - episode_started,
                failure_state="RUNNER",
                failure_reason=f"{type(exc).__name__}: {exc}",
                kinematic_model=requested_model,
                snapshot_hash=snap_hash or None,
                snapshot_path=snap_path or None,
                grasp_servo=settings.grasp_servo,
                scenario=scenario_for_index(settings, index).name,
            )
        finally:
            _terminate_process(process, settings.shutdown_timeout_s, pid_path=pid_path)
            _record_exit_codes(exit_path, process, controller_log_path)


def _error_outcome(
    index: int,
    seed: int,
    trace_path: Path,
    log_path: Path,
    startup_s: float,
    episode_s: float,
    *,
    failure_state: str,
    failure_reason: str,
    kinematic_model: str | None = None,
    snapshot_hash: str | None = None,
    snapshot_path: str | None = None,
    grasp_servo: str | None = None,
    scenario: str | None = None,
) -> EpisodeOutcome:
    """Synthetic failed outcome for runner/policy exceptions (always recordable)."""
    return EpisodeOutcome(
        index=index,
        seed=seed,
        success=False,
        final_state="FAILED",
        failure_state=failure_state,
        failure_reason=failure_reason,
        final_target_distance_m=float("nan"),
        home_reached=False,
        place_distance_ok=False,
        can_height_ok=False,
        can_tilt_ok=False,
        can_height_error_m=None,
        can_tilt_deg=None,
        rejections=0,
        startup_s=startup_s,
        episode_s=episode_s,
        trace_path=trace_path,
        log_path=log_path,
        kinematic_model=kinematic_model,
        snapshot_hash=snapshot_hash,
        snapshot_path=snapshot_path,
        grasp_servo=grasp_servo,
        scenario=scenario,
    )


def _popen(command: list[str], env: Mapping[str, str], log_file: IO[str]) -> subprocess.Popen[str]:
    """Launch a subprocess arg list (never ``shell=True``) with output captured to a log file.

    Starts a new process group/session on POSIX so the whole Webots/controller tree can be
    terminated on exit/timeout/exception (ADR-0020 addendum process hygiene).
    """
    cwd = str(PROJECT_ROOT)
    env_map = dict(env)
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env_map,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env_map,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )


def wait_for_simulator(
    endpoint: str,
    process: subprocess.Popen[str],
    timeout_s: float,
    poll_s: float = 0.25,
    ping_timeout_ms: int | None = None,
) -> ZmqBackendClient:
    """Repeatedly ping the endpoint with a fresh client until the simulator is ready.

    Uses short per-attempt pings so "server not up yet" (startup) is retried until
    ``timeout_s`` without treating a slow boot as a single long request timeout.
    """
    deadline = perf_counter() + timeout_s
    last_error: Exception | None = None
    attempt_timeout = int(
        ping_timeout_ms if ping_timeout_ms is not None else REQUEST_TIMEOUTS_MS["ping"]
    )
    while perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Webots exited with code {process.returncode} before the simulator "
                f"server at {endpoint!r} became ready."
            )
        client = ZmqBackendClient(endpoint, timeout_ms=attempt_timeout)
        try:
            client.ping()
        except (TimeoutError, RuntimeError, OSError) as error:
            last_error = error
            client.close()
            time.sleep(poll_s)
            continue
        return client
    raise TimeoutError(
        f"Simulator at {endpoint!r} did not answer ping within {timeout_s:.0f}s "
        f"(startup wait; server not up/unreachable, last error: {last_error!r})."
    )


def _stop_client(client: ZmqBackendClient) -> None:
    """Ask the simulator to shut down, then close the client socket either way."""
    try:
        client.stop()
    except (RuntimeError, TimeoutError, OSError):
        pass
    try:
        client.close()
    except Exception:
        pass


def _write_pid_file(pid_path: Path, pid: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid), encoding="utf-8")


def _record_exit_codes(
    exit_path: Path, process: subprocess.Popen[str], controller_log: Path
) -> None:
    """Record Webots/controller exit evidence for crash forensics (ADR-0022).

    ``webots_exit_code`` is ``None`` when the process could not be inspected. The tail of the
    per-episode controller diagnostics file (including ``server_exit_code`` and any
    faulthandler dump) is embedded so one file tells the whole story.
    """
    controller_tail: list[str] = []
    try:
        if controller_log.is_file():
            lines = controller_log.read_text(encoding="utf-8", errors="replace").splitlines()
            controller_tail = lines[-30:]
    except OSError:
        controller_tail = ["<unreadable controller log>"]
    payload = {
        "webots_exit_code": process.returncode,
        "controller_log_tail": controller_tail,
    }
    try:
        exit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _remove_pid_file(pid_path: Path | None) -> None:
    if pid_path is None:
        return
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _kill_process_tree(process: subprocess.Popen[str], *, force: bool) -> None:
    """Terminate the full Webots/controller process tree, cross-platform."""
    if process.poll() is not None:
        return
    pid = process.pid
    try:
        if os.name == "nt":
            # /T kills child processes (e.g. controller) too.
            cmd = ["taskkill", "/PID", str(pid), "/T"]
            if force:
                cmd.append("/F")
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    except Exception:
        # Last resort: direct process handle.
        try:
            (process.kill() if force else process.terminate())
        except Exception:
            pass


def _terminate_process(
    process: subprocess.Popen[str], timeout_s: float, pid_path: Path | None = None
) -> None:
    """Terminate the simulator process tree politely, then force-kill if needed."""
    try:
        if process.poll() is not None:
            return
        _kill_process_tree(process, force=False)
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            _kill_process_tree(process, force=True)
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
    finally:
        _remove_pid_file(pid_path)


def _outcome(
    index: int,
    seed: int,
    result: ScriptedCanResult,
    trace_path: Path,
    log_path: Path,
    startup_s: float,
    episode_s: float,
    *,
    kinematic_model: str | None = None,
    snapshot_hash: str | None = None,
    snapshot_path: str | None = None,
    grasp_servo: str | None = None,
    scenario: str | None = None,
) -> EpisodeOutcome:
    """Map one ``ScriptedCanResult`` onto a summary-table row."""
    return EpisodeOutcome(
        index=index,
        seed=seed,
        success=bool(result.success),
        final_state=str(result.final_state),
        failure_state=result.failure_state,
        failure_reason=result.failure_reason,
        final_target_distance_m=float(result.final_target_distance),
        home_reached=bool(result.home_reached),
        place_distance_ok=bool(result.place_distance_ok),
        can_height_ok=bool(result.can_height_ok),
        can_tilt_ok=bool(result.can_tilt_ok),
        can_height_error_m=result.can_height_error_m,
        can_tilt_deg=result.can_tilt_deg,
        rejections=int(result.rejections),
        startup_s=startup_s,
        episode_s=episode_s,
        trace_path=trace_path,
        log_path=log_path,
        pregrasp_realign=result.pregrasp_realign,
        closed_loop_waypoints=result.closed_loop_waypoints,
        kinematic_model=kinematic_model,
        snapshot_hash=snapshot_hash,
        snapshot_path=snapshot_path,
        grasp_servo=grasp_servo,
        scenario=scenario,
    )
