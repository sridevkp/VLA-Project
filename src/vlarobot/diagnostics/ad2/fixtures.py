"""Scripted CPU fixtures only: no process launch, transport, backend or actuator exists."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .admission import Protocol, State, compare
from .evidence import Targets


@dataclass(frozen=True)
class SnapshotBuffer:
    """Already-captured immutable values, not arbitrary callbacks or a live API wrapper."""

    state: State
    tick: int
    time_s: float
    startup_ticks: int
    reset_ticks: int


def snapshot(buffer: SnapshotBuffer) -> SnapshotBuffer:
    """Copy only a sealed value buffer; never call backend observation/history helpers."""
    if type(buffer) is not SnapshotBuffer or type(buffer.state) is not State:
        raise TypeError("Only offline value buffers are supported")
    if not all(
        type(v) is int and v >= 0 for v in (buffer.tick, buffer.startup_ticks, buffer.reset_ticks)
    ):
        raise ValueError("Invalid tick ledger")
    if type(buffer.time_s) not in (float, int) or not math.isfinite(buffer.time_s):
        raise ValueError("Invalid snapshot time")
    return buffer


@dataclass
class FixtureProcess:
    """Inert process identity; alive is a fixture flag, not an operating-system process."""

    pid: int
    nonce: str
    port: int
    alive: bool = False


@dataclass(frozen=True)
class FixtureTransport:
    """Predetermined outcomes; it cannot open a socket or send a command."""

    accepted: bool
    ambiguous_at: str | None = None
    infrastructure_error: str | None = None
    reply_nonce_override: str | None = None
    reply_sequence_override: int | None = None


@dataclass(frozen=True)
class FixtureSession:
    """A scripted session with predeclared observations; not a simulator adapter."""

    case: str
    process: FixtureProcess
    transport: FixtureTransport
    before: SnapshotBuffer
    after: SnapshotBuffer


@dataclass(frozen=True)
class FixtureResult:
    """No diagnostic/scored physical result is produced by this fixture runner."""

    status: str
    completed: tuple[str, ...]
    events: tuple[dict, ...]


def run_fixtures(
    targets: Targets,
    reference: State,
    protocol: Protocol,
    sessions: tuple[FixtureSession, ...],
    output: Path,
) -> FixtureResult:
    """Exercise exactly three inert session descriptions, persisting partial evidence.

    This deliberately does not accept a factory, executable, RPC client, backend or live
    getter. Test-only approval IDs are mandatory; there is no physical execution path.
    """
    if not protocol.approval_id.startswith("fixture:"):
        raise ValueError("Fixture runner requires test-only protocol identity")
    if len(sessions) != 3 or any(type(s) is not FixtureSession for s in sessions):
        raise ValueError("Exactly three inert fixture sessions are required")
    if tuple(s.case for s in sessions) != ("A", "B", "C"):
        raise ValueError("Fixed A/B/C ordering required")
    for s in sessions:
        if type(s.process) is not FixtureProcess or type(s.transport) is not FixtureTransport:
            raise TypeError("Live adapters/callbacks are unsupported")
    output.mkdir(parents=True, exist_ok=False)
    events: list[dict] = []
    completed: list[str] = []
    previous: list[State] = []
    identities: list[FixtureProcess] = []

    def record(event: str, **values: object) -> None:
        item = {"event": event, **values}
        events.append(item)
        # Flush every event so an interruption never silently erases acknowledged work.
        with (output / "fixture-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    record(
        "fixture-protocol",
        protocol=asdict(protocol),
        manifest_sha256=targets.manifest_sha256,
        diagnostic_identity=targets.identity,
    )
    status = "fixture-completed"
    for session in sessions:
        process = session.process
        owned = False
        try:
            if process.alive or not process.nonce or process.pid <= 0 or process.port <= 0:
                raise ValueError("Conflicting process/session identity")
            if any(
                process.pid == p.pid or process.nonce == p.nonce or process.port == p.port
                for p in identities
            ):
                raise ValueError("Session reuse")
            if any(p.alive for p in identities):
                raise ValueError("Prior fixture process remains alive")
            identities.append(process)
            process.alive = True
            owned = True
            record("fixture-start", case=session.case, pid=process.pid, nonce=process.nonce)
            if session.transport.ambiguous_at not in (None, "apply", "step"):
                raise ValueError("Unknown ambiguous receipt")
            if session.transport.infrastructure_error:
                raise RuntimeError(session.transport.infrastructure_error)
            before = snapshot(session.before)
            if before.tick != before.startup_ticks + before.reset_ticks:
                raise ValueError("Unaccounted preparation ticks")
            record(
                "preparation",
                startup_ticks=before.startup_ticks,
                reset_ticks=before.reset_ticks,
                tick=before.tick,
                time_s=before.time_s,
            )
            admitted = compare(reference, before.state, tuple(previous), protocol)
            record("admission", **asdict(admitted))
            if not admitted.accepted:
                raise ValueError("Reset admission rejected")
            record("fixture-before-state", state=asdict(before.state))
            record("fixture-apply-dispatched", case=session.case, target=targets.case(session.case))
            if session.transport.ambiguous_at == "apply":
                raise TimeoutError("Ambiguous apply: do not resend")
            nonce = session.transport.reply_nonce_override
            seq = session.transport.reply_sequence_override
            if (nonce is not None and nonce != process.nonce) or (seq is not None and seq != 1):
                raise ValueError("Reply nonce/sequence mismatch; no resend")
            record("fixture-apply-result", accepted=session.transport.accepted)
            if not session.transport.accepted:
                raise ValueError("SAFETY rejection")
            record("fixture-step-dispatched", count=1)
            if session.transport.ambiguous_at == "step":
                raise TimeoutError("Ambiguous step: do not resend")
            after = snapshot(session.after)
            checked = compare(
                after.state, after.state, (), replace(protocol, historical_omissions=())
            )
            if not checked.accepted:
                raise ValueError("Invalid after snapshot")
            if after.tick - before.tick != 1:
                raise ValueError("Measurement tick mismatch")
            if abs((after.time_s - before.time_s) - 0.032) > protocol.timestamp_tolerance_s:
                raise ValueError("Measurement time mismatch")
            if (after.startup_ticks, after.reset_ticks) != (
                before.startup_ticks,
                before.reset_ticks,
            ):
                raise ValueError("Preparation ledger changed during measurement")
            record("fixture-after", tick=after.tick, time_s=after.time_s, state=asdict(after.state))
            previous.append(before.state)
            completed.append(session.case)
        except Exception as exc:
            status = "fixture-stopped"
            record("failure", case=session.case, kind=type(exc).__name__, reason=str(exc))
        finally:
            # Pure flag cleanup. No stop RPC/step/physical progression can occur here.
            if owned:
                process.alive = False
            record("fixture-cleanup", case=session.case, owned=owned, physics_progression=0)
        if status != "fixture-completed":
            break
    result = FixtureResult(status, tuple(completed), tuple(events))
    with (output / "fixture-result.json").open("x", encoding="utf-8") as stream:
        json.dump(asdict(result), stream, indent=2, allow_nan=False)
    return result
