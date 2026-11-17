"""Unit tests for the fresh-world episode runner's pure helpers (no subprocess launches)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vlarobot.execution.episode_runner as episode_runner
from vlarobot.backends.webots.runtime_attestation import WebotsRuntimeAttestation
from vlarobot.execution.episode_runner import (
    EpisodeOutcome,
    build_webots_command,
    determinism_report,
    format_determinism_report,
    format_summary_table,
    normalized_outcome,
    normalized_trace_rows,
    resolve_webots,
    trace_digest,
    trace_numeric_comparison,
    webots_environment,
)

WORLD = Path("worlds/factory.wbt")


def _outcome_row(index: int = 0, trace_path: Path | None = None) -> EpisodeOutcome:
    return EpisodeOutcome(
        index=index,
        seed=7,
        success=True,
        final_state="home",
        failure_state=None,
        failure_reason=None,
        final_target_distance_m=0.04,
        home_reached=True,
        place_distance_ok=False,
        can_height_ok=True,
        can_tilt_ok=True,
        can_height_error_m=0.0,
        can_tilt_deg=0.0,
        rejections=0,
        startup_s=1.5,
        episode_s=12.0,
        trace_path=trace_path or Path("runs/episode-00.jsonl"),
        log_path=Path("runs/episode-00.log"),
        pregrasp_realign=True,
        closed_loop_waypoints=True,
    )


def test_build_webots_command_keeps_rendering_by_default() -> None:
    command = build_webots_command("webots", WORLD)
    assert command == ["webots", "--batch", "--mode=fast", "--stdout", "--stderr", str(WORLD)]


def test_build_webots_command_headless_adds_no_rendering() -> None:
    command = build_webots_command("webots", WORLD, headless=True)
    assert command == [
        "webots",
        "--batch",
        "--mode=fast",
        "--no-rendering",
        "--stdout",
        "--stderr",
        str(WORLD),
    ]


def test_webots_environment_sets_missing_webots_home() -> None:
    env = webots_environment("/opt/webots/webots", {"PATH": "/usr/bin"})
    assert env["WEBOTS_HOME"] == "/opt/webots"
    assert env["PATH"] == "/usr/bin"


def test_webots_environment_preserves_existing_webots_home() -> None:
    env = webots_environment("/opt/webots/webots", {"WEBOTS_HOME": "/custom"})
    assert env["WEBOTS_HOME"] == "/custom"


def _write_executable(path: Path) -> Path:
    path.write_bytes(b"#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_resolve_webots_accepts_an_explicit_executable_path(tmp_path: Path) -> None:
    binary = _write_executable(tmp_path / "webots-custom")
    assert resolve_webots(str(binary)) == str(binary)


def test_resolve_webots_uses_path_for_bare_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    binary = _write_executable(tmp_path / "webots")
    monkeypatch.delenv("WEBOTS_HOME", raising=False)
    monkeypatch.chdir(work)
    monkeypatch.setattr(episode_runner.shutil, "which", lambda command: str(binary))

    assert resolve_webots("webots") == str(binary.resolve())


def test_resolve_webots_bare_command_falls_back_to_webots_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "webots-home"
    home.mkdir()
    binary = _write_executable(home / "webots")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("WEBOTS_HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setattr(episode_runner.shutil, "which", lambda command: None)

    assert resolve_webots("webots") == str(binary.resolve())


def test_resolve_webots_uses_webots_home_first(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "webots-home"
    home.mkdir()
    binary = _write_executable(home / "webots")
    monkeypatch.setenv("WEBOTS_HOME", str(home))
    monkeypatch.delenv("VLAROBOT_TEST_PATH", raising=False)
    assert resolve_webots(None) == str(binary)


def test_resolve_webots_rejects_invalid_explicit_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = tmp_path / "webots-directory"
    directory.mkdir()
    non_executable = tmp_path / "webots-not-executable"
    non_executable.write_bytes(b"not executable\n")

    with pytest.raises(FileNotFoundError):
        resolve_webots(str(tmp_path / "definitely-not-here"))
    with pytest.raises(ValueError, match="is a directory"):
        resolve_webots(str(directory))
    monkeypatch.setattr(episode_runner, "_is_executable_file", lambda path: False)
    with pytest.raises(PermissionError, match="not executable"):
        resolve_webots(str(non_executable))


def test_resolve_webots_rejects_ambiguous_bare_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "webots-home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    _write_executable(home / "webots")
    _write_executable(work / "webots")
    monkeypatch.setenv("WEBOTS_HOME", str(home))
    monkeypatch.chdir(work)
    monkeypatch.setattr(episode_runner.shutil, "which", lambda command: None)

    with pytest.raises(ValueError, match="ambiguous Webots executable candidates"):
        resolve_webots("webots")


def test_resolve_webots_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.delenv("WEBOTS_HOME", raising=False)
    monkeypatch.chdir(work)
    monkeypatch.setattr(episode_runner.shutil, "which", lambda command: None)

    with pytest.raises(FileNotFoundError):
        resolve_webots(str(WORLD / "definitely-not-here"))
    with pytest.raises(FileNotFoundError, match="Webots was not found"):
        resolve_webots(None)
    with pytest.raises(FileNotFoundError, match="was not found"):
        resolve_webots("webots")


def test_launch_attestation_uses_identical_probe_and_launch_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = str((tmp_path / "webots-launcher").resolve())
    attestation = WebotsRuntimeAttestation(
        executable_path=resolved,
        executable_sha256="a" * 64,
        reported_version="R2025a",
        launch_command_executable=resolved,
        version_probe_command=(resolved, "--version"),
        version_probe_returncode=0,
        evidence_source="launch_time_executable_version_probe",
    )
    probe_calls: list[str] = []
    popen_calls: list[list[str]] = []
    attached_processes: list[object] = []
    process = SimpleNamespace(pid=4242, poll=lambda: 0, returncode=0)

    monkeypatch.setattr(episode_runner, "resolve_webots", lambda webots=None: resolved)
    monkeypatch.setattr(episode_runner, "_reserve_free_port", lambda: 5555)
    monkeypatch.setattr(episode_runner, "wait_for_simulator", lambda *args: object())
    monkeypatch.setattr(episode_runner, "_stop_client", lambda client: None)

    def fake_attest(executable: str, env: object, **kwargs: object) -> WebotsRuntimeAttestation:
        probe_calls.append(executable)
        return attestation

    def fake_popen(command: list[str], env: object, log_file: object) -> object:
        popen_calls.append(list(command))
        return process

    def fake_attach(
        current: WebotsRuntimeAttestation, attached: object
    ) -> WebotsRuntimeAttestation:
        attached_processes.append(attached)
        return replace(current, process_pid=4242, process_identity_verified=True)

    monkeypatch.setattr(episode_runner, "attest_webots_executable", fake_attest)
    monkeypatch.setattr(episode_runner, "_popen", fake_popen)
    monkeypatch.setattr(episode_runner, "attach_process_identity", fake_attach)

    with episode_runner.launch_webots_session(
        webots="webots",
        world=WORLD,
        trace_dir=tmp_path,
        runtime_attestation_required=True,
        grace_s=0.0,
    ) as session:
        assert session.runtime_attestation["executable_path"] == resolved
        assert session.runtime_attestation["launch_command_executable"] == resolved
        assert session.runtime_attestation["version_probe_command"][0] == resolved

    assert probe_calls == [resolved]
    assert popen_calls[0][0] == resolved
    assert attached_processes == [process]


def test_format_summary_table_renders_rows_and_summary(tmp_path: Path) -> None:
    ok_row = _outcome_row(0, tmp_path / "episode-00.jsonl")
    failed = replace(
        ok_row,
        index=1,
        success=False,
        final_state="DESCEND",
        failure_state="DESCEND",
        failure_reason="contact",
        home_reached=False,
        trace_path=tmp_path / "episode-01.jsonl",
        log_path=tmp_path / "episode-01.log",
    )
    table = format_summary_table([ok_row, failed])
    lines = table.splitlines()
    assert "epi seed outcome" in lines[0] and "fail_state" in lines[0]
    assert "ok" in lines[2] and "home" in lines[2]
    assert "no" in lines[3] and "DESCEND" in lines[3]
    assert lines[-1].startswith("summary success=1/2")


def test_outcome_to_dict_serializes_paths() -> None:
    payload = _outcome_row().to_dict()
    assert payload["trace_path"] == "runs/episode-00.jsonl"
    assert payload["log_path"] == "runs/episode-00.log"


def test_normalized_outcome_drops_index_paths_and_timings() -> None:
    payload = normalized_outcome(_outcome_row())
    for key in ("index", "trace_path", "log_path", "startup_s", "episode_s"):
        assert key not in payload
    assert payload["success"] is True
    assert payload["seed"] == 7


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _boundary_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event": "state_boundary",
        "phase": "enter",
        "state": "DESCEND",
        "sim_time_s": 1.0,
        "achieved_ee_base_xyz_m": [0.1, 0.2, 0.3],
        "can_world_xyz_m": [0.5, 0.6, 0.7],
        "commanded_ee_base_xyz_m": [0.1, 0.2, 0.3],
        "reset_fingerprint": "volatile-a",
    }
    row.update(overrides)
    return row


def test_normalized_trace_rows_strips_volatile_keys(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, [_boundary_row(), {"event": "trace_meta", "trace_path": "x"}])

    rows = normalized_trace_rows(trace)

    assert rows[0]["event"] == "state_boundary"
    assert "reset_fingerprint" not in rows[0]
    assert "trace_path" not in rows[1]


def test_trace_digest_is_none_for_missing_file_and_stable_for_content(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_trace(first, [_boundary_row()])
    _write_trace(second, [_boundary_row()])

    assert trace_digest(tmp_path / "missing.jsonl") is None
    assert trace_digest(first) == trace_digest(second)

    _write_trace(second, [_boundary_row(sim_time_s=2.0)])
    assert trace_digest(first) != trace_digest(second)


def test_trace_numeric_comparison_reports_max_deltas(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_trace(
        first,
        [
            _boundary_row(sim_time_s=1.0, achieved_ee_base_xyz_m=[0.1, 0.2, 0.3]),
            {"event": "trace_meta"},
        ],
    )
    _write_trace(
        second,
        [
            _boundary_row(
                sim_time_s=1.125,
                achieved_ee_base_xyz_m=[0.101, 0.199, 0.3],
                can_world_xyz_m=[0.501, 0.6, 0.7],
            ),
            {"event": "trace_meta"},
        ],
    )

    report = trace_numeric_comparison(first, second)

    assert report["boundary_rows_equal"] is True
    assert report["boundary_rows"] == 1
    assert report["sim_time_max_delta_s"] == pytest.approx(0.125)
    assert report["achieved_ee_max_delta_m"] == pytest.approx(0.001)
    assert report["can_world_max_delta_m"] == pytest.approx(0.001)
    assert report["commanded_ee_max_delta_m"] == pytest.approx(0.0)


def test_determinism_report_matches_identical_repeats(tmp_path: Path) -> None:
    first = tmp_path / "episode-00.jsonl"
    second = tmp_path / "episode-01.jsonl"
    _write_trace(first, [_boundary_row()])
    _write_trace(second, [_boundary_row()])
    outcome_a = _outcome_row(0, first)
    # Identical run except for the per-episode index, paths, and wall-clock timings.
    outcome_b = replace(
        outcome_a,
        index=1,
        startup_s=9.0,
        episode_s=99.0,
        trace_path=second,
        log_path=tmp_path / "episode-01.log",
    )

    report = determinism_report([outcome_a, outcome_b])

    assert report["outcomes_match"] is True
    assert report["trace_digests_match"] is True
    assert report["match"] is True
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list) and comparisons
    assert comparisons[0]["boundary_rows_equal"] is True


def test_determinism_report_flags_mismatched_outcomes(tmp_path: Path) -> None:
    first = tmp_path / "episode-00.jsonl"
    second = tmp_path / "episode-01.jsonl"
    _write_trace(first, [_boundary_row()])
    _write_trace(second, [_boundary_row(sim_time_s=1.25)])
    outcome_a = _outcome_row(0, first)
    outcome_b = replace(
        outcome_a,
        index=1,
        success=False,
        final_state="DESCEND",
        failure_state="DESCEND",
        home_reached=False,
        trace_path=second,
        log_path=tmp_path / "episode-01.log",
    )

    report = determinism_report([outcome_a, outcome_b])

    assert report["outcomes_match"] is False
    assert report["match"] is False
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list) and comparisons
    assert comparisons[0]["outcome_match"] is False


def test_format_determinism_report_lists_each_pair(tmp_path: Path) -> None:
    trace = tmp_path / "episode-00.jsonl"
    _write_trace(trace, [_boundary_row()])
    report = determinism_report([_outcome_row(0, trace)])

    text = format_determinism_report(report)

    assert text.startswith("determinism episodes=1")
    # No pairs exist for a single episode.
    assert "pair=" not in text


def test_determinism_report_handles_missing_trace_without_crash(tmp_path: Path) -> None:
    first = tmp_path / "episode-00.jsonl"
    missing = tmp_path / "episode-01.jsonl"
    _write_trace(first, [_boundary_row()])
    outcome_a = _outcome_row(0, first)
    outcome_b = replace(
        outcome_a,
        index=1,
        success=False,
        final_state="FAILED",
        failure_state="RUNNER",
        failure_reason="TimeoutError: reset hung",
        home_reached=False,
        trace_path=missing,
        log_path=tmp_path / "episode-01.log",
    )

    report = determinism_report([outcome_a, outcome_b])

    assert report["match"] is False
    assert report["trace_digests_match"] is False
    assert report["missing_traces"]
    missing_entry = report["missing_traces"][0]
    assert missing_entry["index"] == 1
    assert missing_entry["log_path"].endswith("episode-01.log")
    assert missing_entry["failure_reason"] == "TimeoutError: reset hung"
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list) and comparisons
    assert comparisons[0]["trace_comparison_available"] is False
    assert comparisons[0]["digest_match"] is False


def test_required_attestation_unavailable_fails_before_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(episode_runner, "resolve_webots", lambda webots=None: "/fake/webots")
    monkeypatch.setattr(episode_runner, "_reserve_free_port", lambda: 5555)
    unavailable = WebotsRuntimeAttestation(
        executable_path="/fake/webots",
        executable_sha256=None,
        reported_version=None,
        version_probe_error="probe unavailable",
    )
    monkeypatch.setattr(
        episode_runner,
        "attest_webots_executable",
        lambda *args, **kwargs: unavailable,
    )

    def forbidden_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("no Webots process may start without version evidence")

    monkeypatch.setattr(episode_runner, "_popen", forbidden_popen)

    with pytest.raises(RuntimeError, match="launch-time Webots runtime attestation unavailable"):
        with episode_runner.launch_webots_session(
            trace_dir=tmp_path,
            runtime_attestation_required=True,
        ):
            raise AssertionError("the session must not be yielded")
