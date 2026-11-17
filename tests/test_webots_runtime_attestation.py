"""Inert tests for launch-time Webots version attestation.

The version probe is injected; no test in this module launches Webots or any simulator process.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import vlarobot.backends.webots.runtime_attestation as attestation_module
from vlarobot.backends.webots.runtime_attestation import (
    WebotsRuntimeAttestation,
    attach_process_identity,
    attest_webots_executable,
    controller_runtime_version,
    normalize_webots_version,
    observed_process_executable,
    sha256_file,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Webots R2025a", "R2025a"),
        ("Webots R2024B revision 1", "R2024b"),
        ("Webots 2025a", None),
        ("unknown build", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_normalize_webots_version_only_accepts_observed_r_tokens(raw: object, expected: str | None):
    assert normalize_webots_version(raw) == expected


def test_controller_version_comes_from_environment_and_never_from_config():
    assert controller_runtime_version({"WEBOTS_VERSION": "Webots R2025a"}) == (
        "R2025a",
        "WEBOTS_VERSION",
    )
    assert controller_runtime_version({}) == (None, None)


def test_sha256_file_returns_none_for_unreadable_path(tmp_path: Path):
    assert sha256_file(tmp_path / "missing-executable") is None


def test_attest_webots_executable_probes_the_exact_resolved_path_and_hashes_it(tmp_path: Path):
    executable = tmp_path / "webots-stub"
    executable.write_bytes(b"inert executable bytes\n")
    calls: list[dict[str, Any]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Webots R2025a\nrevision: synthetic\n",
            stderr="",
        )

    result = attest_webots_executable(
        executable,
        {"PROBE_MARKER": "inert"},
        runner=runner,
        timeout_s=1.0,
    )

    assert len(calls) == 1
    assert calls[0]["command"] == [str(executable.resolve()), "--version"]
    assert calls[0]["env"]["PROBE_MARKER"] == "inert"
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True
    assert calls[0]["check"] is False
    assert calls[0]["timeout"] == 1.0
    assert result.executable_path == str(executable.resolve())
    assert result.launch_command_executable == str(executable.resolve())
    assert result.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert result.reported_version == "R2025a"
    assert result.version_probe_returncode == 0
    assert result.version_probe_error is None
    assert result.version_output_available is True
    assert result.verified_executable_version_pair is True
    assert result.evidence_source == "launch_time_executable_version_probe"
    assert result.captured_at_utc
    assert result.process_identity_verified is False


def test_attest_webots_executable_records_missing_version_without_inventing_one(tmp_path: Path):
    executable = tmp_path / "webots-stub"
    executable.write_bytes(b"inert executable bytes\n")

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="unknown build\n", stderr="")

    result = attest_webots_executable(executable, {}, runner=runner)

    assert result.reported_version is None
    assert result.version_output_available is False
    assert result.version_probe_error == "version probe output contained no Webots version token"
    assert result.verified_executable_version_pair is False


def test_attest_webots_executable_records_unavailable_probe_evidence(tmp_path: Path):
    executable = tmp_path / "webots-stub"
    executable.write_bytes(b"inert executable bytes\n")

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("probe unavailable")

    result = attest_webots_executable(executable, {}, runner=runner)

    assert result.reported_version is None
    assert result.version_probe_returncode is None
    assert result.version_probe_error is not None
    assert "FileNotFoundError" in result.version_probe_error
    assert result.version_output_available is False
    assert result.verified_executable_version_pair is False


def test_attach_process_identity_accepts_launcher_shebang_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    executable = tmp_path / "webots-launcher"
    executable.write_text('#!/opt/webots/interpreter\nexec webots-bin "$@"\n', encoding="utf-8")
    attestation = WebotsRuntimeAttestation(
        executable_path=str(executable),
        executable_sha256="a" * 64,
        reported_version="R2025a",
        launch_command_executable=str(executable),
        version_probe_returncode=0,
    )
    monkeypatch.setattr(
        attestation_module,
        "observed_process_executable",
        lambda process: ("/opt/webots/interpreter", "linux_proc_exe"),
    )

    result = attach_process_identity(attestation, SimpleNamespace(pid=321))

    assert result.process_pid == 321
    assert result.process_executable_path == "/opt/webots/interpreter"
    assert result.process_identity_source == "linux_proc_exe"
    assert result.process_identity_verified is True


def test_attach_process_identity_rejects_wrong_process_executable(
    monkeypatch: pytest.MonkeyPatch,
):
    attestation = WebotsRuntimeAttestation(
        executable_path="/opt/webots/webots-bin",
        executable_sha256="b" * 64,
        reported_version="R2025a",
        launch_command_executable="/opt/webots/webots-bin",
        version_probe_returncode=0,
    )
    monkeypatch.setattr(
        attestation_module,
        "observed_process_executable",
        lambda process: ("/opt/other/webots-bin", "linux_proc_exe"),
    )

    result = attach_process_identity(attestation, SimpleNamespace(pid=654))

    assert result.process_pid == 654
    assert result.process_executable_path == "/opt/other/webots-bin"
    assert result.process_identity_verified is False


def test_observed_process_executable_marks_invalid_pid_unavailable():
    assert observed_process_executable(SimpleNamespace(pid=-1)) == (None, "unavailable")


def test_attestation_mapping_includes_provenance_and_verification_state():
    attestation = WebotsRuntimeAttestation(
        executable_path="/opt/webots/webots",
        executable_sha256="c" * 64,
        reported_version="R2025a",
        launch_command_executable="/opt/webots/webots",
        version_probe_command=("/opt/webots/webots", "--version"),
        version_probe_returncode=0,
        process_pid=42,
        process_executable_path="/opt/webots/webots",
        process_identity_source="linux_proc_exe",
        process_identity_verified=True,
        evidence_source="launch_time_executable_version_probe",
        captured_at_utc="2026-11-18T00:00:00+00:00",
    )

    payload = attestation.to_mapping()

    assert payload["executable_path"] == "/opt/webots/webots"
    assert payload["executable_sha256"] == "c" * 64
    assert payload["reported_version"] == "R2025a"
    assert payload["version_probe_command"] == ["/opt/webots/webots", "--version"]
    assert payload["process_identity_verified"] is True
    assert payload["verified_executable_version_pair"] is True
    assert payload["evidence_source"] == "launch_time_executable_version_probe"
    assert payload["captured_at_utc"] == "2026-11-18T00:00:00+00:00"
