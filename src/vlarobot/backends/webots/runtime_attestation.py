"""Launch-time attestation for the Webots executable and controller-reported version.

This module intentionally imports no Webots package. It records independently observed facts:
the resolved executable path and SHA-256, the executable's own ``--version`` output, the spawned
process identity, and the ``WEBOTS_VERSION`` value Webots supplies to controllers. It never reads
the configured expected version, so a mismatch remains visible and fail-closed.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTROLLER_VERSION_ENV = "WEBOTS_VERSION"
VERSION_PROBE_FLAG = "--version"
VERSION_PROBE_TIMEOUT_S = 20.0
_VERSION_RE = re.compile(r"\bR\s*(20\d{2})\s*([a-zA-Z])\b")

_ProbeRunner = Callable[..., subprocess.CompletedProcess[str]]


def normalize_webots_version(value: object) -> str | None:
    """Extract a canonical ``R20XXa`` token from observed Webots version output."""
    if not isinstance(value, str):
        return None
    match = _VERSION_RE.search(value)
    if match is None:
        return None
    return f"R{match.group(1)}{match.group(2).lower()}"


def controller_runtime_version(
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Return the controller-reported version and its provenance, without substituting config."""
    env = os.environ if environ is None else environ
    raw = env.get(CONTROLLER_VERSION_ENV)
    version = normalize_webots_version(raw)
    return version, (CONTROLLER_VERSION_ENV if version is not None else None)


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        first = str(Path(left).expanduser().resolve())
        second = str(Path(right).expanduser().resolve())
    except OSError:
        return False
    if os.name == "nt":
        return first.casefold() == second.casefold()
    return first == second


def _shebang_interpreter(path: str) -> str | None:
    """Return the interpreter a launcher script delegates to, if any."""
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
            first_line = stream.readline().strip()
    except OSError:
        return None
    if not first_line.startswith("#!"):
        return None
    parts = first_line[2:].split()
    if parts and Path(parts[0]).name == "env":
        parts = parts[1:]
    return parts[0] if parts else None


def _windows_process_path(process: Any) -> str | None:
    handle = getattr(process, "_handle", None)
    if handle is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(32768)
        query = ctypes.windll.kernel32.QueryFullProcessImageNameW  # ty: ignore[unresolved-attribute]
        query.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        query.restype = wintypes.BOOL
        if not query(handle, 0, buffer, ctypes.byref(size)):
            return None
        return str(buffer.value)
    except Exception:  # noqa: BLE001 - process introspection is best-effort provenance
        return None


def observed_process_executable(process: Any) -> tuple[str | None, str]:
    """Read the OS-reported executable path for an already-spawned process."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return None, "unavailable"
    if os.name == "nt":
        path = _windows_process_path(process)
        return path, ("windows_query_full_process_image" if path else "unavailable")
    try:
        target = os.readlink(Path("/proc") / str(pid) / "exe")
    except OSError:
        return None, "unavailable"
    return target.removesuffix(" (deleted)"), "linux_proc_exe"


def sha256_file(path: str | Path) -> str | None:
    """Hash an on-disk executable; unreadable files produce no attestation."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


@dataclass(frozen=True)
class WebotsRuntimeAttestation:
    """Observed launch-time evidence; every field is independent of the reviewed config."""

    executable_path: str | None = None
    executable_sha256: str | None = None
    reported_version: str | None = None
    launch_command_executable: str | None = None
    version_probe_command: tuple[str, ...] = ()
    version_probe_returncode: int | None = None
    version_probe_error: str | None = None
    process_pid: int | None = None
    process_executable_path: str | None = None
    process_identity_source: str = "unavailable"
    process_identity_verified: bool = False
    evidence_source: str = "not_collected"
    captured_at_utc: str = ""

    @property
    def version_output_available(self) -> bool:
        return self.reported_version is not None

    @property
    def verified_executable_version_pair(self) -> bool:
        """True only when the probe ran the same hashed file used in the launch command."""
        return bool(
            self.executable_path
            and self.executable_sha256
            and self.reported_version
            and self.version_probe_returncode == 0
            and self.version_probe_error is None
            and _same_path(self.executable_path, self.launch_command_executable)
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "executable_path": self.executable_path,
            "executable_sha256": self.executable_sha256,
            "reported_version": self.reported_version,
            "launch_command_executable": self.launch_command_executable,
            "version_probe_command": list(self.version_probe_command),
            "version_probe_returncode": self.version_probe_returncode,
            "version_probe_error": self.version_probe_error,
            "process_pid": self.process_pid,
            "process_executable_path": self.process_executable_path,
            "process_identity_source": self.process_identity_source,
            "process_identity_verified": self.process_identity_verified,
            "verified_executable_version_pair": self.verified_executable_version_pair,
            "evidence_source": self.evidence_source,
            "captured_at_utc": self.captured_at_utc,
        }


def _process_identity_verified(
    executable_path: str, observed_path: str | None, pid: int | None
) -> bool:
    if not executable_path or not isinstance(pid, int) or pid <= 0:
        return False
    if observed_path is None:
        # Popen still ties the PID to the exact command it spawned; the provenance field records
        # that no OS-level executable path was available on this host.
        return True
    if _same_path(executable_path, observed_path):
        return True
    interpreter = _shebang_interpreter(executable_path)
    return bool(interpreter and _same_path(interpreter, observed_path))


def attach_process_identity(
    attestation: WebotsRuntimeAttestation, process: Any
) -> WebotsRuntimeAttestation:
    """Bind an attested executable to the PID actually spawned by the launch context."""
    observed_path, source = observed_process_executable(process)
    if source == "unavailable":
        source = "popen_pid_and_command"
    pid = getattr(process, "pid", None)
    verified = _process_identity_verified(
        attestation.executable_path or "", observed_path, pid if isinstance(pid, int) else None
    )
    return replace(
        attestation,
        process_pid=pid if isinstance(pid, int) else None,
        process_executable_path=observed_path,
        process_identity_source=source,
        process_identity_verified=verified,
    )


def attest_webots_executable(
    executable: str | Path,
    env: Mapping[str, str],
    *,
    runner: _ProbeRunner = subprocess.run,
    timeout_s: float = VERSION_PROBE_TIMEOUT_S,
    process: Any = None,
) -> WebotsRuntimeAttestation:
    """Run ``<resolved executable> --version`` and hash that same file.

    The probe is a short-lived version query, not a simulation launch. ``runner`` is injectable so
    all tests remain inert and no Webots process is required.
    """
    resolved = str(Path(executable).expanduser().resolve())
    digest = sha256_file(resolved)
    command = [resolved, VERSION_PROBE_FLAG]
    reported: str | None = None
    returncode: int | None = None
    error: str | None = None
    try:
        completed = runner(
            command,
            env=dict(env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        returncode = int(completed.returncode)
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        reported = normalize_webots_version(output)
        if reported is None:
            error = "version probe output contained no Webots version token"
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    attestation = WebotsRuntimeAttestation(
        executable_path=resolved,
        executable_sha256=digest,
        reported_version=reported,
        launch_command_executable=resolved,
        version_probe_command=tuple(command),
        version_probe_returncode=returncode,
        version_probe_error=error,
        evidence_source="launch_time_executable_version_probe",
        captured_at_utc=datetime.now(UTC).isoformat(),
    )
    return attach_process_identity(attestation, process) if process is not None else attestation
