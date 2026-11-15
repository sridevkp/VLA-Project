"""Restricted receiver key cannot run arbitrary commands or change artifact paths."""

import importlib.util
import shlex
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "receiver_dispatch", Path(__file__).parents[1] / "tools/recovery/receiver_dispatch.py"
)
assert spec and spec.loader
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)
CONFIG = {
    "python": "/python",
    "helper": "/helper with space.py",
    "receiver_root": "/independent/received",
    "golden_archive": "/backup/017000.zip",
}


@pytest.mark.parametrize(
    "mode,path_flag,key",
    (
        ("receive", "--root", "receiver_root"),
        ("emit", "--file", "golden_archive"),
    ),
)
def test_exact_operations_allowed(mode, path_flag, key):
    args = [CONFIG["python"], CONFIG["helper"], mode, path_flag, CONFIG[key]]
    assert dispatch.approved_command(shlex.join(args), CONFIG) == args


@pytest.mark.parametrize(
    "command",
    (
        "",
        "bash",
        "cat /etc/passwd",
        "scp -t /tmp/file",
        "sftp",
        "/python '/helper with space.py' receive --root /different",
        "/python '/helper with space.py' emit --file /different",
        "/python '/helper with space.py' receive --root /independent/received; id",
    ),
)
def test_shell_subsystems_and_path_changes_denied(command):
    with pytest.raises(ValueError, match="denied"):
        dispatch.approved_command(command, CONFIG)
