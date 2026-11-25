"""Forced SSH command: permit only the fixed golden-input read and artifact receiver."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def approved_command(original: str, config: dict[str, Any]) -> list[str]:
    command = shlex.split(original)
    prefix = [config["python"], config["helper"]]
    permitted = (
        [*prefix, "receive", "--root", config["receiver_root"]],
        [*prefix, "emit", "--file", config["golden_archive"]],
    )
    if command not in permitted:
        raise ValueError("SSH command denied: only fixed receive/golden-input operations permitted")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = approved_command(os.environ.get("SSH_ORIGINAL_COMMAND", ""), config)
    # Inherit the encrypted SSH stream. Never evaluate a caller-supplied shell command.
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
