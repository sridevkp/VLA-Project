"""Fail-closed verification of the historical comparison-test commit."""

from __future__ import annotations

import subprocess
import sys

EXPECTED = "1b2e4da0cd14a825b6e9c52cbf9e217da7f3bf75"


def main() -> int:
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{EXPECTED}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if resolved.returncode or resolved.stdout.strip() != EXPECTED:
        print(f"Required historical commit unavailable or mismatched: {EXPECTED}", file=sys.stderr)
        return 2
    object_check = subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTED}^{{commit}}"],
        check=False,
    )
    if object_check.returncode:
        print(f"Required commit object is missing: {EXPECTED}", file=sys.stderr)
        return 3
    print(f"verified historical commit {EXPECTED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
