"""Require collected safe-CI nodes to match the reviewed allowlist exactly."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/evidence/safe-ci-test-inventory-v1.json"


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    command = inventory["selection_command"]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        return result.returncode
    actual = sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    expected = sorted(inventory["safe_allowlist_nodeids"])
    if not actual:
        print("safe CI selection collected zero tests", file=sys.stderr)
        return 4
    if actual != expected:
        print("safe CI collected nodes differ from reviewed inventory", file=sys.stderr)
        print(f"unexpected={sorted(set(actual) - set(expected))}", file=sys.stderr)
        print(f"missing={sorted(set(expected) - set(actual))}", file=sys.stderr)
        return 5
    print(f"verified safe-CI inventory: {len(actual)} exact node IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
