"""CLI wrapper for reproducible V2 TRAIN closed-loop diagnostic metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlarobot.eval.v2_diagnostics import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="V2 TRAIN diagnostic run directory")
    parser.add_argument("--out", type=Path, default=None, help="output JSONL path")
    args = parser.parse_args()
    rows = summarize(args.run)
    target = args.out or args.run / "derived-motion-summary.jsonl"
    target.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {target}")


if __name__ == "__main__":
    main()
