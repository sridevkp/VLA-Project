"""Read-only supplemental trajectory metrics with an overwrite-safe output path.

This tool never writes into the source episode directory and refuses to overwrite any
existing destination. It does not modify episode files, summaries, CSVs, provenance, or
frozen campaign evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.eval.derived_metrics import derive_supplemental_trajectory_metrics

_FROZEN_OUTPUT_NAMES = {
    "steps.npz",
    "meta.json",
    "summary.json",
    "episodes.csv",
    "provenance.json",
    "plan.json",
    "timings.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_safe_output(out: Path, episode_dir: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {out}")
    if out.name in _FROZEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to use a frozen evidence filename as output: {out.name}")
    try:
        out.relative_to(episode_dir)
    except ValueError:
        return
    raise ValueError(f"refusing to write inside the source episode directory: {episode_dir}")


def _write_exclusive(out: Path, payload: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with out.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        if out.exists():
            out.unlink()
        raise


def _wall_clock_seconds(timings: Path | None, index: int) -> tuple[float | None, str | None, str]:
    if timings is None:
        return None, None, "not_read; pass --timings to include wall-clock evidence separately"
    data = json.loads(timings.read_text(encoding="utf-8"))
    for row in data.get("episodes", []):
        if int(row.get("index", -1)) == index:
            value = row.get("wall_s")
            if value is None:
                return None, f"{timings}:episodes[{index}].wall_s", "timings field is null"
            return (
                float(value),
                f"{timings}:episodes[{index}].wall_s",
                (
                    "wall clock is non-deterministic; resumed episodes may report 0.0; "
                    "never substitute it for simulation_duration_s"
                ),
            )
    return None, None, f"episode index {index} absent from {timings}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ee-frame", required=True, choices=["base"])
    parser.add_argument("--ee-units", required=True, choices=["m"])
    parser.add_argument("--timings", type=Path, default=None)
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    timings = None if args.timings is None else args.timings.resolve()
    out = args.out.resolve()
    _ensure_safe_output(out, episode_dir)

    steps = episode_dir / "steps.npz"
    meta_path = episode_dir / "meta.json"
    if not steps.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"episode requires both steps.npz and meta.json: {episode_dir}")

    with np.load(steps, allow_pickle=False) as arrays:
        missing = {"t", "ee_xyz"} - set(arrays.files)
        if missing:
            raise ValueError(f"steps.npz missing required arrays: {sorted(missing)}")
        metrics = derive_supplemental_trajectory_metrics(
            ee_xyz=arrays["ee_xyz"],
            t=arrays["t"],
            frame=args.ee_frame,
            units=args.ee_units,
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index = int(meta.get("index", meta.get("seed", 0)))
    wall_s, wall_source, wall_note = _wall_clock_seconds(timings, index)

    payload = {
        "source": {
            "episode_dir": str(episode_dir),
            "steps_npz_sha256": _sha256(steps),
            "timings_json_sha256": None if timings is None else _sha256(timings),
        },
        "frame_and_units": {
            "ee_frame": metrics.ee_frame,
            "ee_units": metrics.ee_units,
            "time_units": metrics.time_units,
            "confirmation": (
                "caller explicitly confirmed base-frame metres; NPZ carries no frame metadata"
            ),
        },
        "supplemental_metrics": {
            "end_effector_path_length_m": metrics.end_effector_path_length_m,
            "simulation_duration_s": metrics.simulation_duration_s,
            "sample_count": metrics.sample_count,
        },
        "execution_time_separation": {
            "simulation_duration_s": metrics.simulation_duration_s,
            "wall_clock_execution_time_s": wall_s,
            "wall_clock_source": wall_source,
            "note": wall_note,
        },
    }
    _write_exclusive(out, payload)
    print(f"wrote supplemental metrics to {out}")


if __name__ == "__main__":
    main()
