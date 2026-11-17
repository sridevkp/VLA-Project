"""CPU-only artifact/static-geometry audit. No live diagnostic entry point."""

from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import typer

from vlarobot.diagnostics.ad2.evidence import confined, digest, load_targets, read_json
from vlarobot.diagnostics.ad2.geometry import geometry_inventory

app = typer.Typer()


@app.command()
def audit(
    manifest: Path,
    manifest_sha256: str,
    main_root: Path,
    analytic_root: Path,
    assets_root: Path,
    output: Path,
) -> None:
    """Read TRAIN evidence and cached source only; output must be a new directory."""
    started = perf_counter()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    confined(output.parent, output.name)
    targets = load_targets(
        manifest, manifest_sha256, {"main": main_root, "analytic": analytic_root}
    )
    assets = {}
    for path in sorted(assets_root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            head = stream.read(16384).decode("utf-8", errors="ignore")
        match = re.search(r"^PROTO\s+([A-Za-z_][A-Za-z0-9_-]*)\s*\[", head, re.MULTILINE)
        if match:
            if match[1] in assets:
                raise ValueError(f"Ambiguous cached PROTO: {match[1]}")
            assets[match[1]] = path
    required = {
        "Tiago",
        "TiagoBase",
        "TiagoBody",
        "TiagoFrontArm",
        "TiagoGripper",
        "Astra",
        "HokuyoUrg04lxug01",
        "Table",
        "SolidBox",
        "Can",
        "Apple",
        "Hammer",
    }
    world = main_root / "worlds/factory.wbt"
    # Include all available world EXTERNPROTO sources, not only the three tracked objects.
    required.update(re.findall(r'/([^/"\n]+)\.proto"', world.read_text(encoding="utf-8")))
    inventory = geometry_inventory(
        main_root / "src/vlarobot/backends/webots/tiago.urdf",
        world,
        {name: path for name, path in assets.items() if name in required},
    )
    inventory["missing_world_proto_sources"] = sorted(required - assets.keys())
    source = main_root / "runs/demos/train-s100-registered-v1/attempts/attempt-0000"
    meta_path, npz_path = source / "episode-00.meta.json", source / "episode-00.steps.npz"
    inventory["scene"] = json.loads(meta_path.read_text(encoding="utf-8"))["scenario"]
    with np.load(npz_path, allow_pickle=False) as recorded:
        inventory["recorded_frame0_objects"] = {
            str(name): {
                "world_xyz_m": recorded["gt_xyz"][0, i].tolist(),
                "world_quaternion_xyzw": recorded["gt_quat"][0, i].tolist(),
                "velocity": None,
            }
            for i, name in enumerate(recorded["gt_names"])
        }
    inventory["scene_source_hashes"] = {str(p): digest(p) for p in (meta_path, npz_path)}
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "offline-source-audit-only",
        "live_authorized": False,
        "numeric_protocol": None,
        "targets": asdict(targets),
        "original_provenance_entries": len(read_json(manifest)["provenance_files"]),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "elapsed_s": perf_counter() - started,
    }
    for name, value in [("artifact-loading.json", report), ("static-geometry.json", inventory)]:
        (output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    seals = {p.name: digest(p) for p in output.iterdir()}
    (output / "sha256.json").write_text(json.dumps(seals, indent=2) + "\n", encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "status": report["status"],
                "A_equals_B": targets.a == targets.b,
                "geometry_safety": False,
                "elapsed_s": report["elapsed_s"],
            }
        )
    )


if __name__ == "__main__":
    app()
