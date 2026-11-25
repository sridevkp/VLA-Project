"""Read-only incremental integrity monitoring of admitted TRAIN recordings."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vlarobot.eval.demo_audit import audit_demo
from vlarobot.eval.demos import build_demo_plan
from vlarobot.execution.demo_collection import audit_spec, write_json


def inspect_collection(collection: Path, cache: Path) -> dict[str, Any]:
    """Re-audit new completed attempts and hash their immutable raw streams and every JPEG."""
    identity = json.loads((collection / "provenance.json").read_text(encoding="utf-8"))
    accepted = json.loads((collection / "accepted.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (collection / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    seen = [r["attempt"] for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError("duplicated attempts in collection manifest")
    plan = build_demo_plan(
        len(identity["attempt_plan"]),
        base_seed=identity["seed_start"],
        instruction=identity["instruction"],
    )
    expected = json.loads(json.dumps([asdict(e) for e in plan]))
    if identity["attempt_plan"] != expected:
        raise ValueError("collection plan no longer matches frozen TRAIN generator")
    previous = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    verified = previous.get("verified", {})
    frames = 0
    for row in accepted["episodes"]:
        index = row["attempt"]
        entry = expected[index]
        if row["seed"] != entry["seed"] or row["scene"] != entry["scene"]:
            raise ValueError(f"admitted attempt {index} differs from its TRAIN plan")
        raw = Path(row["raw_dir"])
        files = [
            raw / f"episode-00.{s}"
            for s in ["meta.json", "steps.npz", "actions.jsonl", "physical.jsonl"]
        ]
        fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        key = str(index)
        if key in verified and fingerprint != verified[key]["stream_sha256"]:
            raise ValueError(f"immutable admitted stream changed: attempt {index}")
        if key not in verified:
            audit = audit_demo(raw / "episode-00.meta.json", audit_spec(1))
            if not audit["accepted"]:
                raise ValueError(f"admitted attempt {index} failed re-audit: {audit['reasons']}")
            meta = json.loads((raw / "episode-00.meta.json").read_text(encoding="utf-8"))
            with np.load(raw / "episode-00.steps.npz", allow_pickle=False) as samples:
                if any(
                    not np.all(np.isfinite(samples[k]))
                    for k in ["joint_pos", "gripper", "ee_xyz", "ee_quat"]
                ):
                    raise ValueError(f"nonfinite robot recording: attempt {index}")
            digest = hashlib.sha256()
            count = meta["frame_count"]
            paths = sorted((raw / "frames/episode-00").glob("*.jpg"))
            if len(paths) != count:
                raise ValueError(f"RGB count mismatch: attempt {index}")
            for i, path in enumerate(paths):
                if path.name != f"{i:06d}.jpg":
                    raise ValueError(f"RGB sequence gap: {path}")
                with Image.open(path) as picture:
                    if picture.mode != "RGB" or picture.size != (640, 480):
                        raise ValueError(f"RGB schema mismatch: {path}")
                    picture.load()
                content = path.read_bytes()
                digest.update(len(content).to_bytes(8, "little"))
                digest.update(content)
            verified[key] = {
                "frames": count,
                "stream_sha256": fingerprint,
                "jpeg_bytes_sha256": digest.hexdigest(),
                "audit": audit,
            }
        frames += verified[key]["frames"]
    free = shutil.disk_usage(collection).free
    result = {
        "admitted": accepted["count"],
        "attempted": len(rows),
        "admission_rate": accepted["count"] / len(rows) if rows else None,
        "quarantined": sum(r["status"] != "accepted" for r in rows),
        "verified_frames": frames,
        "disk_free_bytes": free,
        "verified": verified,
        "training_approved": False,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache, result)
    return result
