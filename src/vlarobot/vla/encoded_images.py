"""Preserve recorded JPEG bytes inside native image features without another image encode."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

from vlarobot.vla.schema import CAMERA_KEY


def embed_recorded_jpegs(dataset: Path, raw_episodes: list[Path]) -> dict[str, Any]:
    """Replace temporary PNG encodings with original JPEGs; decoded source pixels stay identical.

    LeRobot computes statistics from those same JPEG-decoded pixels before this operation.
    Row order, schema, timestamps, labels, statistics and acquisition cadence are unchanged.
    """
    arrow = import_module("pyarrow")
    parquet = import_module("pyarrow.parquet")
    before = after = frames = 0
    digest = hashlib.sha256()
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = parquet.read_table(path)
        encoded = []
        for episode, frame in zip(
            table["episode_index"].to_pylist(), table["frame_index"].to_pylist(), strict=True
        ):
            source = raw_episodes[episode] / "frames/episode-00" / f"{frame:06d}.jpg"
            content = source.read_bytes()
            if not content.startswith(b"\xff\xd8"):
                raise ValueError(f"recorded image is not JPEG: {source}")
            digest.update(len(content).to_bytes(8, "little"))
            digest.update(content)
            encoded.append({"bytes": content, "path": f"frame-{frame:06d}.jpg"})
        index = table.schema.get_field_index(CAMERA_KEY)
        field = table.schema.field(index)
        table = table.set_column(index, field, arrow.array(encoded, type=field.type))
        temporary = path.with_suffix(".tmp")
        parquet.write_table(table, temporary, compression="zstd")
        before += path.stat().st_size
        after += temporary.stat().st_size
        frames += len(encoded)
        temporary.replace(path)
    return {
        "encoding": "original recorded JPEG bytes embedded in native image features",
        "frames": frames,
        "source_jpeg_bytes_sha256": digest.hexdigest(),
        "parquet_bytes_before": before,
        "parquet_bytes_after": after,
        "image_reencoding_performed": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
