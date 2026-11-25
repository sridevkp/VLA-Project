"""Play back the original RGB observations of a completed development episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from vlarobot.core.wire import decode_message


def render(episode: Path, output: Path) -> None:
    """Export the native-rate recorded camera stream without running a simulator."""
    if output.exists():
        raise FileExistsError(output)
    sources = [
        episode / "reset-observation.msgpack",
        *episode.glob("chunk-*/tick-*/post-tick.msgpack"),
        *episode.glob("empty-set-reobserve-*/post-tick.msgpack"),
    ]
    if len(sources) < 2:
        raise ValueError("a recorded reset and at least one observed tick are required")
    # Re-observation ticks share the matching decision index and precede that decision.
    sources = sources[:1] + sorted(
        sources[1:],
        key=lambda path: (
            int(path.relative_to(episode).parts[0].rsplit("-", 1)[-1]),
            0 if path.parent.name.startswith("empty-set-reobserve-") else 1,
            path.parent.name,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    previous = None
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "640x480",
        "-framerate",
        "125/4",
        "-i",
        "pipe:0",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        try:
            for path in sources:
                raw = path.read_bytes()
                digest.update(hashlib.sha256(raw).digest())
                obs = decode_message(raw)["payload"]["observation"]
                if not math.isfinite(obs.t) or (
                    previous is not None and not math.isclose(obs.t - previous, 0.032, abs_tol=1e-8)
                ):
                    raise ValueError(f"non-contiguous recorded timeline: {path}")
                if obs.rgb.shape != (480, 640, 3) or obs.rgb.dtype.name != "uint8":
                    raise ValueError(f"unexpected recorded RGB geometry: {path}")
                encoder.stdin.write(obs.rgb.tobytes())
                previous = obs.t
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg did not produce a complete video")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "scope": "recorded VALIDATION playback",
                "episode": str(episode),
                "frames": len(sources),
                "frames_per_second": 31.25,
                "ordered_source_sha256_of_sha256s": digest.hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.episode, args.output)
