"""Render the frozen development figures from recorded VALIDATION artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vlarobot.core.wire import decode_message

ARMS = ("baseline", "arm3", "arm4-equal", "arm5-equal")
FONT = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")


def render(source: Path, run: Path, output: Path) -> None:
    """Write vector plots, recorded RGB composites and an input-SHA ledger."""
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    def read(path: Path) -> bytes:
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def packet(path: Path) -> dict[str, Any]:
        return decode_message(read(path))["payload"]

    def gnuplot(name: str, commands: str) -> None:
        prefix = """
set encoding utf8
set border 3 lc rgb '#4B5563'
set tics nomirror
set key opaque
set grid ytics lc rgb '#E5E7EB'
"""
        for extension, terminal in (
            ("pdf", "pdfcairo enhanced color font 'DejaVu Sans,10' size 8in,4.7in"),
            ("png", "pngcairo enhanced size 1600,940 font 'DejaVu Sans,19'"),
        ):
            command = f"set terminal {terminal}\nset output '{name}.{extension}'\n"
            subprocess.run(
                ["gnuplot"], cwd=output, input=command + prefix + commands, text=True, check=True
            )

    def block(name: str, rows: list[tuple]) -> str:
        return f"${name} << EOD\n" + "\n".join(" ".join(map(str, r)) for r in rows) + "\nEOD\n"

    gnuplot(
        "architecture",
        """
unset border
unset grid
unset tics
unset key
set xrange [0:10]
set yrange [0:7]
set title 'Evaluated scripted-primary WorldMind | simulation data flow'
set style rectangle fillcolor rgb '#E8F1F8' fs solid border lc rgb '#0072B2'
set object 1 rect from 0.7,5.25 to 3.7,6.25
set object 2 rect from 6.2,5.25 to 9.3,6.25
set object 3 rect from 6.2,3.6 to 9.3,4.6
set object 4 rect from 6.2,1.95 to 9.3,2.95
set object 5 rect from 0.7,1.95 to 3.7,2.95
set object 6 rect from 0.7,0.25 to 3.7,1.25
set label 1 'Webots R2025a / TIAGo' at 2.2,5.94 center
set label 2 'Guarded apply and tick' at 2.2,5.60 center
set label 3 'Oracle task state' at 7.75,5.94 center
set label 4 'Scripted 32 × 50 pool' at 7.75,5.60 center
set label 5 'Analytic executable prefix' at 7.75,4.28 center
set label 6 'Arm 5 hard veto then ranking' at 7.75,3.94 center
set label 7 'Independent motor check' at 7.75,2.63 center
set label 8 'Evidence and selected action' at 7.75,2.29 center
set label 9 'Phase and waypoint feedback' at 2.2,2.63 center
set label 10 'HOME before final verification' at 2.2,2.29 center
set label 11 'Observed placement verdict' at 2.2,0.93 center
set label 12 'Complete / retry once / abort' at 2.2,0.59 center
set arrow 1 from 3.7,5.75 to 6.2,5.75 heads lw 2
set arrow 2 from 7.75,5.25 to 7.75,4.6 head lw 2
set arrow 3 from 7.75,3.6 to 7.75,2.95 head lw 2
set arrow 4 from 6.2,2.45 to 3.7,2.45 head lw 2
set arrow 5 from 2.2,2.95 to 2.2,5.25 head lw 2
set arrow 6 from 2.2,1.95 to 2.2,1.25 head lw 2
set label 13 'Observations / receipts' at 4.95,6.15 center
set label 14 'ZeroMQ protocol v3' at 4.95,5.37 center
plot 1/0
""",
    )

    evidence = json.loads(read(source))
    outcomes = evidence["family_and_wilson"]
    coverage = evidence["campaign_summary_original"]
    repair = json.loads(read(source.with_name("arm3-repair-metrics.json")))
    original = json.loads(read(source.parent.parent / "nov24-matched-analysis/completed.json"))

    # Fig. 4.1/4.3/4.4. The source arrays are the Webots RGB camera observations.
    reset = packet(run / "reset-observation.msgpack")["observation"]
    Image.fromarray(reset.rgb).save(output / "webots-reset.png")
    frames = [("Reset", reset)]
    desired = iter(("PREGRASP", "CLOSE", "LIFT", "TRANSIT", "RELEASE", "HOME"))
    phase = next(desired)
    decisions = []
    for path in sorted(run.glob("chunk-*/decision.msgpack")):
        tick = path.parent / "tick-00" / "post-tick.msgpack"
        if not tick.exists():
            continue
        if phase is not None:
            data = decode_message(path.read_bytes())["payload"]
            if data["phase"] == phase:
                read(path)
                frames.append((phase, packet(tick)["observation"]))
                phase = next(desired, None)
        prediction = path.with_name("selected-prediction.msgpack")
        if prediction.exists():
            count = len(list(path.parent.glob("tick-*/post-tick.msgpack")))
            decisions.append((count, path))
    assert phase is None
    montage = Image.new("RGB", (640 * 4, 530 * 2), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.truetype(str(FONT), 24) if FONT.is_file() else ImageFont.load_default(size=24)
    for index, (label, observation) in enumerate(frames):
        x, y = index % 4 * 640, index // 4 * 530
        montage.paste(Image.fromarray(observation.rgb), (x, y))
        draw.text(
            (x + 10, y + 484), f"{label}   t={observation.t:.3f} s", fill="#202631", font=font
        )
    montage.save(output / "webots-execution.png")

    # Fig. 4.5. An oracle rendering using the frozen verified box coordinate binding.
    state = packet(run / "observed-state.msgpack")
    oracle = state["oracle"]
    robot = state["state"].robot
    poses = {name: pose.position.as_array() for name, pose in oracle.object_poses.items()}
    poses["BASE"] = robot.base_pose_world.position.as_array()
    lo = np.min(list(poses.values()), axis=0) - 0.13
    hi = np.max(list(poses.values()), axis=0) + 0.13
    rows = [(p[0], p[1], f"'{name}'") for name, p in poses.items()]
    gnuplot(
        "oracle-state",
        block("P", rows)
        + f"""set title 'Recorded reset oracle positions'
set xrange [{lo[0]}:{hi[0]}]
set yrange [{lo[1]}:{hi[1]}]
set xlabel 'World x (m)'
set ylabel 'World y (m)'
set size ratio -1
unset key
plot $P using 1:2 with points pt 7 ps 1.6 lc rgb '#0072B2', \
     $P using 1:2:3 with labels offset 1,1
""",
    )

    # Fig. 4.2. Choose the longest fully observed selected prefix, from the recorded pool.
    for n, path in sorted(decisions, key=lambda item: (-item[0], item[1])):
        prediction_path = path.with_name("selected-prediction.msgpack")
        prediction = packet(prediction_path)
        if len(prediction["states"]) == n + 1 and n >= 2:
            break
    else:
        raise ValueError("no complete recorded prediction prefix of at least two ticks")
    decision = packet(path)
    assert prediction["decision_sha256"] == hashes[str(path)]
    assert prediction["candidate_id"] == decision["selected"]
    observations = [decision["observation"]] + [
        packet(t)["observation"] for t in sorted(path.parent.glob("tick-*/post-tick.msgpack"))
    ]
    initial_t = observations[0].t
    samples = []
    for i, (pred, obs) in enumerate(zip(prediction["states"], observations, strict=True)):
        assert abs(obs.t - initial_t - i * prediction["dt_s"]) < 1e-8
        coords = pred.robot.ee_pose.position.as_array()
        samples.append((obs.t - initial_t, *coords, *obs.robot.ee_pose.position.as_array()))
    title = f"{path.parent.name} / {decision['phase']} / {prediction['candidate_id']}"
    script = block("P", samples) + f"set multiplot layout 3,1 title '{title}'\n"
    for axis, column in zip("xyz", (2, 3, 4), strict=True):
        script += f"""
set ylabel 'TCP {axis} (m)'
set xlabel '{"Prefix time (s)" if axis == "z" else ""}'
plot $P using 1:{column} with lines lw 2 lc rgb '#0072B2' title 'Predicted', \
     $P using 1:{column + 3} with lines dt 2 lw 2 lc rgb '#D55E00' title 'Observed'
"""
    gnuplot("prediction-prefix", script + "unset multiplot\n")

    # Fig. 4.6/4.7. Use the first mixed arm-5 decision in this recorded success.
    for _, path in decisions:
        d = decode_message(path.read_bytes())["payload"]
        rows = d["proxy"]["assessments"]
        if d["selected"] is not None and 0 < sum(row["admitted"] for row in rows) < len(rows):
            read(path)
            break
    else:
        raise ValueError("recorded episode lacks a selected pool with at least one veto")
    points = []
    for i, r in enumerate(rows):
        score = sum(float(x) for x in r["components"][:3] if x is not None)
        status = 2 if r["candidate_id"] == d["selected"] else int(r["admitted"])
        points.append((i, score, status))
    gnuplot(
        "candidate-pool",
        block("D", points)
        + f"""set title '{d["phase"]} / {path.parent.name} / K=32'
set xlabel 'Frozen candidate draw index'
set ylabel 'Applicable component sum'
set key outside top center horizontal
plot for [i=0:2] $D using 1:($3==i?$2:1/0) with points pt 7 ps 1.2 \
    lc rgb word('#D55E00 #0072B2 #009E73',i+1) title word('Vetoed Admitted Selected',i+1)
""",
    )
    candidate_rows = [
        {
            "candidate_id": row["candidate_id"],
            "checks": row["checks"],
            "signed_margins": row["signed_margins"],
            "disposition": (
                "selected"
                if row["candidate_id"] == d["selected"]
                else "admitted"
                if row["admitted"]
                else "vetoed"
            ),
        }
        for row in rows
    ]
    (output / "candidate-pool.json").write_text(
        json.dumps(candidate_rows, indent=2) + "\n", encoding="utf-8"
    )

    family_rows = []
    for i, family in enumerate(("clear", "cluttered", "reach-limited"), 1):
        for j, arm in enumerate(ARMS):
            row = outcomes[arm][family]
            family_rows.append(
                (
                    i + (j - 1.5) * 0.18,
                    row["successes"] / row["n"],
                    *row["wilson_95"],
                    j,
                )
            )
    gnuplot(
        "family-success",
        block("F", family_rows)
        + """set title 'Task success by scene family | VALIDATION'
set ylabel 'Success fraction with 95% Wilson intervals'
set yrange [0:1.1]
set xrange [0.55:3.45]
set xtics ('Clear' 1,'Cluttered' 2,'Reach-limited' 3)
set key outside top center horizontal
plot for [i=0:3] $F using 1:($5==i?$2:1/0):3:4 with yerrorbars pt 7 lw 2 \
     lc rgb word('#374151 #0072B2 #009E73 #D55E00',i+1) title word('A1 A3 A4 A5',i+1)
""",
    )

    errors = evidence["horizon_errors"]["arm5-equal"]
    rows = [
        (
            index,
            errors[family]["1"]["object_m"]["rmse"] * 1000,
            errors[family]["1"]["object_m"]["mae"] * 1000,
        )
        for index, family in enumerate(("clear", "cluttered", "reach-limited"), 1)
    ]
    gnuplot(
        "object-prediction-error",
        block("O", rows)
        + """
set title 'Object position error at tick 1 | Arm 5 equal'
set ylabel 'Position error (mm)'
set xrange [0.5:3.5]
set yrange [0:80]
set xtics ('Clear' 1,'Cluttered' 2,'Reach-limited' 3)
set key outside top center horizontal
plot $O using 1:2 with points pt 7 ps 1.5 lc rgb '#0072B2' title 'RMSE', \
     $O using 1:3 with points pt 5 ps 1.5 lc rgb '#D55E00' title 'MAE'
""",
    )

    ordered = sorted(evidence["episodes"]["baseline"])
    paired = [
        (i, j, int(evidence["episodes"][arm][scene]["success"]))
        for i, scene in enumerate(ordered, 1)
        for j, arm in enumerate(ARMS, 1)
    ]
    ticks = ",".join(f"'{scene[11:]}' {i}" for i, scene in enumerate(ordered, 1))
    gnuplot(
        "paired-outcomes",
        block("P", paired)
        + f"""set title 'Matched VALIDATION identities'
set xrange [0:19]
set yrange [0.5:4.5]
set xtics rotate by -55 ({ticks})
set ytics ('Arm 1' 1,'Arm 3' 2,'Arm 4' 3,'Arm 5' 4)
set bmargin 10
set key outside top center horizontal
plot $P using 1:($3==1?$2:1/0) with points pt 7 ps 1.7 lc rgb '#0072B2' title 'Success', \
     $P using 1:($3==0?$2:1/0) with points pt 2 ps 1.7 lw 2 lc rgb '#D55E00' title 'Task failure'
""",
    )

    latency = []
    for j, arm in enumerate(ARMS[1:], 1):
        values = [
            value
            for row in original["episodes"][arm].values()
            for value in row.get("extra", {}).get("planning_ms", [])
        ]
        if arm == "arm3":
            values.extend(repair["planning_latency_ms"])
        p50, p95 = np.quantile(values, [0.5, 0.95], method="inverted_cdf")
        latency.append((j, p50, p95))
    gnuplot(
        "latency-and-veto",
        block("L", latency)
        + block(
            "V",
            [
                (1, 0),
                (2, 100 * coverage["arm5-equal"]["risk"]["value"]),
                (3, 100 * coverage["arm5-goal-x2"]["risk"]["value"]),
            ],
        )
        + """set multiplot layout 1,2
set key top left
set title 'Decision latency'
set ylabel 'Time (ms, log scale)'
set xrange [0.5:3.5]
set xtics ('Arm 3' 1,'Arm 4' 2,'Arm 5' 3)
set logscale y
plot $L using 1:2 with points pt 7 ps 1.5 lc rgb '#0072B2' title 'p50', \
     $L using 1:3 with points pt 5 ps 1.5 lc rgb '#D55E00' title 'p95'
unset logscale y
unset key
set title 'Candidate hard-veto rate'
set ylabel 'Rejected / evaluated (%)'
set xtics ('Arm 3' 1,'Arm 5 equal' 2,'Arm 5 goal' 3)
set yrange [-0.002:0.061]
set boxwidth 0.5
set style fill solid 0.85
plot $V using 1:2 with boxes lc rgb '#0072B2', $V using 1:2 with points pt 7 lc rgb '#0072B2'
unset multiplot
""",
    )
    costs = [
        (
            i,
            evidence["execution"][arm]["successful"]["path_m"],
            evidence["execution"][arm]["successful"]["simulation_s"],
            1000 * evidence["execution"][arm]["successful"]["placement_error_m"],
        )
        for i, arm in enumerate(ARMS, 1)
    ]
    script = block("E", costs) + "set multiplot layout 1,3 title 'Successful VALIDATION episodes'\n"
    for index, label in (
        (2, "TCP path (m)"),
        (3, "Simulated time (s)"),
        (4, "Final CAN error (mm)"),
    ):
        script += f"""
set title '{label}'
set xrange [0.5:4.5]
set yrange [0:*]
set xtics ('1' 1,'3' 2,'4' 3,'5' 4)
set xlabel 'Arm'
set boxwidth 0.65
set style fill solid 0.85
unset key
plot $E using 1:{index} with boxes lc rgb '#0072B2'
"""
    gnuplot("successful-episode-costs", script + "unset multiplot\n")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "scope": "VALIDATION development evidence",
                "sources_sha256": hashes,
                "prediction_prefix": title,
                "candidate_pool_source": str(path),
                "camera_frames": [
                    {"phase": label, "simulation_s": frame.t} for label, frame in frames
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.analysis, args.episode, args.output)
