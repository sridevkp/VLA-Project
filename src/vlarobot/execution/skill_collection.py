"""Small three-worker, matched-reset TRAIN collection batch."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.skill_diagnostic import run_skill_diagnostic
from vlarobot.execution.skills import V4_REGISTRATION_RECORD, Skill, v4_simulator_environment


def collect_approach_pairs(
    out: Path, webots: str | None, seed: int, *, internal: bool = False
) -> dict:
    """Measure a small matched batch before expanding the experimental budget."""
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()

    def pair(index: int) -> dict:
        results = []
        for offset in (False, True):
            name = "offset" if offset else "center"
            folder = out / f"scene-{index}-{name}"
            try:
                result = run_skill_diagnostic(
                    folder,
                    webots,
                    Skill.APPROACH,
                    imperfect_approach=offset,
                    scene_index=index,
                    proxy=True,
                    seed=seed,
                )
                results.append(
                    {
                        "path": str(folder / "skill-result.json"),
                        "executed": bool(result["rows"][0]["commands"]),
                        "settled": result["settled"],
                    }
                )
            except Exception as exc:
                results.append({"path": str(folder), "error": str(exc), "executed": False})
        return {"scene_index": index, "results": results}

    with ThreadPoolExecutor(max_workers=3) as workers:
        pairs = list(workers.map(pair, (6, 7, 5) if internal else (0, 1, 2)))
    elapsed = perf_counter() - started
    executed = sum(result["executed"] for pair in pairs for result in pair["results"])
    report = {
        "split": "train",
        **V4_REGISTRATION_RECORD,
        "seed": seed,
        "cohort": "internal-train-evaluation" if internal else "fit",
        "attempts": 6,
        "executed": executed,
        "wall_s": elapsed,
        "executed_per_minute": 60 * executed / elapsed,
        "pairs": pairs,
    }
    write_json(out / "collection-summary.json", report)
    return report


def collect_outcomes(out: Path, webots: str | None, seed: int, target: int) -> dict:
    """Collect varied eligible outcomes with one persistent simulator per worker."""
    from shutil import disk_usage

    from vlarobot.execution.episode_runner import launch_webots_session

    if not 3 <= target <= 300:
        raise ValueError("Bounded outcome target must be 3–300")
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    plans = (
        ("center", Skill.APPROACH, {}),
        ("offset", Skill.APPROACH, {"imperfect_approach": True}),
        ("standoff", Skill.APPROACH, {"standoff": True}),
        ("empty-recovery", Skill.OPEN, {"empty_grasp": True}),
        ("manipulation", Skill.RETREAT, {}),
    )

    def worker(worker_id: int) -> dict:
        folder = out / f"worker-{worker_id}"
        rows = []
        eligible = executed = attempted = 0
        goal = target // 3 + int(worker_id < target % 3)
        with launch_webots_session(
            webots=webots,
            trace_dir=folder,
            headless=True,
            extra_env=v4_simulator_environment(),
        ) as session:
            for cycle in range(12):
                index = (worker_id, (6, 7, 5)[worker_id])[cycle % 2]
                for name, skill, options in plans:
                    if eligible >= goal or disk_usage(out).free < 20 * 1024**3:
                        break
                    run = folder / f"{cycle:02d}-{name}"
                    try:
                        summary = run_skill_diagnostic(
                            run,
                            webots,
                            skill,
                            scene_index=index,
                            proxy=True,
                            seed=seed + cycle,
                            session=session,
                            **options,
                        )
                    except Exception as exc:
                        rows.append({"run": str(run), "error": str(exc)})
                        write_json(folder / "progress.json", {"runs": rows, "eligible": eligible})
                        continue
                    outcomes = summary["rows"]
                    executed += sum(bool(row["commands"]) for row in outcomes)
                    attempted += len(outcomes)
                    eligible += sum(row["settled"] and row["isolated"] for row in outcomes)
                    rows.append(
                        {
                            "run": str(run),
                            "seed": seed + cycle,
                            "scene_index": index,
                            "recipe": name,
                            "attempted": len(outcomes),
                            "eligible": sum(row["settled"] and row["isolated"] for row in outcomes),
                        }
                    )
                    write_json(
                        folder / "progress.json",
                        {
                            "runs": rows,
                            "eligible": eligible,
                            "executed": executed,
                            "attempted": attempted,
                        },
                    )
                if eligible >= goal or disk_usage(out).free < 20 * 1024**3:
                    break
        return {
            "worker": worker_id,
            "runs": rows,
            "eligible": eligible,
            "executed": executed,
            "attempted": attempted,
        }

    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(worker, range(3)))
    elapsed = perf_counter() - started
    report = {
        "split": "train",
        **V4_REGISTRATION_RECORD,
        "target": target,
        "wall_s": elapsed,
        "workers": results,
        **{
            key: sum(result[key] for result in results)
            for key in ("attempted", "executed", "eligible")
        },
    }
    write_json(out / "collection-summary.json", report)
    return report


def outcome_snapshot(out: Path) -> dict:
    """Summarize completed simulator evidence while collection is running."""
    import json
    from collections import defaultdict

    import numpy as np

    from vlarobot.execution.skill_empirical import measured_utility

    reports = sorted(out.glob("worker-*/*/skill-result.json"))
    pairs = defaultdict(dict)
    attempted = executed = eligible = labeled = 0
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["split"] != "train":
            raise ValueError("outcome batch requires TRAIN provenance")
        rows = report["rows"]
        attempted += len(rows)
        executed += sum(bool(row["commands"]) for row in rows)
        eligible += sum(row["settled"] and row["isolated"] for row in rows)
        labeled += sum(measured_utility(row) is not None for row in rows)
        recipe = path.parent.name.split("-", 1)[1]
        if recipe in {"center", "offset", "standoff"} and measured_utility(rows[0]) is not None:
            pairs[report["scene"]][recipe] = rows[0]

    def reset_vector(row: dict) -> list[float]:
        robot = row["robot_before"]
        pose = robot["base_pose_world"]
        return [
            *(robot["joint_pos"][key] for key in sorted(robot["joint_pos"])),
            *pose["position"].values(),
            *pose["orientation"],
            *row["can_before_world_m"].values(),
            robot["gripper_opening"],
        ]

    matched = 0
    for by_recipe in pairs.values():
        center = by_recipe.get("center")
        if center is None:
            continue
        variants = sum(
            np.allclose(reset_vector(center), reset_vector(row), rtol=0, atol=1e-6)
            for recipe, row in by_recipe.items()
            if recipe != "center"
        )
        matched += 1 + variants if variants else 0
    return {
        "attempted": attempted,
        "executed": executed,
        "eligible": eligible,
        "utility_labeled": labeled,
        "strictly_matched_outcomes": matched,
        "reports": [str(path) for path in reports],
    }


def update_outcome_models(
    batch: Path,
    dataset: Path,
    out: Path,
    prior: list[Path],
    *,
    watch: bool = False,
) -> dict:
    """Freeze incremental fit snapshots while keeping the collected evidence read-only."""
    from time import sleep

    from vlarobot.execution.skill_empirical import train_empirical

    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    last_count = -12
    while True:
        snapshot = outcome_snapshot(batch)
        labeled = snapshot["utility_labeled"]
        complete = (batch / "collection-summary.json").exists()
        if labeled and (labeled >= last_count + 12 or complete or not watch):
            model = out / f"empirical-{labeled:03d}.json"
            if not model.exists():
                report = train_empirical(
                    dataset, model, prior + [Path(p) for p in snapshot["reports"]]
                )
                snapshot["measured_fit_n"] = report["measured_fit_n"]
                snapshot["model"] = str(model)
                write_json(out / "latest-fit.json", snapshot)
            last_count = labeled
        if not watch or complete or perf_counter() - started > 3 * 3600:
            return snapshot
        sleep(30)
