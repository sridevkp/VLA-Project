"""Small three-worker, matched-reset TRAIN collection batch."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.skill_diagnostic import run_skill_diagnostic
from vlarobot.execution.skills import Skill


def collect_approach_pairs(out: Path, webots: str | None, seed: int) -> dict:
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
        pairs = list(workers.map(pair, range(3)))
    elapsed = perf_counter() - started
    executed = sum(result["executed"] for pair in pairs for result in pair["results"])
    report = {
        "split": "train",
        "seed": seed,
        "attempts": 6,
        "executed": executed,
        "wall_s": elapsed,
        "executed_per_minute": 60 * executed / elapsed,
        "pairs": pairs,
    }
    write_json(out / "collection-summary.json", report)
    return report
