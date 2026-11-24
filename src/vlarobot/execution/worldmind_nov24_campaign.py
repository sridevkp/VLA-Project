"""Owned frozen VALIDATION campaigns under the November 24 decision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any

from vlarobot.core.wire import decode_message
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.worldmind_proxy_policy import GRID, OPTIONAL_GRID, select_policy
from vlarobot.execution.episode_runner import launch_webots_session
from vlarobot.execution.worldmind_heuristic_qualification import (
    _CONFIG,  # noqa: PLC2701
    _EVAL,  # noqa: PLC2701
    _PLAN,  # noqa: PLC2701
    _PROXY_CONFIG,  # noqa: PLC2701
    _ROOT,  # noqa: PLC2701
    _SCENARIOS,  # noqa: PLC2701
    _WORLD,  # noqa: PLC2701
    LIFT_RECOVERY_PROFILE_PATH,
    LIFT_RECOVERY_PROFILE_SHA,
    MATCHED_PROXY_PATH,
    MATCHED_PROXY_SHA,
    SETTLED_CONFIG_SHA,
    STATEFUL_CONFIG_PATH,
    STATEFUL_CONFIG_SHA,
    STATEFUL_PROXY_PROFILES,
    STATEFUL_PROXY_SHA,
    STATEFUL_REGISTRATION,
    STATEFUL_REGISTRATION_SHA,
    STATEFUL_WORLD_MODEL_PROFILE,
    VERIFIED_HOME_PROFILE_PATH,
    VERIFIED_HOME_PROFILE_SHA,
    _sha,  # noqa: PLC2701
    _verified_head,  # noqa: PLC2701
    executor_profile_sha,
    run_frozen_test_episode,
    run_heuristic_validation_episode,
    run_proxy_validation_episode,
)
from vlarobot.execution.worldmind_offline_boundary import _verify_event  # noqa: PLC2701
from vlarobot.execution.worldmind_stage1_pipeline import _plan_sha256  # noqa: PLC2701

_ELIGIBLE_TERMINALS = frozenset(
    {
        "COMPLETE",
        "VERIFICATION_ABORT",
        "NO_SAFE_SET",
        "LIFT_RECOVERY_FAILED",
        "NO_RANKABLE_CANDIDATE",
        "motor_veto",
        "backend_rejected",
        "IK_INFEASIBLE",
        "PLACEMENT_FAILED",
        "GRIPPER_OPEN_TIMEOUT",
        "WORLDMIND_BOUND_EXHAUSTED",
    }
)


def run_trial(output: Path, episode_id: str, seed: int, run_attempt: Any) -> dict[str, Any]:
    """Keep both attempts when the first runner outcome is unreliable."""
    episode_path = output / episode_id
    episode_path.mkdir()
    attempts = []
    for attempt in (1, 2):
        attempt_path = episode_path / f"attempt-{attempt}"
        attempt_path.mkdir()
        started = time.perf_counter()
        try:
            terminal = run_attempt(attempt_path, episode_id)
        except Exception as exc:
            terminal = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "attempt": attempt,
                "terminal": terminal,
                "eligible": terminal in _ELIGIBLE_TERMINALS,
                "seconds": time.perf_counter() - started,
            }
        )
        if attempts[-1]["eligible"]:
            break
    return {
        "episode_id": episode_id,
        "seed": seed,
        "attempts": attempts,
        "eligible": attempts[-1]["eligible"],
        "success": attempts[-1]["terminal"] == "COMPLETE" if attempts[-1]["eligible"] else None,
    }


def _read_event(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    return _verify_event(path, document["sha256"])


def verify_policy_contrast(reference: Path, policy: str, proxy_hash: str) -> None:
    """Require an actual same-pool selection difference before a second policy launch."""
    launch = _read_event(reference / "launch.json")
    if (
        launch["proxy_policy"] != "equal"
        or launch["proxy_config_sha256"] != proxy_hash
        or launch["plan_sha256"] != _PLAN
    ):
        raise ValueError("policy contrast requires the matching equal-policy VALIDATION campaign")
    home_profile = launch.get("home_profile_sha256")
    if home_profile not in (None, VERIFIED_HOME_PROFILE_SHA):
        raise ValueError("unapproved HOME contrast profile")
    lift_profile = launch.get("lift_recovery_profile_sha256")
    if lift_profile not in (None, LIFT_RECOVERY_PROFILE_SHA):
        raise ValueError("unapproved LIFT contrast profile")
    if launch.get("executor_config_sha256") == executor_profile_sha(
        STATEFUL_PROXY_SHA, home_profile is not None, lift_profile is not None
    ):
        from vlarobot.execution.worldmind_stateful_proxy import select_phase_assessments
        from vlarobot.risk.gate import CandidateAssessment

        if policy not in STATEFUL_PROXY_PROFILES:
            raise ValueError("unapproved stateful contrast policy")
        for path in sorted(
            reference.glob("validation-*/attempt-*/episode/chunk-*/decision.msgpack")
        ):
            payload = decode_message(path.read_bytes())["payload"]
            records = payload["proxy"]["assessments"]
            if not any("placement_tolerance" in row["checks"] for row in records):
                continue
            candidates = [
                CandidateAssessment(tuple(row["checks"].items()), row["components"])
                for row in records
            ]
            first = select_phase_assessments("equal", True, candidates).selected_index
            second = select_phase_assessments(policy, True, candidates).selected_index
            pool = payload["pool"]
            if payload["selected"] != (None if first is None else pool[first]["candidate_id"]):
                raise ValueError("saved stateful equal selection does not reconcile")
            if (
                first is not None
                and second is not None
                and pool[first]["chunk"] != pool[second]["chunk"]
            ):
                return
        raise ValueError(f"no distinct observed-pool selections for {policy}")
    for path in reference.glob("validation-*/attempt-*/episode/decision-*/decision.msgpack"):
        choices = decode_message(path.read_bytes())["payload"]["proxy"]["policy_choices"]
        if choices["equal"] != choices[policy]:
            return
    raise ValueError(f"no distinct observed-pool selections for {policy}")


def freeze_validation(out: Path, runs: list[Path]) -> str:
    """Verify eligible per-trial evidence and record the selected policy without touching TEST."""
    from vlarobot.execution.worldmind_idle_fingers import record

    if not date(2026, 11, 21) <= date.today() <= date(2026, 11, 29):
        raise ValueError("selection must finish in the authorized VALIDATION window")
    if not runs:
        raise ValueError("at least one completed VALIDATION campaign is required")
    first = _read_event(runs[0] / "launch.json")
    stateful = first["proxy_config_sha256"] == MATCHED_PROXY_SHA
    home_profile = first.get("home_profile_sha256")
    if home_profile is not None and (
        not stateful
        or home_profile != VERIFIED_HOME_PROFILE_SHA
        or _sha((_ROOT / VERIFIED_HOME_PROFILE_PATH).read_bytes()) != VERIFIED_HOME_PROFILE_SHA
    ):
        raise ValueError("unapproved HOME freeze profile")
    verified_home = home_profile is not None
    lift_profile = first.get("lift_recovery_profile_sha256")
    if lift_profile is not None and (
        not verified_home
        or lift_profile != LIFT_RECOVERY_PROFILE_SHA
        or _sha((_ROOT / LIFT_RECOVERY_PROFILE_PATH).read_bytes()) != LIFT_RECOVERY_PROFILE_SHA
    ):
        raise ValueError("unapproved LIFT freeze profile")
    if stateful and len(runs) != len(STATEFUL_PROXY_PROFILES):
        raise ValueError("stateful freeze requires both approved equal and goal-heavy outcomes")
    proxy_path = MATCHED_PROXY_PATH if stateful else "configs/worldmind_proxy_nov24.yaml"
    proxy_hash = _sha((_ROOT / proxy_path).read_bytes())
    if stateful and proxy_hash != MATCHED_PROXY_SHA:
        raise ValueError("matched clearance profile pin mismatch")
    outcomes = {}
    sources = {}
    episode_ids = tuple(entry.scene.name for entry in build_episode_plan("validation", 0))
    for run_dir in runs:
        launch_path = run_dir / "launch.json"
        launch = _read_event(launch_path)
        policy = launch["proxy_policy"]
        if (
            policy not in GRID | OPTIONAL_GRID
            or launch["arm"] != "full_system"
            or launch["split"] != "validation"
            or (
                stateful
                and (
                    policy not in STATEFUL_PROXY_PROFILES
                    or launch.get("home_profile_sha256") != home_profile
                    or launch.get("lift_recovery_profile_sha256") != lift_profile
                    or launch["executor_config_sha256"]
                    != executor_profile_sha(
                        STATEFUL_PROXY_PROFILES[policy][1], verified_home, lift_profile is not None
                    )
                    or _sha((_ROOT / STATEFUL_PROXY_PROFILES[policy][0]).read_bytes())
                    != STATEFUL_PROXY_PROFILES[policy][1]
                )
            )
            or policy in sources
            or launch["proxy_config_sha256"] != proxy_hash
            or launch["config_sha256"] != _CONFIG
            or launch["plan_sha256"] != _PLAN
        ):
            raise ValueError("incompatible, duplicated, or unpinned policy campaign")
        terminal = run_dir / "terminal.json"
        campaign = _read_event(terminal)
        if campaign["eligible"] != 18 or len(campaign["results"]) != 18:
            raise ValueError("missing or unreliable trial blocks selection")
        outcomes[policy] = {}
        for row in campaign["results"]:
            episode = row["episode_id"]
            if (
                episode not in episode_ids
                or episode in outcomes[policy]
                or row != _read_event(run_dir / episode / "result.json")
                or row["eligible"] is not True
                or type(row["success"]) is not bool
            ):
                raise ValueError("corrupt or unreliable trial blocks selection")
            outcomes[policy][episode] = row["success"]
        sources[policy] = {
            "path": str(run_dir),
            "terminal_sha256": _sha(terminal.read_bytes()),
            **(
                {
                    "launch_sha256": _sha(launch_path.read_bytes()),
                    "executor_config_sha256": launch["executor_config_sha256"],
                }
                if stateful
                else {}
            ),
        }
    for policy in sources.keys() - {"equal"}:
        verify_policy_contrast(Path(sources["equal"]["path"]), policy, proxy_hash)
    winner, scores = select_policy(episode_ids, outcomes)
    record(
        out,
        {
            "schema_version": 2 if stateful else 1,
            "selected_policy": winner,
            "scores": scores,
            "sources": sources,
            "weights": (GRID | OPTIONAL_GRID)[winner],
            "maximize": False,
            "split": "validation",
            "config_sha256": proxy_hash,
            **(
                {
                    "executor_config_sha256": executor_profile_sha(
                        STATEFUL_PROXY_PROFILES[winner][1], verified_home, lift_profile is not None
                    ),
                    "home_profile_sha256": home_profile,
                    "lift_recovery_profile_sha256": lift_profile,
                }
                if stateful
                else {}
            ),
            "plan_sha256": _PLAN,
            "interpretation": "same-set selection statistic",
        },
    )
    return winner


def committed_test_policy(path: Path) -> str:
    """Ensure TEST only uses the immutable November 29 or earlier selection commit."""
    relative = path.resolve().relative_to(_ROOT)
    required_hash = _sha((_ROOT / "configs/worldmind_proxy_nov24.yaml").read_bytes())
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative.as_posix()}"], cwd=_ROOT)
    commit_date = subprocess.check_output(
        ["git", "log", "-1", "--format=%cs", "--", relative.as_posix()], cwd=_ROOT, text=True
    ).strip()
    if (
        committed != path.read_bytes()
        or not commit_date
        or date.fromisoformat(commit_date) > date(2026, 11, 29)
    ):
        raise ValueError("TEST requires the selection committed by November 29")
    frozen = _read_event(path)
    policy = frozen["selected_policy"]
    if (
        frozen["split"] != "validation"
        or frozen["plan_sha256"] != _PLAN
        or frozen["config_sha256"] != required_hash
        or policy not in GRID | OPTIONAL_GRID
        or tuple(frozen["weights"]) != (GRID | OPTIONAL_GRID)[policy]
        or frozen["maximize"] is not False
    ):
        raise ValueError("TEST freeze artifact does not match the approved policy")
    return policy


def run(
    output: Path,
    expected_head: str,
    workers: int,
    policy: str | None = None,
    contrast_evidence: Path | None = None,
    *,
    split: str = "validation",
    frozen_policy: Path | None = None,
    baseline: bool = False,
    settled_init: bool = False,
    matched_clearance: bool = False,
    stateful: bool = False,
    world_model: bool = False,
    verified_home: bool = False,
) -> None:
    from vlarobot.execution.worldmind_idle_fingers import RUNTIME, RUNTIME_SHA, record

    lift_recovery = bool(verified_home and policy and not world_model)
    if lift_recovery and (
        _sha((_ROOT / LIFT_RECOVERY_PROFILE_PATH).read_bytes()) != LIFT_RECOVERY_PROFILE_SHA
    ):
        raise ValueError("unapproved LIFT campaign profile")
    if verified_home and (
        not stateful
        or split != "validation"
        or _sha((_ROOT / VERIFIED_HOME_PROFILE_PATH).read_bytes()) != VERIFIED_HOME_PROFILE_SHA
    ):
        raise ValueError("verified HOME requires pinned stateful VALIDATION")
    if world_model and (
        split != "validation" or not stateful or not matched_clearance or policy != "equal"
    ):
        raise ValueError("arm-4 is scoped to the matched equal VALIDATION profile")
    if split == "validation":
        if baseline or frozen_policy or date.today() < date(2026, 11, 21):
            raise ValueError("live VALIDATION is outside the approved scope or dates")
    elif split == "test":
        if policy is not None or date.today() < date(2026, 11, 30):
            raise ValueError("live TEST is outside the approved scope or dates")
        if baseline and frozen_policy:
            raise ValueError("baseline does not use a WorldMind policy")
        policy = committed_test_policy(frozen_policy) if frozen_policy else None
    else:
        raise ValueError("only authorized VALIDATION and TEST are supported")
    if not 1 <= workers <= 3:
        raise ValueError("use one to three isolated workers")
    if stateful:
        if (
            split != "validation"
            or baseline
            or settled_init
            or (policy, matched_clearance)
            not in {(None, False), ("equal", True), ("goal_x2", True)}
        ):
            raise ValueError(
                "stateful executor is scoped to approved arm-3, arm-4 or arm-5 VALIDATION"
            )
        branch = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=_ROOT, check=False, capture_output=True
        )
        if branch.returncode != 1:
            raise ValueError("stateful VALIDATION requires a detached campaign worktree")
        stateful_path, stateful_hash = STATEFUL_PROXY_PROFILES.get(
            policy, (STATEFUL_CONFIG_PATH, STATEFUL_CONFIG_SHA)
        )
        if world_model:
            stateful_path, stateful_hash = STATEFUL_WORLD_MODEL_PROFILE
        if (
            _sha((_ROOT / stateful_path).read_bytes()) != stateful_hash
            or os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE") != STATEFUL_REGISTRATION
            or _sha((_ROOT / "configs" / STATEFUL_REGISTRATION).read_bytes())
            != STATEFUL_REGISTRATION_SHA
        ):
            raise ValueError("stateful executor profile pin mismatch")
    if matched_clearance and (
        split != "validation"
        or settled_init
        or (policy != "equal" and not (stateful and policy == "goal_x2"))
    ):
        raise ValueError("matched clearance is scoped to arm-5 equal VALIDATION")
    if settled_init and (split != "validation" or policy is not None or baseline):
        raise ValueError("settled INIT is the corrected arm-3 VALIDATION executor")
    if (
        settled_init
        and _sha((_ROOT / "configs/worldmind_arm3_settled_init.yaml").read_bytes())
        != SETTLED_CONFIG_SHA
    ):
        raise ValueError("settled INIT profile pin mismatch")
    _verified_head(expected_head)
    world = _ROOT / "worlds/factory.wbt"
    config = _ROOT / "configs/worldmind_diagnostic.approved.yaml"
    for path, digest in (
        (world, _WORLD),
        (config, _CONFIG),
        (RUNTIME, RUNTIME_SHA),
        (_ROOT / "configs/eval.yaml", _EVAL),
        (_ROOT / "configs/scenarios.yaml", _SCENARIOS),
    ):
        if _sha(path.read_bytes()) != digest:
            raise ValueError(f"campaign provenance mismatch: {path}")
    proxy_hash = None
    if policy is not None:
        if policy not in GRID | OPTIONAL_GRID:
            raise ValueError("unknown proxy policy")
        from vlarobot.execution.worldmind_proxy_binding import verified_table_world

        verified_table_world(_ROOT)
        proxy_path = (
            MATCHED_PROXY_PATH if matched_clearance else "configs/worldmind_proxy_nov24.yaml"
        )
        proxy_hash = _sha((_ROOT / proxy_path).read_bytes())
        if proxy_hash != (MATCHED_PROXY_SHA if matched_clearance else _PROXY_CONFIG):
            raise ValueError("November 24 proxy configuration pin mismatch")
        if policy != "equal" and split == "validation":
            if contrast_evidence is None:
                raise ValueError("second policy requires observed-pool selection contrast evidence")
            verify_policy_contrast(contrast_evidence, policy, proxy_hash)
    plan = build_episode_plan(split, 0, allow_test=split == "test")
    plan_hash = _plan_sha256(plan)
    if (split == "validation" and (len(plan) != 18 or plan_hash != _PLAN)) or (
        split == "test" and len(plan) != 30
    ):
        raise ValueError("campaign requires the frozen episode plan")
    entries = {entry.scene.name: entry for entry in plan}
    output.mkdir(exist_ok=False)
    record(
        output / "launch.json",
        {
            "arm": "scripted_baseline"
            if baseline
            else (
                "vla_world_model" if world_model else "full_system" if policy else "vla_risk_veto"
            ),
            "prediction_mode": "analytic" if policy else "none",
            "predictive_veto": bool(policy and not world_model),
            "policy_status": "approved_pre_calibration_equal" if world_model else None,
            "proxy_policy": policy,
            "executor_config_sha256": (
                executor_profile_sha(stateful_hash, verified_home, lift_recovery)
                if stateful
                else SETTLED_CONFIG_SHA
                if settled_init or matched_clearance
                else None
            ),
            "proxy_config_sha256": proxy_hash,
            "home_profile_sha256": VERIFIED_HOME_PROFILE_SHA if verified_home else None,
            "lift_recovery_profile_sha256": LIFT_RECOVERY_PROFILE_SHA if lift_recovery else None,
            "config_sha256": _CONFIG,
            "split": split,
            "trials": len(plan),
            "workers": workers,
            "started_at": datetime.now().astimezone().isoformat(),
            "head": expected_head,
            "plan_sha256": plan_hash,
        },
    )
    launch = _read_event(output / "launch.json")
    print(json.dumps({"event": "campaign_launched", "evidence": str(output), **launch}), flush=True)
    started = time.perf_counter()

    def run_attempt(path: Path, episode_id: str) -> str:
        status = path / "controller-status.json"
        launch_started_at_s = time.time()
        with launch_webots_session(
            webots=str(RUNTIME),
            world=world,
            trace_dir=path,
            trace_prefix="worldmind",
            runtime_attestation_required=True,
            rpc_shutdown_on_exit=baseline,
            extra_env={
                "VLAROBOT_WEBOTS_MODE": "server",
                "VLAROBOT_FINGER_IDLE_ONLY": "0",
                "VLAROBOT_WORLD_SHA256": _WORLD,
                "VLAROBOT_DIAGNOSTIC_STATUS_PATH": str(status),
                **(
                    {
                        "VLAROBOT_URDF_REGISTRATION_FILE": STATEFUL_REGISTRATION,
                        "WEBOTS_PYTHON": sys.executable,
                    }
                    if stateful
                    else {}
                ),
            },
        ) as owned:
            arguments = dict(
                status_path=status,
                launch_started_at_s=launch_started_at_s,
                directory=path / "episode",
                expected_head=expected_head,
                episode_id=episode_id,
            )
            if baseline:
                from vlarobot.backends.webots.runtime_attestation import (
                    read_owned_controller_session,
                )
                from vlarobot.eval.runner import EvalRunConfig, run_evaluation
                from vlarobot.execution.scripted_baseline import scripted_policy_factory

                session_id = read_owned_controller_session(
                    status,
                    endpoint=owned.endpoint,
                    process=owned.process,
                    runtime_attestation=owned.runtime_attestation,
                    launch_started_at_s=launch_started_at_s,
                )
                ping = owned.client.ping()["payload"]
                if ping.get("session_id") != session_id or ping.get("world_sha256") != _WORLD:
                    raise ValueError("baseline TEST ownership or world mismatch")
                report = run_evaluation(
                    [entries[episode_id]],
                    owned.client,
                    EvalRunConfig(
                        path / "episode", split="test", record_frames=True, stop_on_exception=True
                    ),
                    policy_factory=scripted_policy_factory,
                )
                terminal = "COMPLETE" if report.rows[0]["success"] else "PLACEMENT_FAILED"
            elif split == "test":
                terminal = run_frozen_test_episode(owned, frozen_policy=frozen_policy, **arguments)
            elif policy is None:
                terminal = run_heuristic_validation_episode(
                    owned,
                    settled_init=settled_init,
                    stateful=stateful,
                    verified_home=verified_home,
                    **arguments,
                )
            else:
                terminal = run_proxy_validation_episode(
                    owned,
                    policy_id=policy,
                    matched_clearance=matched_clearance,
                    stateful=stateful,
                    world_model=world_model,
                    verified_home=verified_home,
                    **arguments,
                )
        if owned.process.poll() != 0 or (path / "episode/shutdown-failure.json").exists():
            raise RuntimeError("owned episode shutdown failed")
        return terminal

    def trial(entry: Any) -> dict[str, Any]:
        result = run_trial(output, entry.scene.name, entry.seed, run_attempt)
        record(output / entry.scene.name / "result.json", result)
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(trial, plan))
    eligible = [row for row in results if row["eligible"]]
    seconds = time.perf_counter() - started
    record(
        output / "terminal.json",
        {
            "seconds": seconds,
            "results": results,
            "eligible": len(eligible),
            "successes": sum(row["success"] is True for row in eligible),
            "eligible_per_hour": 3600 * len(eligible) / seconds,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--expected-head")
    parser.add_argument("--policy", choices=tuple(GRID | OPTIONAL_GRID))
    parser.add_argument("--contrast-evidence", type=Path)
    parser.add_argument("--freeze-from", type=Path, nargs="+")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--frozen-policy", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--settled-init", action="store_true")
    parser.add_argument("--matched-clearance", action="store_true")
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--world-model", action="store_true")
    parser.add_argument("--verified-home", action="store_true")
    args = parser.parse_args()
    if args.freeze_from:
        if (
            args.split != "validation"
            or args.policy
            or args.contrast_evidence
            or args.frozen_policy
            or args.baseline
            or args.settled_init
            or args.matched_clearance
            or args.stateful
            or args.world_model
            or args.verified_home
        ):
            parser.error("--freeze-from is offline and cannot be combined with run options")
        print(freeze_validation(args.output.resolve(), args.freeze_from), flush=True)
        raise SystemExit(0)
    head = (
        args.expected_head
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    )
    run(
        args.output.resolve(),
        head,
        args.workers,
        args.policy,
        args.contrast_evidence,
        split=args.split,
        frozen_policy=args.frozen_policy,
        baseline=args.baseline,
        settled_init=args.settled_init,
        matched_clearance=args.matched_clearance,
        stateful=args.stateful,
        world_model=args.world_model,
        verified_home=args.verified_home,
    )
