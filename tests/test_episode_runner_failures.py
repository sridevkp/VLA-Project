from __future__ import annotations

from pathlib import Path

from vlarobot.execution.episode_runner import EpisodeOutcome, RunnerSettings, run_episodes


def _settings(tmp_path: Path) -> RunnerSettings:
    return RunnerSettings(
        webots="webots",
        world=tmp_path / "factory.wbt",
        endpoint="tcp://127.0.0.1:5555",
        trace_dir=tmp_path / "traces",
        seed=0,
    )


def test_run_episodes_fresh_world_records_runner_exception(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    def boom(settings, index, seed):
        raise RuntimeError("boom")

    monkeypatch.setattr("vlarobot.execution.episode_runner._run_fresh_episode", boom)
    outcomes = run_episodes(settings, n=2, fresh_world=True)
    assert len(outcomes) == 2
    assert all(isinstance(o, EpisodeOutcome) for o in outcomes)
    assert all(o.success is False for o in outcomes)
    assert all(o.failure_state == "RUNNER" for o in outcomes)
    assert all("RuntimeError: boom" in (o.failure_reason or "") for o in outcomes)


def test_run_episodes_fresh_world_records_failed_result(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    def failed(settings, index, seed):
        return EpisodeOutcome(
            index=index,
            seed=seed,
            success=False,
            final_state="FAILED",
            failure_state="CLOSE",
            failure_reason="policy failed",
            final_target_distance_m=0.2,
            home_reached=False,
            place_distance_ok=False,
            can_height_ok=False,
            can_tilt_ok=False,
            can_height_error_m=None,
            can_tilt_deg=None,
            rejections=0,
            startup_s=1.0,
            episode_s=2.0,
            trace_path=tmp_path / f"episode-{index:02d}.jsonl",
            log_path=tmp_path / f"episode-{index:02d}.log",
        )

    monkeypatch.setattr("vlarobot.execution.episode_runner._run_fresh_episode", failed)
    outcomes = run_episodes(settings, n=1, fresh_world=True)
    assert len(outcomes) == 1
    assert outcomes[0].success is False
    assert outcomes[0].failure_state == "CLOSE"
    assert outcomes[0].failure_reason == "policy failed"
