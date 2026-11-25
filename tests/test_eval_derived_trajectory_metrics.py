"""Inert tests for supplemental trajectory metrics and the read-only derivation tool."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vlarobot.eval.derived_metrics import (
    derive_supplemental_trajectory_metrics,
    end_effector_path_length_m,
    simulation_duration_s,
)

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "derive_supplemental_metrics.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("derive_supplemental_metrics", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_path_length_requires_explicit_base_frame_and_metres() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="frame"):
        end_effector_path_length_m(positions, frame="world", units="m")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="units"):
        end_effector_path_length_m(positions, frame="base", units="mm")  # type: ignore[arg-type]


def test_path_length_is_sum_of_consecutive_base_frame_metre_deltas() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
    assert end_effector_path_length_m(positions, frame="base", units="m") == pytest.approx(17.0)


def test_path_length_empty_and_single_sample_policies_are_explicit() -> None:
    with pytest.raises(ValueError, match="empty"):
        end_effector_path_length_m(np.empty((0, 3)), frame="base", units="m")
    assert end_effector_path_length_m(np.zeros((1, 3)), frame="base", units="m") == 0.0


def test_path_length_rejects_invalid_shape_and_nonfinite_coordinates() -> None:
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        end_effector_path_length_m(np.zeros((2, 2)), frame="base", units="m")
    with pytest.raises(ValueError, match="non-finite"):
        end_effector_path_length_m(
            np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]), frame="base", units="m"
        )


def test_simulation_duration_is_not_wall_clock_and_has_explicit_edge_policies() -> None:
    timestamps = np.array([0.0, 0.032, 0.064])
    assert simulation_duration_s(timestamps) == pytest.approx(0.064)
    assert simulation_duration_s(np.array([1.5])) == 0.0
    with pytest.raises(ValueError, match="empty"):
        simulation_duration_s(np.array([]))
    with pytest.raises(ValueError, match="non-finite"):
        simulation_duration_s(np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="strictly increasing"):
        simulation_duration_s(np.array([0.0, 0.0, 0.032]))
    with pytest.raises(ValueError, match="strictly increasing"):
        simulation_duration_s(np.array([0.0, 0.032, 0.016]))


def test_derive_rejects_time_axis_length_mismatch() -> None:
    with pytest.raises(ValueError, match="time axis|samples"):
        derive_supplemental_trajectory_metrics(
            ee_xyz=np.zeros((3, 3)), t=np.zeros(2), frame="base", units="m"
        )


def test_supplemental_tool_writes_once_and_does_not_touch_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    episode_dir = tmp_path / "episode-00"
    episode_dir.mkdir()
    steps = episode_dir / "steps.npz"
    meta = episode_dir / "meta.json"
    np.savez(
        steps,
        t=np.array([0.0, 0.032, 0.064]),
        ee_xyz=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]),
    )
    meta.write_text(json.dumps({"index": 0}), encoding="utf-8")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (steps, meta)}

    tool = _load_tool()
    out = tmp_path / "supplemental" / "episode-00.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_supplemental_metrics.py",
            "--episode-dir",
            str(episode_dir),
            "--out",
            str(out),
            "--ee-frame",
            "base",
            "--ee-units",
            "m",
        ],
    )
    tool.main()
    assert "wrote supplemental metrics" in capsys.readouterr().out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["supplemental_metrics"]["end_effector_path_length_m"] == pytest.approx(17.0)
    assert payload["supplemental_metrics"]["simulation_duration_s"] == pytest.approx(0.064)
    assert payload["frame_and_units"]["ee_frame"] == "base"
    assert payload["execution_time_separation"]["wall_clock_execution_time_s"] is None

    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (steps, meta)}
    assert after == before

    monkeypatch.setattr(sys, "argv", sys.argv)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tool._ensure_safe_output(out, episode_dir)


def test_tool_rejects_frozen_names_and_output_inside_episode(tmp_path: Path) -> None:
    tool = _load_tool()
    episode_dir = tmp_path / "episode-00"
    episode_dir.mkdir()
    with pytest.raises(ValueError, match="frozen evidence filename"):
        tool._ensure_safe_output(tmp_path / "summary.json", episode_dir)
    with pytest.raises(ValueError, match="inside the source episode"):
        tool._ensure_safe_output(episode_dir / "supplemental.json", episode_dir)
