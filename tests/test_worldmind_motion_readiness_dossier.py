"""Inert tests for the machine-readable motion-readiness dossier and CLI.

No test launches Webots, opens a socket, moves the robot, solves IK, scores an episode or reads a
data split. The dossier is diagnostic and can never approve motion.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from vlarobot.cli import app
from vlarobot.eval.worldmind_motion_readiness_dossier import (
    SWEEP_INEQUALITY,
    build_motion_readiness_dossier,
)

_RUNNER = CliRunner()


def test_dossier_is_deterministic_fail_closed_and_never_approved():
    first = build_motion_readiness_dossier()
    second = build_motion_readiness_dossier()
    assert first == second
    assert first["schema_version"] == 1
    assert first["status"] == "diagnostic_not_certified"
    assert first["approved"] is False
    assert first["motion_allowed"] is False
    assert first["motion_readiness"] == "NOT_READY"
    assert first["overall_gate"] == "FAIL"
    assert first["gate_counts"] == {"PASS": 4, "FAIL": 1, "UNVERIFIED": 4}
    assert first["swept_inequality"] == SWEEP_INEQUALITY
    assert "max(point_speed*dt/substeps, joint_displacement*reach)" in SWEEP_INEQUALITY
    assert "+ numerical_error + transform_error" in SWEEP_INEQUALITY


def test_gate_table_marks_missing_bounds_and_authorization_explicitly():
    dossier = build_motion_readiness_dossier()
    gates = {item["id"]: item for item in dossier["gates"]}
    assert gates["frozen_evidence"]["status"] == "PASS"
    assert gates["swept_residual"]["status"] == "UNVERIFIED"
    assert gates["swept_residual"]["details"]["residual_m"] is None
    assert gates["swept_residual"]["details"]["numerical_error_bound_m"] is None
    assert gates["swept_residual"]["details"]["transform_error_bound_m"] is None
    assert gates["gripper_envelope"]["status"] == "PASS"
    assert gates["can_lifecycle"]["status"] == "UNVERIFIED"
    assert gates["self_collision"]["status"] == "UNVERIFIED"
    assert gates["table_transform"]["status"] == "UNVERIFIED"
    assert gates["reference_cases"]["status"] == "PASS"
    assert gates["production_binding"]["status"] == "PASS"
    assert gates["motion_authorization"]["status"] == "FAIL"
    assert gates["motion_authorization"]["details"]["approved"] is False
    assert any(
        item["id"] == "independent_safety_certificate"
        for item in gates["motion_authorization"]["details"]["owner_approvals_required"]
    )
    assert "nested Table SolidBox nodes" in dossier["smallest_remaining_experiment"]


def test_every_independent_reference_case_is_replayed_and_passes():
    dossier = build_motion_readiness_dossier()
    gate = next(item for item in dossier["gates"] if item["id"] == "reference_cases")
    cases = gate["details"]["cases"]
    assert dossier["reference_case_count"] == 12
    assert len(cases) == 12
    assert len({item["id"] for item in cases}) == 12
    assert all(item["pass"] is True for item in cases)
    assert {item["kind"] for item in cases} == {
        "swept_bound",
        "required_substeps",
        "self_collision",
        "carried_lifecycle",
        "gripper_report",
        "table_report",
        "checker_gate",
    }
    joint_dominated = next(item for item in cases if item["id"] == "joint_dominated_motion_bound")
    assert joint_dominated["actual"]["residual_m"] == -0.3


def test_cli_writes_dossier_and_exits_nonzero_without_approving(tmp_path):
    output = tmp_path / "dossier.json"
    result = _RUNNER.invoke(app, ["worldmind-motion-readiness-dossier", "--output", str(output)])
    assert result.exit_code == 1
    assert "overall=FAIL" in result.stdout
    assert "motion_readiness=NOT_READY" in result.stdout
    assert "approved=False" in result.stdout
    assert "  UNVERIFIED: swept_residual -" in result.stdout
    assert "  FAIL: motion_authorization -" in result.stdout
    assert "webots_launch=0 apply=0 step=0" in result.stdout
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == build_motion_readiness_dossier()
