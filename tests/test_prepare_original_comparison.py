from __future__ import annotations

import io
import runpy
import subprocess
import tarfile
from pathlib import Path


prepare = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_original_comparison.py")
)["prepare"]


def test_fixture_v2_prepares_clean_1b2e4da_archive(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    archive = subprocess.run(
        ["git", "archive", "1b2e4da"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        source.extractall(tmp_path)

    prepare(tmp_path)

    controller = tmp_path / "controllers" / "tiago_controller"
    sources = list(controller.glob("*.py"))
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    pick_place = (controller / "pick_place.py").read_text(encoding="utf-8")
    motion = (controller / "motion.py").read_text(encoding="utf-8")
    kinematics = (controller / "kinematics.py").read_text(encoding="utf-8")
    entrypoint = (controller / "tiago_controller.py").read_text(encoding="utf-8")
    gripper = (controller / "gripper.py").read_text(encoding="utf-8")

    for field in (
        "can_world_xyz_m",
        "can_base_xyz_m",
        "can_displacement_world_xyz_m",
        "lift_height_m",
        "gripper_opening_m",
        "grasp_detected",
        "close_stall_counter",
        "ik_error_m",
        "ik_tilt_deg",
    ):
        assert field in pick_place
    assert "joint_limits_rad" in pick_place
    assert "joint_target" in motion
    assert '"measured_joint_positions_rad"' in pick_place
    assert '"commanded_joint_target_rad"' in pick_place
    assert '"commanded_joint_positions_rad"' in pick_place
    assert '"measured_joint_positions_rad"' in motion
    assert '"commanded_joint_target_rad"' in motion
    assert "record.update(self.trace_task._trace_measurements())" in motion
    assert '"ik_tilt_deg": self.last_ik_tilt' in motion
    assert "last_close_stall_counter = still" in gripper
    assert "self.rng = np.random.default_rng(0)" in kinematics
    assert "restarts=True" in motion
    assert entrypoint.index("ik = Kinematics()") < entrypoint.index("task.initialize_posture()")
    assert 'STOP_AFTER = "LIFT"' in (controller / "config.py").read_text(encoding="utf-8")
    assert 'controller "<extern>"' in (tmp_path / "worlds" / "factory.wbt").read_text(
        encoding="utf-8"
    )

    prepare(tmp_path)


