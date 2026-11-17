"""Authorized A-D1: checked TRAIN snapshot and exactly two frozen CPU IK calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

IDENTITY = "A-D1-train-clear-i00-s100-frame0-v1"


def sha(path: Path) -> str:
    """Fingerprint a regular input without modifying it."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing/symlinked input: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read an existing evidence document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Never overwrite diagnostic evidence; reject nonfinite JSON."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked(path: Path, expected: str) -> None:
    """Stop instead of repairing a provenance mismatch."""
    if sha(path) != expected:
        raise ValueError(f"provenance mismatch: {path}")


def prepare(project: Path, evidence: Path, out: Path) -> None:
    """Extract only the existing native first row; no model/solver imports."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    if out.exists():
        raise FileExistsError("preserve existing diagnostic output")
    dataset = project / "runs/datasets/tiago-train-v1"
    approval = read(dataset / "training_approval.json")
    checked(dataset / "dataset-integrity.json", approval["dataset_integrity_sha256"])
    checked(dataset / "source-provenance.json", approval["source_provenance_sha256"])
    integrity = read(dataset / "dataset-integrity.json")
    parquet = dataset / "data/chunk-000/file-000.parquet"
    checked(parquet, integrity["files_sha256"][parquet.relative_to(dataset).as_posix()])
    local = read(evidence / "docs/evidence/local-model-validation-step20000.json")
    validation = evidence / "runs/local-model-validation-v1"
    audit_path = validation / "processor-action-audit.json"
    checked(audit_path, local["validation_artifact_hashes"][audit_path.name])
    audit = read(audit_path)["source_observation"]
    columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "action",
        "observation.images.camera1",
        "task_index",
    ]
    row = next(pq.ParquetFile(parquet).iter_batches(batch_size=1, columns=columns)).to_pylist()[0]
    if (row["episode_index"], row["frame_index"], row["task_index"]) != (0, 0, 0):
        raise ValueError("not the authorized native episode0/frame0/task0")
    encoded = row["observation.images.camera1"]["bytes"]
    rgb = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
    if hashlib.sha256(encoded).hexdigest() != audit["encoded_sha256"]:
        raise ValueError("native encoded image differs from saved inference input")
    if hashlib.sha256(rgb.tobytes()).hexdigest() != audit["rgb_sha256"]:
        raise ValueError("native RGB differs from saved inference input")
    np.testing.assert_array_equal(
        np.asarray(row["observation.state"], dtype=np.float32),
        np.asarray(audit["state"], dtype=np.float32),
    )
    row.pop("observation.images.camera1")
    snapshot = {
        "identity": IDENTITY,
        "row": row,
        "encoded_sha256": audit["encoded_sha256"],
        "rgb_sha256": audit["rgb_sha256"],
        "parquet_sha256": sha(parquet),
        "audit_sha256": sha(audit_path),
        "producer_executable": sys.executable,
        "pyarrow_version": importlib.metadata.version("pyarrow"),
    }
    out.mkdir(parents=True, exist_ok=False)
    write(out / "native-snapshot.json", snapshot)
    print("SNAPSHOT_PREPARED_NO_IK_NO_INFERENCE", flush=True)


def decompose(jacobian: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    """Euclidean joint row/null-space split; no calibrated singularity cutoff."""
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
    cutoff = float(np.finfo(float).eps * max(jacobian.shape) * singular[0])
    rank = int(np.count_nonzero(singular > cutoff))
    task = vt[:rank].T @ (vt[:rank] @ delta)
    null = vt[rank:].T @ (vt[rank:] @ delta)
    return {
        "singular_values": singular.tolist(),
        "numerical_rank": rank,
        "rank_cutoff_machine_precision": cutoff,
        "task_component_rad": task.tolist(),
        "null_component_rad": null.tolist(),
        "task_norm_rad": float(np.linalg.norm(task)),
        "null_norm_rad": float(np.linalg.norm(null)),
        "reconstruction_error_rad": float(np.linalg.norm(delta - task - null)),
        "null_task_residual": (jacobian @ null).tolist(),
        "linearized_task_delta": (jacobian @ delta).tolist(),
    }


def geometric_jacobian(kin: Any, q: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Base-frame spatial Jacobian of the existing registered revolute chain."""
    frames = kin.chain.forward_kinematics(q, full_kinematics=True)
    endpoint = frames[-1][:3, 3]
    columns = []
    for name in names:
        index = kin.idx[name]
        link = kin.chain.links[index]
        if link.joint_type != "revolute":
            raise ValueError("authorized arm Jacobian expects only revolute joints")
        axis = frames[index][:3, :3] @ np.asarray(link.rotation)
        columns.append(np.r_[np.cross(axis, endpoint - frames[index][:3, 3]), axis])
    return np.column_stack(columns)


def run(project: Path, evidence: Path, out: Path) -> None:
    """Use the frozen brain interpreter; preflight precedes the only two solves."""
    from PIL import Image

    if sys.prefix != str(project / ".venv"):
        raise ValueError("solver identity requires the frozen brain .venv interpreter")
    marker = out / "run-started.json"
    if marker.exists():
        raise FileExistsError("no diagnostic rerun: two-solve budget must not be repeated")
    if not out.is_dir():
        raise FileNotFoundError("prepare the verified native snapshot first")
    write(marker, {"identity": IDENTITY, "executable": sys.executable, "solve_budget": 2})
    inputs: dict[str, str] = {}

    def pin(path: Path, expected: str | None = None) -> None:
        if expected is not None:
            checked(path, expected)
        inputs[str(path.resolve())] = sha(path)

    snapshot = read(out / "native-snapshot.json")
    if snapshot["identity"] != IDENTITY:
        raise ValueError("wrong diagnostic snapshot identity")
    pin(out / "native-snapshot.json")
    local = read(evidence / "docs/evidence/local-model-validation-step20000.json")
    validation = evidence / "runs/local-model-validation-v1"
    audit_path = validation / "processor-action-audit.json"
    pin(audit_path, local["validation_artifact_hashes"][audit_path.name])
    audit = read(audit_path)
    pin(
        validation / "existing-import/import-validation.json",
        local["validation_artifact_hashes"]["existing-import/import-validation.json"],
    )
    prediction = validation / "existing-import/physical-action-chunk.npy"
    pin(
        prediction, local["validation_artifact_hashes"]["existing-import/physical-action-chunk.npy"]
    )
    dataset = project / "runs/datasets/tiago-train-v1"
    approval = read(dataset / "training_approval.json")
    pin(dataset / "training_approval.json")
    pin(dataset / "source-provenance.json", approval["source_provenance_sha256"])
    pin(dataset / "dataset-integrity.json", approval["dataset_integrity_sha256"])
    provenance = read(dataset / "source-provenance.json")
    integrity = read(dataset / "dataset-integrity.json")
    parquet = dataset / "data/chunk-000/file-000.parquet"
    pin(parquet, integrity["files_sha256"]["data/chunk-000/file-000.parquet"])
    if snapshot["parquet_sha256"] != inputs[str(parquet.resolve())]:
        raise ValueError("prepared native source hash changed")
    entry = provenance["sources"][0]
    if (
        entry["attempt"] != 0
        or entry["seed"] != 100
        or entry["scene"]["name"] != "train-clear-i00-s100"
        or entry["scene"]["metadata"]["split"] != "train"
    ):
        raise ValueError("not authorized TRAIN source attempt0000")
    raw = Path(entry["raw_dir"])
    if raw != project / "runs/demos/train-s100-registered-v1/attempts/attempt-0000":
        raise ValueError("unexpected source recording location")
    for name, key in (
        ("episode-00.steps.npz", "steps_sha256"),
        ("episode-00.actions.jsonl", "commands_sha256"),
        ("episode-00.meta.json", "meta_sha256"),
        ("policy.jsonl", "policy_trace_sha256"),
    ):
        pin(raw / name, entry[key])
    if read(raw / "episode-00.meta.json")["scenario"] != entry["scene"]:
        raise ValueError("source recording scene mismatch")
    image = raw / "frames/episode-00/000000.jpg"
    pin(image, snapshot["encoded_sha256"])
    rgb = np.asarray(Image.open(image).convert("RGB"))
    if (
        hashlib.sha256(rgb.tobytes()).hexdigest() != snapshot["rgb_sha256"]
        or snapshot["rgb_sha256"] != audit["source_observation"]["rgb_sha256"]
    ):
        raise ValueError("saved inference/native/raw RGB mismatch")
    baseline = project / "runs/eval/baseline-train-s0-fresh-v2-registered"
    bp = read(baseline / "provenance.json")
    for name, h in bp["files_sha256"].items():
        pin(project / name, h)
    campaign = project / "runs/eval/vla-train-s0-step20000-recovered-v1"
    campaign_inventory = read(evidence / "docs/evidence/vla-train-step20000-files.json")
    for name in ("summary.json", "vla-gate.json", "proposer-role.json", "provenance.json"):
        pin(campaign / name, campaign_inventory[name]["sha256"])
    for name, h in read(campaign / "provenance.json")["controller_sha256"].items():
        pin(project / name, h)
    for name, h in local["source_hashes"].items():
        path = Path(name)
        if not path.is_absolute():
            path = evidence / path
        pin(path, h)
    archive = project / "runs/colab-prep-v3/source.zip"
    pin(archive, "2d3cb4bd06ac8313b1bb3cdb5aae7f916c2b614dd3884172c4570be9ff36b2f5")
    with zipfile.ZipFile(archive) as z:
        approved_source = json.loads(z.read("snapshot-manifest.json"))
        lock = tomllib.loads(z.read("uv.lock").decode())
    for name in (
        "src/vlarobot/backends/webots/tiago.urdf",
        "src/vlarobot/backends/webots/pose_solver.py",
        "src/vlarobot/vla/dataset_export.py",
    ):
        pin(project / name, approved_source[name])
    versions = {}
    for name in ("ikpy", "numpy", "scipy", "sympy"):
        expected = next(p["version"] for p in lock["package"] if p["name"] == name)
        versions[name] = importlib.metadata.version(name)
        if versions[name] != expected:
            raise ValueError(f"frozen solver dependency identity mismatch: {name}")
    export = validation / "export-step-020000"
    manifest = read(export / "export-manifest.json")
    for name, h in manifest["files_sha256"].items():
        pin(export / name, h)
    if (
        sha(export / "export-manifest.json")
        != read(validation / "existing-import/import-validation.json")["checkpoint_manifest_sha256"]
    ):
        raise ValueError("saved action checkpoint identity mismatch")
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = bp["registration_file"].removeprefix("configs/")
    sys.path.insert(0, str(project / "src"))
    from ikpy.chain import Chain
    from scipy.spatial.transform import Rotation

    from vlarobot.backends.webots.config import ARM_JOINTS, SETTINGS
    from vlarobot.backends.webots.pose_solver import quaternion_matrix
    from vlarobot.core.types import (
        EEPoseTarget,
        GripperCmd,
        JointTarget,
        Pose2D,
        Pose3D,
        RobotState,
        Vec3,
    )
    from vlarobot.execution.vla_campaign import make_vla_adapter
    from vlarobot.vla.action_adapter import pose_delta

    adapter = make_vla_adapter(bp["files_sha256"][bp["registration_file"]])
    solver = adapter.solver
    kin = solver.kinematics
    solver_file = Path(sys.modules[type(solver).__module__].__file__).resolve()
    if solver_file != project / "src/vlarobot/backends/webots/pose_solver.py":
        raise ValueError("loaded solver is not the frozen project's solver")
    pin(Path(sys.modules[Chain.__module__].__file__))
    if kin.model != "urdf" or tuple(ARM_JOINTS) != tuple(f"arm_{i}_joint" for i in range(1, 8)):
        raise ValueError("unexpected solver model/arm mapping")
    with np.load(raw / "episode-00.steps.npz", allow_pickle=False) as z:
        joints = dict(zip(z["joint_names"].tolist(), map(float, z["joint_pos"][0]), strict=True))
        vel = dict(zip(z["joint_names"].tolist(), map(float, z["joint_vel"][0]), strict=True))
        state = RobotState(
            joints,
            vel,
            Pose2D(*map(float, z["base_pose"][0])),
            Pose3D(Vec3(*map(float, z["ee_xyz"][0])), tuple(map(float, z["ee_quat"][0]))),
            float(z["gripper"][0]),
        )
        timestamp = float(z["t"][0])
    row = snapshot["row"]
    np.testing.assert_array_equal(
        np.asarray(row["observation.state"], dtype=np.float32),
        np.asarray([joints[n] for n in ARM_JOINTS] + [state.gripper_opening], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(row["observation.state"], dtype=np.float32),
        np.asarray(audit["source_observation"]["state"], dtype=np.float32),
    )
    boundary = json.loads((raw / "policy.jsonl").read_text(encoding="utf-8").splitlines()[0])
    if (
        boundary["event"] != "state_boundary"
        or boundary["state"] != "INIT"
        or boundary["sim_time_s"] != timestamp
    ):
        raise ValueError("expert target reference is not the matched INIT boundary")
    q_star = {n: float(boundary["commanded_joint_target_rad"][n]) for n in ARM_JOINTS}
    if any(q_star[n] != joints[n] for n in ARM_JOINTS):
        raise ValueError("initial expert hold does not match the approved frame")
    commands = [
        json.loads(line) for line in (raw / "episode-00.actions.jsonl").read_text().splitlines()
    ]
    active = dict(joints)
    grip = 0.0
    prior = -math.inf
    past_joint_events = 0
    for command in commands:
        if command.get("accepted") is not True or command["t"] < prior:
            raise ValueError("unordered/nonaccepted command evidence")
        prior = command["t"]
        if command["t"] <= timestamp + 1e-9:
            if "q" in command["action"]:
                active.update(command["action"]["q"])
                past_joint_events += 1
            elif "opening" in command["action"]:
                grip = float(command["action"]["opening"])
            else:
                raise ValueError("unsupported source command")
    if any(active[n] != q_star[n] for n in ARM_JOINTS):
        raise ValueError("recorded expert target and accepted command replay disagree")
    q0 = np.zeros(len(kin.chain.links))
    for n, v in joints.items():
        if n in kin.idx:
            q0[kin.idx[n]] = v
    q_a = q0.copy()
    for n, v in q_star.items():
        q_a[kin.idx[n]] = v
    fk0 = kin.forward(q0)
    fk_a = kin.forward(q_a)
    recorded = np.eye(4)
    recorded[:3, 3] = state.ee_pose.position.as_array()
    recorded[:3, :3] = quaternion_matrix(state.ee_pose.orientation)
    atol = float(128 * np.finfo(float).eps * max(1, np.linalg.norm(fk0, ord=np.inf)))
    np.testing.assert_allclose(
        fk0,
        recorded,
        atol=atol,
        rtol=0,
        err_msg="registered FK and recorded starting pose mismatch",
    )
    target_b = Pose3D(
        Vec3(*map(float, fk_a[:3, 3])),
        tuple(map(float, Rotation.from_matrix(fk_a[:3, :3]).as_quat())),
    )
    action_b = np.r_[pose_delta(state.ee_pose, target_b), grip]
    np.testing.assert_allclose(
        action_b.astype(np.float32), np.asarray(row["action"], dtype=np.float32), atol=atol, rtol=0
    )
    actions = np.load(prediction, allow_pickle=False)
    if actions.shape != (50, 7) or actions.dtype != np.float32 or not np.isfinite(actions).all():
        raise ValueError("saved prediction is not the approved finite float32 chunk")
    # Case A: validate the original hold, no IK.
    for command in (
        GripperCmd(grip),
        JointTarget(q_star),
        EEPoseTarget(target_b, grip, adapter.safety_monitor.limits.command_horizon),
    ):
        verdict = adapter.safety_monitor.validate(command, state)
        if not verdict.accepted:
            raise ValueError(f"A safety assumption failed: {verdict.reason}")
    write(
        out / "provenance.json",
        {
            "identity": IDENTITY,
            "inputs_sha256": inputs,
            "solver_versions": versions,
            "runtime_executable": sys.executable,
            "registration_sha256": kin.registration_sha256,
            "observed_torso_m": joints["torso_lift_joint"],
            "configured_torso_m_not_substituted": SETTINGS.torso_height,
            "roundoff_pose_match_atol": atol,
            "explicit_joint_apply_events_before_frame0": past_joint_events,
            "expert_target_source": (
                "policy.jsonl line1 commanded_joint_target_rad INIT hold reference; "
                "corroborated by dataset label replay; no JointTarget apply event at frame0"
            ),
            "diagnostic_script_sha256": sha(Path(__file__)),
        },
    )
    # Count at the actual numerical solver entry, not merely at adapter invocation.
    original_ik = Chain.inverse_kinematics
    calls = []

    def counted_ik(self: Any, *args: Any, **kwargs: Any) -> Any:
        if len(calls) >= 2:
            raise RuntimeError("authorized two-solve budget exhausted")
        np.testing.assert_array_equal(kwargs["initial_position"], q0)
        np.testing.assert_array_equal(self.active_links_mask, kin.chain.active_links_mask)
        if kwargs.get("orientation_mode") != "all":
            raise ValueError("frozen full-orientation solver mode changed")
        for joint_name in ARM_JOINTS:
            if self.links[kin.idx[joint_name]].bounds != kin._trust_bounds(
                joint_name, q0, solver.max_joint_step_rad
            ):
                raise ValueError("numerical solve does not use the frozen trust bounds")
        calls.append(
            {
                "case": "B" if not calls else "C",
                "target_position_m": np.asarray(args[0]).tolist(),
                "kwargs": {
                    k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
                    for k, v in kwargs.items()
                },
            }
        )
        write(out / f"solve-{len(calls)}-started.json", calls[-1])
        return original_ik(self, *args, **kwargs)

    Chain.inverse_kinematics = counted_ik
    results = {"A": {"accepted": True, "joints": q_star, "ik_calls": 0}}
    vectors = {"A": q_a}
    try:
        for name, action in (("B", action_b), ("C", actions[0])):
            before = len(calls)
            try:
                decoded = adapter.decode(action, state)
            except (ValueError, RuntimeError) as error:
                write(
                    out / f"case-{name}-rejected.json",
                    {
                        "accepted": False,
                        "error": str(error),
                        "actual_ik_calls": len(calls) - before,
                    },
                )
                raise
            if len(calls) != before + 1:
                raise RuntimeError("decoder did not execute exactly one solve")
            q = q0.copy()
            for n, v in decoded.joint_target.q.items():
                q[kin.idx[n]] = v
            vectors[name] = q
            actual = kin.forward(q)
            desired = np.eye(4)
            desired[:3, 3] = decoded.target_pose.position.as_array()
            desired[:3, :3] = quaternion_matrix(decoded.target_pose.orientation)
            if name == "B":
                np.testing.assert_allclose(desired, fk_a, rtol=0, atol=atol)
            results[name] = {
                "accepted": True,
                "ik_calls": 1,
                "joints": decoded.joint_target.q,
                "physical_input_action": np.asarray(action).tolist(),
                "target_matrix": desired.tolist(),
                "achieved_fk_matrix": actual.tolist(),
                "translation_error_vector_m": (actual[:3, 3] - desired[:3, 3]).tolist(),
                "translation_residual_m": decoded.position_error_m,
                "orientation_error_vector_base_rad": Rotation.from_matrix(
                    actual[:3, :3] @ desired[:3, :3].T
                )
                .as_rotvec()
                .tolist(),
                "orientation_residual_rad": decoded.orientation_error_rad,
            }
            write(out / f"case-{name}.json", results[name])
    finally:
        Chain.inverse_kinematics = original_ik
        write(
            out / "solve-ledger.json",
            {"actual_inverse_kinematics_calls": len(calls), "calls": calls},
        )
    if len(calls) != 2:
        raise RuntimeError("A-D1 requires exactly two solves")
    jac = geometric_jacobian(kin, q0, tuple(ARM_JOINTS))
    # Independent centered-FK derivative validation; no extra IK or policy calls.
    h = float(np.cbrt(np.finfo(float).eps))
    columns = []
    for n in ARM_JOINTS:
        left = q0.copy()
        right = q0.copy()
        left[kin.idx[n]] -= h
        right[kin.idx[n]] += h
        fk_left = kin.forward(left)
        fk_right = kin.forward(right)
        columns.append(
            np.r_[
                (fk_right[:3, 3] - fk_left[:3, 3]) / (2 * h),
                Rotation.from_matrix(fk_right[:3, :3] @ fk_left[:3, :3].T).as_rotvec() / (2 * h),
            ]
        )
    finite = np.column_stack(columns)
    np.testing.assert_allclose(jac, finite, rtol=1e-5, atol=5e-8)
    indices = [kin.idx[n] for n in ARM_JOINTS]
    differences = {}
    for label, to, from_ in (
        ("A_minus_q0", "A", None),
        ("B_minus_A", "B", "A"),
        ("C_minus_B", "C", "B"),
        ("C_minus_A", "C", "A"),
    ):
        delta = (vectors[to] - (q0 if from_ is None else vectors[from_]))[indices]
        differences[label] = {
            "delta_joint_rad": delta.tolist(),
            "max_abs_joint_rad": float(np.max(np.abs(delta))),
            "l2_joint_rad": float(np.linalg.norm(delta)),
            "q0_local_decomposition": decompose(jac, delta),
        }
    for name, q in vectors.items():
        margins = {
            n: {
                "position_bounds": list(kin.bounds(n)),
                "trust_bounds": list(kin._trust_bounds(n, q0, solver.max_joint_step_rad)),
                "delta_from_q0_rad": float(q[kin.idx[n]] - q0[kin.idx[n]]),
            }
            for n in ARM_JOINTS
        }
        results[name]["bounds"] = margins
        results[name]["fk_matrix"] = kin.forward(q).tolist()
    for path, hsh in inputs.items():
        checked(Path(path), hsh)
    forbidden = [
        n
        for n in sys.modules
        if n == "controller" or n == "torch" or n.startswith("lerobot.") or n == "mujoco"
    ]
    if forbidden:
        raise RuntimeError(f"forbidden runtime imports: {forbidden}")
    result = {
        "identity": IDENTITY,
        "status": "completed",
        "cases": results,
        "q0": joints,
        "starting_pose_matrix": recorded.tolist(),
        "saved_action0": actions[0].tolist(),
        "native_expert_action0": row["action"],
        "joint_differences": differences,
        "jacobian": {
            "reference": "q0 only",
            "matrix": jac.tolist(),
            "row_units": ["m/rad"] * 3 + ["rad/rad"] * 3,
            "translation_singular_values_m_per_rad": np.linalg.svd(
                jac[:3], compute_uv=False
            ).tolist(),
            "rotation_singular_values": np.linalg.svd(jac[3:], compute_uv=False).tolist(),
            "mixed_SI_singular_values": np.linalg.svd(jac, compute_uv=False).tolist(),
            "finite_difference_step_rad": h,
            "finite_difference_check_max_error": float(np.max(np.abs(jac - finite))),
            "warning": (
                "Combined spectrum depends on translation/rotation row units; "
                "no calibrated condition-number claim. Null decomposition is local at q0, "
                "not a finite-motion causal proof."
            ),
        },
        "safety": {
            "position_tolerance_m": adapter.position_tolerance_m,
            "orientation_tolerance_rad": adapter.orientation_tolerance_rad,
            "trust_region_rad": solver.max_joint_step_rad,
            "gripper_bounds_m": list(adapter.safety_monitor.limits.gripper_opening),
            "command_horizon_s": adapter.safety_monitor.limits.command_horizon,
        },
        "actual_numerical_ik_calls": len(calls),
        "inference_calls": 0,
        "simulator_starts": 0,
        "actuator_commands": 0,
        "scored_evaluations": 0,
        "original_hashes_unchanged": True,
        "original_gate": "FAILED_0_of_24",
        "primary": "scripted",
    }
    write(out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    """Explicit two-stage interface; never auto-runs a simulation or model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def guard(event: str, values: tuple[Any, ...]) -> None:
        if event in ("subprocess.Popen", "os.system", "os.posix_spawn", "socket.connect"):
            raise RuntimeError(f"A-D1 prohibits process/network launch: {event}")
        if event == "import" and str(values[0]).split(".")[0] in {
            "controller",
            "torch",
            "lerobot",
            "mujoco",
            "rospy",
            "rclpy",
        }:
            raise ImportError("A-D1 prohibits simulator and neural runtime imports")

    sys.addaudithook(guard)
    try:
        (prepare if args.mode == "prepare" else run)(
            args.project.resolve(), args.evidence.resolve(), args.out.resolve()
        )
    except Exception as error:
        if args.out.is_dir() and not (args.out / "STOP.json").exists():
            write(
                args.out / "STOP.json",
                {"status": "STOPPED", "exception": type(error).__name__, "reason": str(error)},
            )
        raise


if __name__ == "__main__":
    main()
