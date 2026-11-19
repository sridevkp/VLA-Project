"""Offline compatibility checks for the narrowed Packet-B collision proposal.

Reads the recorded expert campaign (NPZ steps) and the pinned exclusion manifest. It never
launches Webots, solves IK, applies an action or touches TEST/TRAIN. Results are a
compatibility regression, not a motion certificate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.worldmind_bindings import (
    PACKET_B_GRIPPER_LINKS,
    UNVERIFIED_ARM_LINKS,
    UNVERIFIED_LINK_FRAMES,
    PacketBManifest,
    _full_joint_vector,
    _gripper_box_geometry,
    corrected_kinematics,
)
from vlarobot.core.types import (
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.world_model.arm_geometry import (
    Capsule,
    primitive_clearance,
    segment_segment_distance_batch,
)
from vlarobot.world_model.arm_sweep import urdf_collision_boxes, urdf_collision_capsules
from vlarobot.world_model.rollout import rotate
from vlarobot.world_model.scene_geometry import SceneCollisionCatalog


@dataclass(frozen=True)
class CampaignPairResult:
    """Worst recorded slack for one retained capsule pair."""

    pair: tuple[str, str]
    links: tuple[str, str]
    worst_slack_m: float
    compatible_with_all_steps: bool
    worst_episode: str
    worst_step: int

    def report(self) -> dict[str, object]:
        return {
            "pair": list(self.pair),
            "links": list(self.links),
            "worst_slack_m": self.worst_slack_m,
            "compatible_with_all_steps": self.compatible_with_all_steps,
            "worst_episode": self.worst_episode,
            "worst_step": self.worst_step,
        }


@dataclass(frozen=True)
class PacketBCampaignReport:
    """Aggregate recorded-campaign compatibility of the narrowed self-collision set."""

    evidence_dir: Path
    episodes: int
    steps: int
    required_separation_m: float
    pairs: tuple[CampaignPairResult, ...]

    @property
    def compatible_pairs(self) -> tuple[CampaignPairResult, ...]:
        return tuple(item for item in self.pairs if item.compatible_with_all_steps)

    @property
    def rejected_pairs(self) -> tuple[CampaignPairResult, ...]:
        return tuple(item for item in self.pairs if not item.compatible_with_all_steps)

    @property
    def passes_all(self) -> bool:
        return not self.rejected_pairs

    def report(self) -> dict[str, object]:
        return {
            "status": "diagnostic_proposal",
            "classification": "compatibility_regression_not_a_certificate",
            "evidence_dir": str(self.evidence_dir),
            "episodes": self.episodes,
            "steps": self.steps,
            "required_separation_m": self.required_separation_m,
            "retained_pairs": len(self.pairs),
            "compatible_pairs": len(self.compatible_pairs),
            "rejected_pairs": len(self.rejected_pairs),
            "rejected": [item.report() for item in self.rejected_pairs],
            "worst": [
                item.report()
                for item in sorted(self.pairs, key=lambda item: item.worst_slack_m)[:8]
            ],
        }


def _episode_paths(evidence_dir: Path) -> tuple[Path, ...]:
    campaign = Path(evidence_dir) / "campaign"
    paths = tuple(sorted(campaign.glob("episode-*.steps.npz")))
    if not paths:
        raise FileNotFoundError(f"no campaign step files under {campaign}")
    return paths


PACKET_B_EVIDENCE_PIN_PATH = (
    Path(__file__).resolve().parents[3] / "configs/geometry/packet-b-campaign-evidence-v1.json"
)


def verify_campaign_evidence(
    evidence_dir: Path | None = None,
    *,
    pin_path: Path = PACKET_B_EVIDENCE_PIN_PATH,
) -> dict[str, object]:
    """Check every recorded campaign step file against the committed SHA-256 pin.

    The pin is committed with the manifest so a later measurement can prove it read the same
    bytes the campaign produced. A missing file, an extra file, a changed hash, a changed step
    count or a changed joint-vector layout fails loudly.
    """
    pin = json.loads(Path(pin_path).read_text(encoding="utf-8"))
    rows = pin.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"evidence pin {pin_path} has no files")
    root = Path(evidence_dir) if evidence_dir is not None else Path(str(pin["source"]))
    campaign = root / "campaign"
    expected = {str(row["episode"]): row for row in rows}
    present = {
        path.name.removesuffix(".steps.npz") for path in campaign.glob("episode-*.steps.npz")
    }
    if present != set(expected):
        missing = sorted(set(expected) - present)
        extra = sorted(present - set(expected))
        raise ValueError(f"campaign files do not match the pin: missing={missing} extra={extra}")
    total_steps = 0
    for episode, row in sorted(expected.items()):
        path = campaign / f"{episode}.steps.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"{episode} sha256 {digest} != pinned {row['sha256']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"{episode} byte size changed since the pin")
        with np.load(path) as data:
            steps = int(len(data["joint_pos"]))
            joint_names = [str(name) for name in data["joint_names"]]
        if steps != int(row["steps"]):
            raise ValueError(f"{episode} has {steps} steps, pin records {row['steps']}")
        if joint_names != list(pin["joint_names"]):
            raise ValueError(f"{episode} joint layout changed since the pin")
        total_steps += steps
    if total_steps != int(pin["steps"]):
        raise ValueError(f"campaign totals {total_steps} steps, pin records {pin['steps']}")
    return {
        "status": "verified",
        "pin": str(pin_path),
        "evidence_dir": str(root),
        "files": len(expected),
        "steps": total_steps,
    }


def evaluate_campaign_pair_compatibility(
    manifest: PacketBManifest, *, evidence_dir: Path | None = None
) -> PacketBCampaignReport:
    """Recompute every retained pair's worst slack over every recorded campaign step.

    The formula mirrors :class:`ExactSweptArmCheck`'s self-pair requirement with the recorded
    substep allowance: ``distance - radii - (margin + motion + error)``. A pair is rejected when
    any recorded configuration violates that requirement.
    """
    evidence = Path(evidence_dir or manifest.evidence_dir).expanduser().resolve()
    paths = _episode_paths(evidence)
    if len(paths) != manifest.episodes:
        raise ValueError(f"campaign has {len(paths)} episodes, manifest pins {manifest.episodes}")
    kinematics = corrected_kinematics()
    primitives = urdf_collision_capsules(
        manifest.urdf_path, UNVERIFIED_ARM_LINKS, max_point_speed_m_s=manifest.speed_bound_m_s
    )
    frames = {link: kinematics.idx[UNVERIFIED_LINK_FRAMES[link]] for link in UNVERIFIED_ARM_LINKS}
    names: list[str] = []
    links: list[str] = []
    radii: list[float] = []
    local: list[tuple[np.ndarray, np.ndarray]] = []
    for link in UNVERIFIED_ARM_LINKS:
        for primitive in primitives[link]:
            capsule = primitive.capsule
            names.append(capsule.name)
            links.append(link)
            radii.append(capsule.radius_m)
            local.append((np.asarray(capsule.start), np.asarray(capsule.end)))
    index = {name: position for position, name in enumerate(names)}
    retained = manifest.manifest.retained_pairs
    try:
        first = np.asarray([index[pair[0]] for pair in retained], dtype=int)
        second = np.asarray([index[pair[1]] for pair in retained], dtype=int)
    except KeyError as error:  # pragma: no cover - manifest loader already checks names
        raise ValueError(
            f"retained pair names a capsule that is not in the URDF: {error}"
        ) from None
    local_first = np.asarray([local[position][0] for position in range(len(local))])
    local_second = np.asarray([local[position][1] for position in range(len(local))])
    pair_radius = np.asarray(radii)[first] + np.asarray(radii)[second]
    required = manifest.required_separation_m

    worst = np.full(len(retained), np.inf)
    worst_episode = np.full(len(retained), "", dtype=object)
    worst_step = np.zeros(len(retained), dtype=int)
    steps = 0
    for path in paths:
        episode = np.load(path)
        episode_name = Path(path).stem.replace(".steps", "")
        joint_names = [str(name) for name in episode["joint_names"]]
        for step in range(len(episode["joint_pos"])):
            joint_pos = {
                name: float(value)
                for name, value in zip(joint_names, episode["joint_pos"][step], strict=True)
            }
            transforms = kinematics.chain.forward_kinematics(
                _full_joint_vector(joint_pos), full_kinematics=True
            )
            start = np.empty_like(local_first)
            end = np.empty_like(local_second)
            for link in UNVERIFIED_ARM_LINKS:
                matrix = transforms[frames[link]]
                positions = [position for position, body in enumerate(links) if body == link]
                start[positions] = local_first[positions] @ matrix[:3, :3].T + matrix[:3, 3]
                end[positions] = local_second[positions] @ matrix[:3, :3].T + matrix[:3, 3]
            distance = segment_segment_distance_batch(
                start[first], end[first], start[second], end[second]
            )
            slack = distance - pair_radius - required
            improved = slack < worst
            worst = np.minimum(worst, slack)
            worst_episode[improved] = episode_name
            worst_step[improved] = step
            steps += 1
    if steps != manifest.steps:
        raise ValueError(f"campaign has {steps} steps, manifest pins {manifest.steps}")
    results = tuple(
        CampaignPairResult(
            pair=tuple(retained[position]),
            links=(links[first[position]], links[second[position]]),
            worst_slack_m=float(worst[position]),
            compatible_with_all_steps=bool(worst[position] >= 0.0),
            worst_episode=str(worst_episode[position]),
            worst_step=int(worst_step[position]),
        )
        for position in range(len(retained))
    )
    return PacketBCampaignReport(
        evidence_dir=evidence,
        episodes=len(paths),
        steps=steps,
        required_separation_m=required,
        pairs=results,
    )


@dataclass(frozen=True)
class GripperMountCampaignResult:
    """Worst recorded clearance for one pinned ``front`` housing/arm-capsule pair."""

    pair: tuple[str, str]
    max_clearance_m: float
    required_separation_m: float

    @property
    def unsatisfiable_in_campaign(self) -> bool:
        return self.max_clearance_m < self.required_separation_m

    def report(self) -> dict[str, object]:
        return {
            "pair": list(self.pair),
            "max_campaign_clearance_m": self.max_clearance_m,
            "required_separation_m": self.required_separation_m,
            "unsatisfiable_in_campaign": self.unsatisfiable_in_campaign,
        }


def evaluate_campaign_gripper_mounts(
    manifest: PacketBManifest, *, evidence_dir: Path | None = None
) -> tuple[GripperMountCampaignResult, ...]:
    """Measure every pinned gripper-mount pair's maximum clearance over the campaign.

    A pinned exclusion is only defensible when its *maximum* recorded clearance stays below the
    required separation: no recorded expert configuration could satisfy the check, so the pair
    describes one wrist assembly rather than an avoided collision. This recomputes that
    measurement from the recorded steps; it never reads a data split or launches a simulator.
    """
    evidence = Path(evidence_dir or manifest.evidence_dir).expanduser().resolve()
    paths = _episode_paths(evidence)
    if len(paths) != manifest.episodes:
        raise ValueError(f"campaign has {len(paths)} episodes, manifest pins {manifest.episodes}")
    primitives = urdf_collision_capsules(
        manifest.urdf_path, UNVERIFIED_ARM_LINKS, max_point_speed_m_s=manifest.speed_bound_m_s
    )
    kinematics = corrected_kinematics()
    frames = {link: kinematics.idx[UNVERIFIED_LINK_FRAMES[link]] for link in UNVERIFIED_ARM_LINKS}
    pinned = manifest.gripper_mount_pairs
    if not pinned:
        return ()
    box_specs = urdf_collision_boxes(
        manifest.urdf_path,
        PACKET_B_GRIPPER_LINKS,
        max_point_speed_m_s=manifest.speed_bound_m_s,
        clearance_by_link={link: manifest.margin_m for link in PACKET_B_GRIPPER_LINKS},
    )
    box_geometry, _ = _gripper_box_geometry(box_specs)
    capsule_index = {
        primitive.capsule.name: (link, primitive.capsule)
        for link in UNVERIFIED_ARM_LINKS
        for primitive in primitives[link]
    }
    required = manifest.required_separation_m
    measured = np.full(len(pinned), -np.inf)

    for path in paths:
        episode = np.load(path)
        joint_names = [str(name) for name in episode["joint_names"]]
        for step in range(len(episode["joint_pos"])):
            joint_pos = {
                name: float(value)
                for name, value in zip(joint_names, episode["joint_pos"][step], strict=True)
            }
            transforms = kinematics.chain.forward_kinematics(
                _full_joint_vector(joint_pos), full_kinematics=True
            )
            state = _recorded_geometry_state(episode, step, joint_pos)
            boxes = {box.name: box for box in box_geometry(state)}
            capsules: dict[str, Capsule] = {}
            for item in pinned:
                if item.second in capsules:
                    continue
                link, capsule = capsule_index[item.second]
                matrix = transforms[frames[link]]
                start = matrix[:3, :3] @ np.asarray(capsule.start) + matrix[:3, 3]
                end = matrix[:3, :3] @ np.asarray(capsule.end) + matrix[:3, 3]
                capsules[item.second] = Capsule(
                    capsule.name,
                    tuple(float(value) for value in start),
                    tuple(float(value) for value in end),
                    capsule.radius_m,
                    capsule.max_point_speed_m_s,
                )
            for position, item in enumerate(pinned):
                gap = primitive_clearance(capsules[item.second], boxes[item.first])
                if gap > measured[position]:
                    measured[position] = float(gap)
    results = tuple(
        GripperMountCampaignResult(
            pair=(item.first, item.second),
            max_clearance_m=float(measured[position]),
            required_separation_m=required,
        )
        for position, item in enumerate(pinned)
    )
    return results


def recorded_gripper_mount_clearance(
    manifest: PacketBManifest,
    first: str,
    second: str,
    episode_name: str,
    step: int,
) -> float:
    """Exact clearance of one pinned ``front``/arm-capsule pair at one recorded step."""
    paths = _episode_paths(Path(manifest.evidence_dir).expanduser().resolve())
    match = [path for path in paths if path.stem.replace(".steps", "") == episode_name]
    if len(match) != 1:
        raise ValueError(f"recorded episode {episode_name!r} is missing or ambiguous")
    primitives = urdf_collision_capsules(
        manifest.urdf_path, UNVERIFIED_ARM_LINKS, max_point_speed_m_s=manifest.speed_bound_m_s
    )
    capsule_index = {
        primitive.capsule.name: (link, primitive.capsule)
        for link in UNVERIFIED_ARM_LINKS
        for primitive in primitives[link]
    }
    if second not in capsule_index:
        raise ValueError(f"{second} is not an enumerated arm capsule")
    box_specs = urdf_collision_boxes(
        manifest.urdf_path,
        PACKET_B_GRIPPER_LINKS,
        max_point_speed_m_s=manifest.speed_bound_m_s,
        clearance_by_link={link: manifest.margin_m for link in PACKET_B_GRIPPER_LINKS},
    )
    if first not in {box.name for box in box_specs["front"]}:
        raise ValueError(f"{first} is not an enumerated front solid")
    box_geometry, _ = _gripper_box_geometry(box_specs)
    kinematics = corrected_kinematics()
    frames = {link: kinematics.idx[UNVERIFIED_LINK_FRAMES[link]] for link in UNVERIFIED_ARM_LINKS}
    episode = np.load(match[0])
    joint_names = [str(name) for name in episode["joint_names"]]
    joint_pos = {
        name: float(value)
        for name, value in zip(joint_names, episode["joint_pos"][step], strict=True)
    }
    transforms = kinematics.chain.forward_kinematics(
        _full_joint_vector(joint_pos), full_kinematics=True
    )
    link, capsule = capsule_index[second]
    matrix = transforms[frames[link]]
    start = matrix[:3, :3] @ np.asarray(capsule.start) + matrix[:3, 3]
    end = matrix[:3, :3] @ np.asarray(capsule.end) + matrix[:3, 3]
    moved = Capsule(
        capsule.name,
        tuple(float(value) for value in start),
        tuple(float(value) for value in end),
        capsule.radius_m,
        capsule.max_point_speed_m_s,
    )
    state = _recorded_geometry_state(episode, step, joint_pos)
    boxes = {box.name: box for box in box_geometry(state)}
    return float(primitive_clearance(moved, boxes[first]))


def _recorded_geometry_state(
    episode: Mapping[str, Any], step: int, joint_pos: Mapping[str, float]
) -> WorldState:
    """Minimal WorldState carrying the recorded end-effector and gripper opening."""
    base_pose = np.asarray(episode["base_pose"][step], dtype=float)
    return WorldState(
        scene=SceneState(objects=[]),
        robot=RobotState(
            joint_pos=dict(joint_pos),
            joint_vel={},
            base_pose=Pose2D(float(base_pose[0]), float(base_pose[1]), float(base_pose[2])),
            ee_pose=Pose3D(
                Vec3(*map(float, episode["ee_xyz"][step])),
                tuple(float(value) for value in episode["ee_quat"][step]),
            ),
            gripper_opening=float(episode["gripper"][step]),
            gripper_holding=None,
        ),
    )


def _base_pose(catalog: SceneCollisionCatalog) -> tuple[np.ndarray, np.ndarray]:
    base = catalog.expected_base_pose_world
    rotation = np.column_stack(
        [rotate(np.asarray(base.orientation, dtype=float), axis) for axis in np.eye(3)]
    )
    return rotation, np.asarray(base.position.as_array(), dtype=float)


def recorded_world_state(
    episode: Mapping[str, Any],
    step: int,
    *,
    catalog: SceneCollisionCatalog,
    object_ids: Sequence[str] = ("CAN",),
) -> WorldState:
    """Build one base-frame WorldState from a recorded campaign step (read-only)."""
    joint_names = [str(name) for name in episode["joint_names"]]
    joint_pos = {
        name: float(value)
        for name, value in zip(joint_names, episode["joint_pos"][step], strict=True)
    }
    joint_vel = {
        name: float(value)
        for name, value in zip(joint_names, episode["joint_vel"][step], strict=True)
    }
    base_pose = np.asarray(episode["base_pose"][step], dtype=float)
    if base_pose.shape != (3,) or not np.isfinite(base_pose).all():
        raise ValueError("recorded base pose is malformed")
    rotation_base_world, translation = _base_pose(catalog)
    gt_names = [str(name) for name in episode["gt_names"]]
    objects: list[ObjectState] = []
    envelope = {item.name: item for item in catalog.geometry.carried_objects}
    for object_id in object_ids:
        if object_id not in gt_names:
            raise ValueError(f"recorded step does not contain object {object_id}")
        position = gt_names.index(object_id)
        world_center = np.asarray(episode["gt_xyz"][step][position], dtype=float)
        world_quaternion = np.asarray(episode["gt_quat"][step][position], dtype=float)
        center = rotation_base_world.T @ (world_center - translation)
        rotation_object = np.column_stack([rotate(world_quaternion, axis) for axis in np.eye(3)])
        rotation_base = rotation_base_world.T @ rotation_object
        quaternion = _matrix_quaternion(rotation_base)
        if object_id not in envelope:
            raise ValueError(f"no pinned carried-object envelope for {object_id}")
        item = envelope[object_id]
        objects.append(
            ObjectState(
                object_id=object_id,
                label=object_id,
                pose=Pose3D(Vec3(*map(float, center)), quaternion),
                dimensions=Vec3(2.0 * item.radius_m, 2.0 * item.radius_m, item.height_m),
                confidence=1.0,
            )
        )
    robot = RobotState(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_pose=Pose2D(float(base_pose[0]), float(base_pose[1]), float(base_pose[2])),
        ee_pose=Pose3D(
            Vec3(*map(float, episode["ee_xyz"][step])),
            tuple(float(value) for value in episode["ee_quat"][step]),
        ),
        gripper_opening=float(episode["gripper"][step]),
        gripper_holding=None,
        base_pose_world=catalog.expected_base_pose_world,
    )
    return WorldState(scene=SceneState(objects=objects), robot=robot)


def _matrix_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    values = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate object rotation")
    return tuple(float(value) / norm for value in values)
