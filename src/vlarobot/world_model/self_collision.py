"""Fail-closed self-collision pair coverage derived from pinned URDF capsule identities.

This module does not certify contact safety. It enumerates every capsule pair that a swept
check must consider and records which pairs a contract would exclude. Adjacent links and
grasp contacts are never excluded automatically: only an owner-reviewed pair list passed by
the caller can remove a pair from the checked set, and the shipped contract supplies none.
"""

from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from vlarobot.world_model.arm_sweep import SweepGeometryError, urdf_collision_capsules


@dataclass(frozen=True)
class SelfCollisionCoverage:
    """Exact capsule-pair coverage of one pinned self-collision model."""

    required_links: tuple[str, ...]
    capsule_names: tuple[str, ...]
    adjacent_link_pairs: tuple[tuple[str, str], ...]
    adjacent_capsule_pairs: tuple[tuple[str, str], ...]
    non_adjacent_capsule_pairs: tuple[tuple[str, str], ...]
    same_link_capsule_pairs: tuple[tuple[str, str], ...]
    gripper_arm_pairs: tuple[tuple[str, str], ...]
    carried_robot_pairs: tuple[tuple[str, str], ...]
    excluded_pairs: tuple[tuple[str, str], ...]
    uncovered_collision_bodies: tuple[str, ...]

    @property
    def all_capsule_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(itertools.combinations(self.capsule_names, 2))

    @property
    def checked_capsule_pairs(self) -> tuple[tuple[str, str], ...]:
        excluded = {frozenset(pair) for pair in self.excluded_pairs}
        return tuple(pair for pair in self.all_capsule_pairs if frozenset(pair) not in excluded)

    @property
    def coverage_proven(self) -> bool:
        return not self.excluded_pairs and not self.uncovered_collision_bodies

    @staticmethod
    def _pair_rows(rows: tuple[tuple[str, str], ...]) -> list[list[str]]:
        return [[first, second] for first, second in rows]

    def report(self) -> dict[str, object]:
        pairs = self._pair_rows
        counts = {
            "required_links": len(self.required_links),
            "capsules": len(self.capsule_names),
            "all_capsule_pairs": len(self.all_capsule_pairs),
            "checked_capsule_pairs": len(self.checked_capsule_pairs),
            "adjacent_link_pairs": len(self.adjacent_link_pairs),
            "adjacent_capsule_pairs": len(self.adjacent_capsule_pairs),
            "non_adjacent_capsule_pairs": len(self.non_adjacent_capsule_pairs),
            "same_link_capsule_pairs": len(self.same_link_capsule_pairs),
            "gripper_arm_pairs": len(self.gripper_arm_pairs),
            "carried_robot_pairs": len(self.carried_robot_pairs),
            "excluded_pairs": len(self.excluded_pairs),
            "uncovered_collision_bodies": len(self.uncovered_collision_bodies),
        }
        return {
            "required_links": list(self.required_links),
            "capsule_names": list(self.capsule_names),
            "adjacent_link_pairs": pairs(self.adjacent_link_pairs),
            "adjacent_capsule_pairs": pairs(self.adjacent_capsule_pairs),
            "non_adjacent_capsule_pairs": pairs(self.non_adjacent_capsule_pairs),
            "same_link_capsule_pairs": pairs(self.same_link_capsule_pairs),
            "gripper_arm_pairs": pairs(self.gripper_arm_pairs),
            "carried_robot_pairs": pairs(self.carried_robot_pairs),
            "excluded_pairs": pairs(self.excluded_pairs),
            "uncovered_collision_bodies": list(self.uncovered_collision_bodies),
            "counts": counts,
            "coverage_proven": self.coverage_proven,
            "policy": (
                "all pinned capsule pairs are checked; adjacency and grasp contact are not "
                "exempt unless an owner-reviewed pair list is supplied explicitly"
            ),
        }


def urdf_collision_body_names(urdf_path: Path) -> tuple[str, ...]:
    """Return every URDF collision body as ``link:index`` in deterministic order."""
    root = ET.parse(Path(urdf_path)).getroot()
    names: list[str] = []
    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if not link_name:
            raise SweepGeometryError("URDF link has no name")
        collisions = link.findall("collision")
        names.extend(f"{link_name}:{index}" for index in range(len(collisions)))
    return tuple(sorted(names))


def urdf_adjacent_link_pairs(
    urdf_path: Path, required_links: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """Return adjacent link pairs induced by URDF joints inside ``required_links``."""
    required = frozenset(required_links)
    if not required or len(required) != len(required_links):
        raise SweepGeometryError("adjacency requires unique required links")
    root = ET.parse(Path(urdf_path)).getroot()
    pairs: set[tuple[str, str]] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise SweepGeometryError("URDF joint has incomplete parent/child")
        first, second = parent.attrib.get("link"), child.attrib.get("link")
        if first in required and second in required and first != second:
            pair: tuple[str, str] = (first, second) if first <= second else (second, first)
            pairs.add(pair)
    return tuple(sorted(pairs))


def _link_of(capsule_name: str) -> str:
    link, separator, index = capsule_name.rpartition(":")
    if not separator or not link or not index.isdigit():
        raise SweepGeometryError(f"unsupported capsule identity: {capsule_name}")
    return link


def build_self_collision_coverage(
    urdf_path: Path,
    required_links: Sequence[str],
    *,
    max_point_speed_m_s: float,
    gripper_name: str | None = None,
    carried_object_id: str | None = None,
    excluded_pairs: frozenset[frozenset[str]] = frozenset(),
) -> SelfCollisionCoverage:
    """Enumerate all self-collision pairs for the pinned arm, gripper and carried object.

    ``excluded_pairs`` is reported, never assumed. The checker receives an empty exclusion set
    from the blocked production contract, so every adjacent and non-adjacent capsule pair is
    conservatively checked.
    """
    if not isinstance(max_point_speed_m_s, (int, float)) or isinstance(max_point_speed_m_s, bool):
        raise SweepGeometryError("point-speed bound must be a finite positive number")
    if not math.isfinite(float(max_point_speed_m_s)) or float(max_point_speed_m_s) <= 0.0:
        raise SweepGeometryError("point-speed bound must be a finite positive number")
    primitives = urdf_collision_capsules(
        Path(urdf_path), tuple(required_links), max_point_speed_m_s=float(max_point_speed_m_s)
    )
    capsule_names = tuple(
        sorted(primitive.capsule.name for group in primitives.values() for primitive in group)
    )
    adjacent_links = urdf_adjacent_link_pairs(urdf_path, required_links)
    adjacent_link_set = {frozenset(pair) for pair in adjacent_links}

    adjacent: list[tuple[str, str]] = []
    non_adjacent: list[tuple[str, str]] = []
    same_link: list[tuple[str, str]] = []
    for first, second in itertools.combinations(capsule_names, 2):
        first_link, second_link = _link_of(first), _link_of(second)
        if first_link == second_link:
            same_link.append((first, second))
        elif frozenset((first_link, second_link)) in adjacent_link_set:
            adjacent.append((first, second))
        else:
            non_adjacent.append((first, second))

    all_names = capsule_names + ((gripper_name,) if gripper_name else ())
    gripper_pairs = tuple((name, gripper_name) for name in capsule_names) if gripper_name else ()
    carried_pairs = (
        tuple((carried_object_id, name) for name in all_names) if carried_object_id else ()
    )
    excluded_normalized = tuple(
        sorted((sorted(pair)[0], sorted(pair)[1]) for pair in excluded_pairs if len(pair) == 2)
    )
    known = set(all_names) | ({carried_object_id} if carried_object_id else set())
    if any(name not in known for pair in excluded_normalized for name in pair):
        raise SweepGeometryError("excluded self-collision pair names are not in the model")

    all_bodies = set(urdf_collision_body_names(urdf_path))
    uncovered = tuple(sorted(all_bodies - set(capsule_names)))
    return SelfCollisionCoverage(
        required_links=tuple(required_links),
        capsule_names=capsule_names,
        adjacent_link_pairs=adjacent_links,
        adjacent_capsule_pairs=tuple(sorted(adjacent)),
        non_adjacent_capsule_pairs=tuple(sorted(non_adjacent)),
        same_link_capsule_pairs=tuple(sorted(same_link)),
        gripper_arm_pairs=gripper_pairs,
        carried_robot_pairs=carried_pairs,
        excluded_pairs=excluded_normalized,
        uncovered_collision_bodies=uncovered,
    )


def validate_contract_pair_coverage(
    coverage: SelfCollisionCoverage, excluded_pairs: frozenset[frozenset[str]]
) -> bool:
    """Return ``True`` only when a contract checks every enumerated capsule pair."""
    expected = {frozenset(pair) for pair in coverage.all_capsule_pairs}
    expected |= {frozenset(pair) for pair in coverage.gripper_arm_pairs}
    excluded = {frozenset(pair) for pair in excluded_pairs}
    return not excluded and not (expected & excluded) and coverage.coverage_proven


__all__ = [
    "SelfCollisionCoverage",
    "build_self_collision_coverage",
    "urdf_adjacent_link_pairs",
    "urdf_collision_body_names",
    "validate_contract_pair_coverage",
]
