"""Campaign metrics and statistics for ADR-0029 (contract: docs/STATISTICS.md).

Reporting rules encoded here:

- The eight **Table 5.1** metrics of the submitted WorldMind Phase I report are preserved
  verbatim once the report enumerates them (report PDF pending — HUMAN STEP). This module
  provides the statistical primitives and the episode/aggregation plumbing those metrics
  are computed with; nothing here replaces a report metric.
- Collision is reported through **two separate channels that must never merge**: the
  displacement (>5 mm) / tilt (>5 deg) **PROXY** (frozen thresholds in
  ``configs/eval.yaml``, labelled a proxy everywhere) and **actual** contact-node
  detection recorded from the simulator (``episode-XX.contacts.jsonl``).
- **FK clearance** is its own standalone measurement (minimum surface clearance of
  supplied FK points), not a collision proxy substitute.
- Continuous-target errors use **RMSE/MAE**; probabilistic binary outcomes use the
  **Brier score**; the two are never conflated.
- The **pairing unit is (scene, seed)**. Episode-level paired results are preserved, but
  repeated seeds within a scene are correlated, so significance is assessed with a
  scene-level (cluster-aware) exact sign test; episode-level McNemar is reported as
  descriptive and labelled with its independence-assumption limitation.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.eval.settings import EvalSettings, load_eval_settings

# 97.5th percentile of the standard normal (two-sided 95% interval).
Z_95 = 1.959963984540054

PAIRING_UNIT = "(scene, seed)"


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion (better small-n coverage than Wald)."""
    if total <= 0:
        raise ValueError("total must be positive.")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, {total}]; got {successes}.")
    if z <= 0.0:
        raise ValueError("z must be positive.")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (max(0.0, center - margin), min(1.0, center + margin))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Arithmetic mean and sample standard deviation (ddof=1).

    The standard deviation of a single sample is undefined and reported as NaN.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must not be empty.")
    mean = float(array.mean())
    if array.size < 2:
        return (mean, math.nan)
    return (mean, float(array.std(ddof=1)))


def rmse(errors: Sequence[float]) -> float:
    """Root-mean-square error for **continuous** targets (never for probabilities)."""
    array = np.asarray(list(errors), dtype=np.float64)
    if array.size == 0:
        raise ValueError("errors must not be empty.")
    return float(np.sqrt(np.mean(np.square(array))))


def mae(errors: Sequence[float]) -> float:
    """Mean absolute error for **continuous** targets (never for probabilities)."""
    array = np.asarray(list(errors), dtype=np.float64)
    if array.size == 0:
        raise ValueError("errors must not be empty.")
    return float(np.mean(np.abs(array)))


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool | int]) -> float:
    """Brier score (mean squared error) for **probabilistic binary** outcomes.

    Distinct from :func:`rmse`: inputs here are predicted probabilities against observed
    0/1 outcomes. The two are never reported interchangeably.
    """
    probs = np.asarray(list(probabilities), dtype=np.float64)
    observed = np.asarray([int(value) for value in outcomes], dtype=np.int64)
    if probs.size == 0 or observed.size == 0:
        raise ValueError("probabilities and outcomes must not be empty.")
    if probs.shape != observed.shape:
        raise ValueError(
            f"length mismatch: {probs.size} probabilities vs {observed.size} outcomes."
        )
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be in [0, 1].")
    if np.any((observed < 0) | (observed > 1)):
        raise ValueError("outcomes must be binary (0/1).")
    return float(np.mean(np.square(probs - observed)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar/sign-test p-value on ``b`` vs ``c`` discordant units.

    ``b`` = units where A succeeded and B failed; ``c`` = the reverse. Under the null the
    discordant split is Binomial(b+c, 0.5); the two-sided p-value doubles the smaller
    tail (capped at 1). With correlated units (repeated seeds within a scene) the
    episode-level value is anti-conservative — see docs/STATISTICS.md; prefer the
    scene-level call for inference.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative.")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# Episode records (loader for the EpisodeRecorder file contract, ADR-0031)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRecord:
    """One recorded episode: metadata plus the numeric streams."""

    meta: dict[str, Any]
    t: np.ndarray
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    gripper: np.ndarray
    ee_xyz: np.ndarray
    base_pose: np.ndarray
    gt_names: tuple[str, ...]
    gt_xyz: np.ndarray
    gt_quat: np.ndarray
    contacts: tuple[dict[str, Any], ...]


_REQUIRED_ARRAYS = (
    "t",
    "joint_names",
    "joint_pos",
    "joint_vel",
    "gripper",
    "ee_xyz",
    "base_pose",
    "gt_names",
    "gt_xyz",
    "gt_quat",
)


def load_episode(meta_path: Path) -> EpisodeRecord:
    """Load ``episode-XX.meta.json`` with its ``.steps.npz`` and ``.contacts.jsonl``."""
    meta_path = Path(meta_path)
    if not meta_path.name.endswith(".meta.json"):
        raise ValueError(f"not an episode meta file: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stem = meta_path.name[: -len(".meta.json")]
    steps_path = meta_path.with_name(f"{stem}.steps.npz")
    contacts_path = meta_path.with_name(f"{stem}.contacts.jsonl")

    with np.load(steps_path) as data:
        missing = [key for key in _REQUIRED_ARRAYS if key not in data.files]
        if missing:
            raise ValueError(f"{steps_path}: missing arrays {missing}.")
        arrays = {key: np.array(data[key]) for key in _REQUIRED_ARRAYS}

    contacts: list[dict[str, Any]] = []
    if contacts_path.is_file():
        for line in contacts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or "t" not in row or "nodes" not in row:
                    raise ValueError(f"{contacts_path}: malformed contact row: {row!r}")
                contacts.append(row)

    return EpisodeRecord(
        meta=meta,
        t=arrays["t"],
        joint_names=tuple(str(name) for name in arrays["joint_names"]),
        joint_pos=arrays["joint_pos"],
        joint_vel=arrays["joint_vel"],
        gripper=arrays["gripper"],
        ee_xyz=arrays["ee_xyz"],
        base_pose=arrays["base_pose"],
        gt_names=tuple(str(name) for name in arrays["gt_names"]),
        gt_xyz=arrays["gt_xyz"],
        gt_quat=arrays["gt_quat"],
        contacts=tuple(contacts),
    )


def scene_id(meta: Mapping[str, Any]) -> str:
    """Scene identity for pairing: the scenario name without its ``-s<seed>`` suffix."""
    scenario = meta.get("scenario")
    name = ""
    if isinstance(scenario, Mapping):
        name = str(scenario.get("name", ""))
    seed = meta.get("reset_seed")
    if seed is None:
        seed = meta.get("seed")
    if seed is not None and name.endswith(f"-s{int(seed)}"):
        return name[: -len(f"-s{int(seed)}")]
    return name


# ---------------------------------------------------------------------------
# Collision PROXY (displacement/tilt) — separate from actual contact detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectProxy:
    """Per-object displacement/tilt between its reset pose and later poses."""

    name: str
    max_displacement_m: float
    final_displacement_m: float
    max_tilt_deg: float
    final_tilt_deg: float
    displacement_hit: bool
    tilt_hit: bool


@dataclass(frozen=True)
class CollisionProxy:
    """ADR-0029 collision **PROXY**: object displacement (>5 mm) / tilt (>5 deg).

    This is NOT ground-truth collision detection; actual contacts are recorded
    separately (``contacts.jsonl``). Both the max excursion from the reset pose and the
    final (reset -> last) displacement are reported; ``hit`` uses the max excursion so a
    transient knock is not missed.
    """

    objects: tuple[ObjectProxy, ...]
    max_displacement_threshold_m: float
    max_tilt_threshold_deg: float
    max_displacement_m: float | None
    max_tilt_deg: float | None
    hit: bool


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def tilt_angle_deg(from_quat: Sequence[float], to_quat: Sequence[float]) -> float:
    """Angle (deg) of the rotation taking ``from_quat`` to ``to_quat`` (x, y, z, w)."""
    relative = _quat_multiply(
        np.asarray(to_quat, dtype=np.float64), _quat_conjugate(np.asarray(from_quat))
    )
    vector_norm = float(np.linalg.norm(relative[:3]))
    scalar = abs(float(relative[3]))
    return math.degrees(2.0 * math.atan2(vector_norm, scalar))


def collision_proxy(
    record: EpisodeRecord,
    *,
    max_displacement_m: float,
    max_tilt_deg: float,
) -> CollisionProxy:
    """Compute the displacement/tilt proxy for every tracked object in ``record``."""
    if max_displacement_m <= 0.0 or max_tilt_deg <= 0.0:
        raise ValueError("proxy thresholds must be positive.")
    objects: list[ObjectProxy] = []
    for index, name in enumerate(record.gt_names):
        positions = record.gt_xyz[:, index, :]
        valid = ~np.isnan(positions[:, 0])
        if not np.any(valid):
            continue
        valid_rows = np.flatnonzero(valid)
        reset_position = positions[valid_rows[0]]
        displacements = [
            float(np.linalg.norm(positions[row] - reset_position)) for row in valid_rows
        ]

        quaternions = record.gt_quat[:, index, :]
        quat_valid = ~np.isnan(quaternions[:, 0])
        tilts: list[float] = []
        if np.any(quat_valid):
            quat_rows = np.flatnonzero(quat_valid)
            reset_quat = quaternions[quat_rows[0]]
            tilts = [tilt_angle_deg(reset_quat, quaternions[row]) for row in quat_rows]

        max_displacement = max(displacements)
        final_displacement = displacements[-1]
        max_tilt = max(tilts) if tilts else 0.0
        final_tilt = tilts[-1] if tilts else 0.0
        displacement_hit = max_displacement > max_displacement_m
        tilt_hit = max_tilt > max_tilt_deg
        objects.append(
            ObjectProxy(
                name=name,
                max_displacement_m=max_displacement,
                final_displacement_m=final_displacement,
                max_tilt_deg=max_tilt,
                final_tilt_deg=final_tilt,
                displacement_hit=displacement_hit,
                tilt_hit=tilt_hit,
            )
        )

    if not objects:
        return CollisionProxy(
            objects=(),
            max_displacement_threshold_m=max_displacement_m,
            max_tilt_threshold_deg=max_tilt_deg,
            max_displacement_m=None,
            max_tilt_deg=None,
            hit=False,
        )
    return CollisionProxy(
        objects=tuple(objects),
        max_displacement_threshold_m=max_displacement_m,
        max_tilt_threshold_deg=max_tilt_deg,
        max_displacement_m=max(obj.max_displacement_m for obj in objects),
        max_tilt_deg=max(obj.max_tilt_deg for obj in objects),
        hit=any(obj.displacement_hit or obj.tilt_hit for obj in objects),
    )


# ---------------------------------------------------------------------------
# FK clearance — standalone measurement (pure geometry; no kinematics import)
# ---------------------------------------------------------------------------


def min_fk_clearance_m(
    fk_points: Sequence[tuple[str, Sequence[float], float]],
    objects: Sequence[tuple[str, Sequence[float], float]],
) -> float | None:
    """Minimum surface-to-surface clearance (m) between FK points and object spheres.

    Each entry is ``(name, xyz, radius)``; negative values mean penetration. Returns
    ``None`` when either side is empty. This is a standalone measurement, deliberately
    separate from the displacement/tilt proxy.
    """
    if not fk_points or not objects:
        return None
    best = math.inf
    for _point_name, point, point_radius in fk_points:
        point_array = np.asarray(point, dtype=np.float64)
        if point_array.shape != (3,):
            raise ValueError(f"FK point {_point_name!r} must be an xyz triple.")
        for _object_name, object_xyz, object_radius in objects:
            object_array = np.asarray(object_xyz, dtype=np.float64)
            if object_array.shape != (3,):
                raise ValueError(f"object {_object_name!r} must be an xyz triple.")
            distance = (
                float(np.linalg.norm(point_array - object_array))
                - float(point_radius)
                - float(object_radius)
            )
            if distance < best:
                best = distance
    return best


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeMetrics:
    """One episode's metric row (all values MEASURED from the recording/outcome)."""

    scene: str
    seed: int
    arm: str
    success: bool
    failure_state: str | None
    failure_reason: str | None
    final_target_distance_m: float | None
    can_height_error_m: float | None
    can_tilt_deg: float | None
    rejections: int | None
    proxy_hit: bool
    proxy_max_displacement_m: float | None
    proxy_max_tilt_deg: float | None
    contact_steps: int
    contact_nodes: tuple[str, ...]
    fk_min_clearance_m: float | None


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def compute_episode_metrics(
    record: EpisodeRecord, settings: EvalSettings | None = None
) -> EpisodeMetrics:
    """Derive one episode's metrics from its recording (fail-loud on missing outcome)."""
    resolved = settings or load_eval_settings()
    outcome = record.meta.get("outcome")
    if not isinstance(outcome, Mapping) or "success" not in outcome:
        raise ValueError("episode recording has no outcome with a 'success' flag.")

    proxy = collision_proxy(
        record,
        max_displacement_m=resolved.collision_proxy_max_displacement_m,
        max_tilt_deg=resolved.collision_proxy_max_tilt_deg,
    )
    contact_nodes = sorted({str(node) for row in record.contacts for node in row.get("nodes", [])})
    fk_clearance = outcome.get("fk_min_clearance_m")
    seed = record.meta.get("reset_seed")
    if seed is None:
        seed = record.meta.get("seed", 0)

    return EpisodeMetrics(
        scene=scene_id(record.meta),
        seed=int(seed),
        arm=str(record.meta.get("pipeline", "unknown")),
        success=bool(outcome["success"]),
        failure_state=_optional_str(outcome.get("failure_state")),
        failure_reason=_optional_str(outcome.get("failure_reason")),
        final_target_distance_m=_optional_float(outcome.get("final_target_distance")),
        can_height_error_m=_optional_float(outcome.get("can_height_error_m")),
        can_tilt_deg=_optional_float(outcome.get("can_tilt_deg")),
        rejections=None if outcome.get("rejections") is None else int(outcome["rejections"]),
        proxy_hit=proxy.hit,
        proxy_max_displacement_m=proxy.max_displacement_m,
        proxy_max_tilt_deg=proxy.max_tilt_deg,
        contact_steps=len(record.contacts),
        contact_nodes=tuple(contact_nodes),
        fk_min_clearance_m=None if fk_clearance is None else float(fk_clearance),
    )


def metrics_to_row(metrics: EpisodeMetrics) -> dict[str, object]:
    """Flatten :class:`EpisodeMetrics` into a CSV/JSON-friendly row."""
    row: dict[str, object] = asdict(metrics)
    row["contact_nodes"] = ";".join(metrics.contact_nodes)
    return row


# ---------------------------------------------------------------------------
# Aggregation: success summaries and paired (scene, seed) comparisons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuccessSummary:
    """Success count, rate, and Wilson 95% interval."""

    successes: int
    total: int
    rate: float
    ci_low: float
    ci_high: float


def summarize_success(flags: Sequence[bool]) -> SuccessSummary:
    """Aggregate binary success flags with a Wilson interval (no single-run claims)."""
    if not flags:
        raise ValueError("flags must not be empty.")
    successes = sum(1 for flag in flags if flag)
    total = len(flags)
    low, high = wilson_interval(successes, total)
    return SuccessSummary(
        successes=successes, total=total, rate=successes / total, ci_low=low, ci_high=high
    )


@dataclass(frozen=True)
class ScenePairRow:
    """Per-scene paired outcomes (preserves within-scene seed variation)."""

    scene: str
    seeds: tuple[int, ...]
    n: int
    a_successes: int
    b_successes: int
    rate_a: float
    rate_b: float
    std_a: float
    std_b: float
    b_discordant: int
    c_discordant: int


@dataclass(frozen=True)
class PairedComparison:
    """Paired A-vs-B comparison with the pairing unit and both analysis levels.

    ``episode_mcnemar_p`` treats every (scene, seed) pair as independent — that
    assumption is violated by repeated seeds within a scene, so the value is descriptive
    and anti-conservative. ``scene_sign_p`` is the cluster-aware exact sign test whose
    independent units are scenes (``independent_units``).
    """

    pairing_unit: str
    episode_pairs: int
    episode_b: int
    episode_c: int
    episode_mcnemar_p: float
    independent_units: int
    scene_b: int
    scene_c: int
    scene_sign_p: float
    rows: tuple[ScenePairRow, ...]


def paired_comparison(
    a: Mapping[tuple[str, int], bool],
    b: Mapping[tuple[str, int], bool],
) -> PairedComparison:
    """Compare two arms keyed by ``(scene, seed)``; episode pairs preserved, scenes clustered."""
    if not a:
        raise ValueError("paired comparison needs at least one (scene, seed) pair.")
    if set(a) != set(b):
        missing_in_b = sorted(set(a) - set(b))
        missing_in_a = sorted(set(b) - set(a))
        raise ValueError(
            f"arms must share identical (scene, seed) keys; "
            f"missing in B={missing_in_b}, missing in A={missing_in_a}."
        )

    by_scene: dict[str, list[tuple[int, bool, bool]]] = {}
    for (scene, seed), outcome_a in a.items():
        by_scene.setdefault(scene, []).append((seed, bool(outcome_a), bool(b[(scene, seed)])))

    episode_b = sum(
        1
        for scene_pairs in by_scene.values()
        for _seed, outcome_a, outcome_b in scene_pairs
        if outcome_a and not outcome_b
    )
    episode_c = sum(
        1
        for scene_pairs in by_scene.values()
        for _seed, outcome_a, outcome_b in scene_pairs
        if outcome_b and not outcome_a
    )
    episode_pairs = sum(len(scene_pairs) for scene_pairs in by_scene.values())

    rows: list[ScenePairRow] = []
    scene_b = 0
    scene_c = 0
    for scene in sorted(by_scene):
        scene_pairs = sorted(by_scene[scene])
        seeds = tuple(seed for seed, _outcome_a, _outcome_b in scene_pairs)
        flags_a = [outcome_a for _seed, outcome_a, _outcome_b in scene_pairs]
        flags_b = [outcome_b for _seed, _outcome_a, outcome_b in scene_pairs]
        rate_a = sum(1 for flag in flags_a if flag) / len(flags_a)
        rate_b = sum(1 for flag in flags_b if flag) / len(flags_b)
        _mean_a, std_a = mean_std([1.0 if flag else 0.0 for flag in flags_a])
        _mean_b, std_b = mean_std([1.0 if flag else 0.0 for flag in flags_b])
        b_discordant = sum(
            1
            for outcome_a, outcome_b in zip(flags_a, flags_b, strict=True)
            if outcome_a and not outcome_b
        )
        c_discordant = sum(
            1
            for outcome_a, outcome_b in zip(flags_a, flags_b, strict=True)
            if outcome_b and not outcome_a
        )
        if rate_a > rate_b:
            scene_b += 1
        elif rate_b > rate_a:
            scene_c += 1
        rows.append(
            ScenePairRow(
                scene=scene,
                seeds=seeds,
                n=len(scene_pairs),
                a_successes=sum(1 for flag in flags_a if flag),
                b_successes=sum(1 for flag in flags_b if flag),
                rate_a=rate_a,
                rate_b=rate_b,
                std_a=std_a,
                std_b=std_b,
                b_discordant=b_discordant,
                c_discordant=c_discordant,
            )
        )

    return PairedComparison(
        pairing_unit=PAIRING_UNIT,
        episode_pairs=episode_pairs,
        episode_b=episode_b,
        episode_c=episode_c,
        episode_mcnemar_p=mcnemar_exact(episode_b, episode_c),
        independent_units=len(rows),
        scene_b=scene_b,
        scene_c=scene_c,
        scene_sign_p=mcnemar_exact(scene_b, scene_c),
        rows=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write dict rows to CSV (column order = first-seen key order across rows)."""
    if not rows:
        raise ValueError("no rows to write.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    """JSON encoder for dataclasses, NumPy arrays/scalars, tuples, and Path."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(payload: object, path: Path) -> None:
    """Write JSON (dataclasses/NumPy/tuples handled) with a stable indent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
