"""Fit a V4 CAN localizer from the frozen V3 DINOv2 TRAIN image features."""

import json
from pathlib import Path

import numpy as np

from vlarobot.execution.demo_collection import write_json
from vlarobot.v3.localization import _metrics, _ridge_fit, _ridge_pred


def fit_cached_localizer(dataset: Path, source_manifest: Path, cache: Path, output: Path) -> dict:
    """Align cached image features to the source frames and fit on V4 TRAIN groups."""
    if output.exists():
        raise FileExistsError(output)
    partition = json.loads(dataset.read_text(encoding="utf-8"))
    if partition["split"] != "train":
        raise ValueError("TRAIN partition required")
    fit_groups = set(partition["partition"]["fit_groups"])
    sources = json.loads(source_manifest.read_text(encoding="utf-8"))["sources"]
    with np.load(cache, allow_pickle=False) as archive:
        keys, features = archive["keys"], archive["pooled"]
        encoder = json.loads(archive["meta"].tobytes())
    labels = np.empty((len(keys), 3))
    phases = np.empty(len(keys), dtype="<U24")
    fit = np.zeros(len(keys), dtype=bool)
    maximum_alignment_s = 0.0
    for episode in np.unique(keys[:, 0]):
        source = sources[int(episode)]
        if source["scene"]["metadata"]["split"] != "train":
            raise ValueError("TRAIN visual sources required")
        root = Path(source["raw_dir"])
        mask = keys[:, 0] == episode
        with np.load(root / "episode-00.steps.npz", allow_pickle=False) as steps:
            times = steps["t"][keys[mask, 1]]
        entries = []
        with (root / "policy.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("can_base_xyz_m") is not None and row.get("sim_time_s") is not None:
                    entries.append(
                        (row["sim_time_s"], row["can_base_xyz_m"], row.get("state") or "")
                    )
        trace_times = np.array([t for t, _, _ in entries])
        nearest = np.abs(times[:, None] - trace_times).argmin(axis=1)
        maximum_alignment_s = max(
            maximum_alignment_s, float(np.abs(times - trace_times[nearest]).max())
        )
        labels[mask] = np.array([entries[i][1] for i in nearest])
        phases[mask] = np.array([entries[i][2] for i in nearest])
        fit[mask] = source["scene"]["name"].rsplit("-s", 1)[0] in fit_groups
    model = _ridge_fit(features[fit], labels[fit])
    predicted = _ridge_pred(model, features[~fit])
    errors = np.linalg.norm((predicted - labels[~fit])[:, :2], axis=1)
    held_phases = phases[~fit]
    tails = {}
    for name, mask in (
        ("all", np.ones(len(errors), dtype=bool)),
        ("approach", np.isin(held_phases, ["PREGRASP", "DESCEND"])),
        ("grasp", held_phases == "CLOSE"),
    ):
        if mask.any():
            p50, p90, p95 = np.percentile(errors[mask], [50, 90, 95])
            tails[name] = {"n": int(mask.sum()), "p50_m": p50, "p90_m": p90, "p95_m": p95}
    result = {
        "identity": "v4-frozen-dinov2-cls-can-localizer",
        "encoder": encoder,
        "partition": partition["partition"],
        "fit_n": int(fit.sum()),
        "internal_train_evaluation_n": int((~fit).sum()),
        "internal_train_error": _metrics(predicted, labels[~fit]),
        "internal_train_lateral_error": tails,
        "maximum_label_alignment_s": maximum_alignment_s,
        "model": {key: value.tolist() for key, value in model.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


class DinoCanLocalizer:
    """Live frozen encoder paired with the cached-feature ridge head."""

    def __init__(self, model: Path) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        payload = json.loads(model.read_text(encoding="utf-8"))
        if payload["identity"] != "v4-frozen-dinov2-cls-can-localizer":
            raise ValueError("expected a V4 DINOv2 CAN-localizer artifact")
        encoder = payload["encoder"]
        options = {"revision": encoder["revision"], "local_files_only": True}
        self.processor = AutoImageProcessor.from_pretrained(encoder["model_name"], **options)
        self.encoder = AutoModel.from_pretrained(encoder["model_name"], **options).eval()
        self.torch = torch
        self.ridge = {key: np.asarray(value) for key, value in payload["model"].items()}

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        features = self.processor(images=[rgb], return_tensors="pt")
        with self.torch.inference_mode():
            tokens = self.encoder(**features).last_hidden_state[:, 0].cpu().numpy()
        xyz = _ridge_pred(self.ridge, tokens)[0]
        if not np.isfinite(xyz).all():
            raise ValueError("nonfinite visual CAN estimate")
        return xyz
