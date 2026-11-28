"""Frozen pretrained visual features for V3 offline experiments.

The encoder is loaded lazily and never fine-tuned. Features are cached per
(episode_index, frame_index) so expensive CPU forward passes run once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

DEFAULT_MODEL = "facebook/dinov2-small"


@dataclass(frozen=True)
class EncoderSpec:
    model_name: str
    revision: str | None
    license: str
    hidden_size: int
    patch_size: int
    image_size: int
    num_register_tokens: int
    spatial_hw: tuple[int, int]
    pooled_dim: int
    patch_grid_dim: int
    preprocess: dict[str, Any]


def _load_dinov2(model_name: str = DEFAULT_MODEL):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return torch, processor, model


def describe_encoder(model_name: str = DEFAULT_MODEL) -> EncoderSpec:
    torch, processor, model = _load_dinov2(model_name)
    cfg = model.config
    size = int(getattr(processor, "size", {}).get("shortest_edge", cfg.image_size))
    if isinstance(getattr(processor, "size", None), dict):
        if "height" in processor.size and "width" in processor.size:
            size = int(processor.size["height"])
        elif "shortest_edge" in processor.size:
            size = int(processor.size["shortest_edge"])
    crop = getattr(processor, "crop_size", None) or {}
    image_size = int(crop.get("height", size))
    patch = int(cfg.patch_size)
    spatial = (image_size // patch, image_size // patch)
    revision = getattr(cfg, "_commit_hash", None) or getattr(model, "name_or_path", model_name)
    preprocess = {
        "backend": "transformers.AutoImageProcessor",
        "resize_shortest_edge": size,
        "center_crop": [image_size, image_size],
        "do_rescale": bool(getattr(processor, "do_rescale", True)),
        "rescale_factor": float(getattr(processor, "rescale_factor", 1 / 255)),
        "do_normalize": bool(getattr(processor, "do_normalize", True)),
        "image_mean": list(getattr(processor, "image_mean", [0.485, 0.456, 0.406])),
        "image_std": list(getattr(processor, "image_std", [0.229, 0.224, 0.225])),
        "interpolation": str(getattr(processor, "resample", "bicubic")),
    }
    return EncoderSpec(
        model_name=model_name,
        revision=revision,
        license="apache-2.0",
        hidden_size=int(cfg.hidden_size),
        patch_size=patch,
        image_size=image_size,
        num_register_tokens=int(getattr(cfg, "num_register_tokens", 0)),
        spatial_hw=spatial,
        pooled_dim=int(cfg.hidden_size),
        patch_grid_dim=int(spatial[0] * spatial[1] * cfg.hidden_size),
        preprocess=preprocess,
    )


def _prepare_images(processor, image_bytes_list: list[bytes]):
    from PIL import Image

    images = []
    for value in image_bytes_list:
        with Image.open(BytesIO(value)) as image:
            images.append(image.convert("RGB"))
    inputs = processor(images=images, return_tensors="pt")
    return inputs


def encode_image_bytes(
    image_bytes_list: list[bytes],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Encode JPEG bytes with a frozen DINOv2; returns pooled and spatial features.

    Spatial features retain patch tokens as float16 [N, H, W, C] for localization.
    """
    torch, processor, model = _load_dinov2(model_name)
    model.to(device)
    spec = describe_encoder(model_name)
    pooled = []
    spatial = []
    total_s = 0.0
    n = 0
    with torch.inference_mode():
        for start in range(0, len(image_bytes_list), batch_size):
            batch = image_bytes_list[start : start + batch_size]
            inputs = _prepare_images(processor, batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t0 = perf_counter()
            out = model(**inputs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            total_s += perf_counter() - t0
            tokens = out.last_hidden_state  # [B, 1+reg+HW, C]
            cls = tokens[:, 0]
            reg = spec.num_register_tokens
            patches = tokens[:, 1 + reg :, :]
            h, w = spec.spatial_hw
            patches = patches[:, : h * w, :].reshape(len(batch), h, w, -1)
            pooled.append(cls.cpu().numpy().astype(np.float32))
            spatial.append(patches.cpu().numpy().astype(np.float16))
            n += len(batch)
    pooled_arr = (
        np.concatenate(pooled, axis=0) if pooled else np.zeros((0, spec.pooled_dim), np.float32)
    )
    spatial_arr = (
        np.concatenate(spatial, axis=0)
        if spatial
        else np.zeros((0, *spec.spatial_hw, spec.hidden_size), np.float16)
    )
    return {
        "spec": asdict(spec),
        "pooled": pooled_arr,
        "spatial": spatial_arr,
        "images": n,
        "seconds": total_s,
        "images_per_s": (n / total_s) if total_s > 0 else None,
    }


def feature_cache_path(cache_dir: Path, model_name: str, image_size: int) -> Path:
    safe = model_name.replace("/", "__")
    return cache_dir / f"{safe}-{image_size}.npz"


def save_feature_cache(
    path: Path,
    *,
    keys: np.ndarray,
    pooled: np.ndarray,
    spatial: np.ndarray,
    spec: EncoderSpec,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(asdict(spec), sort_keys=True).encode("utf-8")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        keys=keys.astype(np.int32),
        pooled=pooled.astype(np.float32),
        spatial=spatial.astype(np.float16),
        meta=np.frombuffer(meta, dtype=np.uint8),
    )
    tmp.replace(path)


def load_feature_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["meta"].tolist()).decode("utf-8"))
        return {
            "keys": data["keys"],
            "pooled": data["pooled"],
            "spatial": data["spatial"],
            "meta": meta,
        }


def build_resumable_cache(
    image_bytes_by_key: dict[tuple[int, int], bytes],
    *,
    cache_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    device: str = "cpu",
    checkpoint_every: int = 64,
) -> dict[str, Any]:
    """Encode images into a resumable cache; existing keys are skipped."""
    spec = describe_encoder(model_name)
    existing = load_feature_cache(cache_path)
    done: dict[tuple[int, int], int] = {}
    keys_list: list[tuple[int, int]] = []
    pooled_list: list[np.ndarray] = []
    spatial_list: list[np.ndarray] = []
    if existing and existing.get("meta", {}).get("model_name") == spec.model_name:
        old_keys = [tuple(int(x) for x in row) for row in existing["keys"].tolist()]
        for i, k in enumerate(old_keys):
            if k in image_bytes_by_key and k not in done:
                done[k] = len(keys_list)
                keys_list.append(k)
                pooled_list.append(existing["pooled"][i])
                spatial_list.append(existing["spatial"][i])
    pending = [k for k in image_bytes_by_key.keys() if k not in done]
    t0 = perf_counter()
    for i in range(0, len(pending), batch_size):
        batch_keys = pending[i : i + batch_size]
        batch = [image_bytes_by_key[k] for k in batch_keys]
        out = encode_image_bytes(batch, model_name=model_name, batch_size=batch_size, device=device)
        for j, k in enumerate(batch_keys):
            keys_list.append(k)
            pooled_list.append(out["pooled"][j])
            spatial_list.append(out["spatial"][j])
        checkpoint = (i // batch_size) % max(1, checkpoint_every // max(1, batch_size)) == 0
        if checkpoint or i + batch_size >= len(pending):
            save_feature_cache(
                cache_path,
                keys=np.array(keys_list, dtype=np.int32),
                pooled=np.stack(pooled_list).astype(np.float32),
                spatial=np.stack(spatial_list).astype(np.float16),
                spec=spec,
            )
    if keys_list:
        save_feature_cache(
            cache_path,
            keys=np.array(keys_list, dtype=np.int32),
            pooled=np.stack(pooled_list).astype(np.float32),
            spatial=np.stack(spatial_list).astype(np.float16),
            spec=spec,
        )
    return {
        "cache": str(cache_path),
        "total_requested": len(image_bytes_by_key),
        "newly_encoded": len(pending),
        "cached_total": len(keys_list),
        "seconds": round(perf_counter() - t0, 3),
        "spec": asdict(spec),
    }
