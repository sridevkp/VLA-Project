"""Versioned LeRobot 0.4.3 dataset-to-SmolVLA padding-key compatibility."""

from __future__ import annotations

from functools import wraps
from typing import Any


def with_padding_alias(batch: dict[str, Any]) -> dict[str, Any]:
    """Supply SmolVLA's misspelled mask key using the native dataset's unchanged tensor."""
    if "action_is_pad" not in batch:
        return batch
    if "actions_id_pad" in batch and batch["actions_id_pad"] is not batch["action_is_pad"]:
        raise ValueError("ambiguous native and SmolVLA action padding masks")
    return {**batch, "actions_id_pad": batch["action_is_pad"]}


def install_padding_alias(policy_class: Any) -> None:
    """Adapt the input key only; preserve installed loss/noise/reduction/checkpoint logic."""
    original = policy_class.forward
    if getattr(original, "_tiago_padding_alias", False):
        return

    @wraps(original)
    def forward(self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return original(self, with_padding_alias(batch), *args, **kwargs)

    forward.__dict__["_tiago_padding_alias"] = True
    policy_class.forward = forward
