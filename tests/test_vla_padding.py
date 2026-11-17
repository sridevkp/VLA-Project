"""Native tail masks reach installed SmolVLA without changing samples or loss logic."""

import pytest

from vlarobot.vla.padding import install_padding_alias, with_padding_alias


def test_padding_alias_preserves_native_tensor_and_other_fields():
    mask = object()
    batch = {"action_is_pad": mask, "action": object()}
    adapted = with_padding_alias(batch)
    assert adapted["actions_id_pad"] is mask
    assert adapted["action"] is batch["action"]
    assert "actions_id_pad" not in batch
    with pytest.raises(ValueError, match="ambiguous"):
        with_padding_alias({**batch, "actions_id_pad": object()})


def test_forward_key_adapter_is_idempotent_and_preserves_arguments():
    class Policy:
        def forward(self, batch, noise=None, reduction="mean"):
            return batch, noise, reduction

    original = Policy.forward
    install_padding_alias(Policy)
    adapted = Policy.forward
    assert adapted is not original
    install_padding_alias(Policy)
    assert Policy.forward is adapted
    mask = object()
    batch, noise, reduction = Policy().forward({"action_is_pad": mask}, noise=12, reduction="none")
    assert batch["actions_id_pad"] is mask
    assert (noise, reduction) == (12, "none")
