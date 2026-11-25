"""Model transport carries only learned inputs and never runs a simulator or action source."""

import socket
import threading

import numpy as np
import pytest

from vlarobot.vla.model_service import SERVICE, ModelClient, serve_model, validate_inputs
from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES


class Model:
    metadata = {
        "service": SERVICE,
        "camera_key": CAMERA_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "chunk_size": 2,
    }

    def __init__(self):
        self.seed = None
        self.calls = 0

    def reset(self, seed):
        self.seed = seed

    def predict(self, state, rgb, instruction):
        self.calls += 1
        return np.tile(np.r_[state[:6], 0.0225], (2, 1)).astype(np.float32)


def test_rpc_reset_required_input_contract_and_exact_chunk_roundtrip():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    model = Model()
    worker = threading.Thread(target=serve_model, args=(endpoint, model), daemon=True)
    worker.start()
    client = ModelClient(endpoint, timeout_ms=3000)
    payload = {
        "state": np.arange(8, dtype=np.float32),
        "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "instruction": "place the can",
    }
    try:
        with pytest.raises(RuntimeError, match="reset required"):
            client.request("vla_predict", payload)
        client.reset(12)
        assert model.seed == 12
        result = client.request("vla_predict", payload)
        assert result["actions"].shape == (2, 7)
        np.testing.assert_array_equal(result["actions"][0, :6], np.arange(6, dtype=np.float32))
        with pytest.raises(RuntimeError, match="state8"):
            client.request("vla_predict", {**payload, "state": np.ones(9, dtype=np.float32)})
        assert model.calls == 1
    finally:
        client.request("vla_shutdown", {})
        client.close()
        worker.join(timeout=3)
    assert not worker.is_alive()


def test_input_validation_rejects_nan_wrong_rgb_and_missing_instruction():
    valid_state = np.zeros(8, dtype=np.float32)
    valid_image = np.zeros((480, 640, 3), dtype=np.uint8)
    for state, rgb, text in [
        (np.full(8, np.nan, dtype=np.float32), valid_image, "task"),
        (valid_state, np.zeros((480, 640, 3), dtype=np.float32), "task"),
        (valid_state, valid_image, ""),
        ([], valid_image, "task"),
    ]:
        with pytest.raises(ValueError):
            validate_inputs(state, rgb, text)
