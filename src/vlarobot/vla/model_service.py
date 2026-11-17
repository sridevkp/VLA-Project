"""Local model-process RPC: learned inputs only, physical 7-D chunks, no simulator actions."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import zmq

from vlarobot.core.types import Observation
from vlarobot.core.wire import decode_message, encode_message, make_message
from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES, observation_state

SERVICE = "tiago-smolvla-v1"


class ChunkModel(Protocol):
    metadata: dict[str, Any]

    def reset(self, seed: int) -> None: ...
    def predict(self, state: np.ndarray, rgb: np.ndarray, instruction: str) -> np.ndarray: ...


def validate_inputs(state: np.ndarray, rgb: np.ndarray, instruction: str) -> None:
    """Fail before inference if observation keys/dtypes/shapes are incompatible."""
    if (
        not isinstance(state, np.ndarray)
        or not isinstance(rgb, np.ndarray)
        or state.shape != (8,)
        or state.dtype != np.float32
        or not np.all(np.isfinite(state))
        or rgb.shape != (480, 640, 3)
        or rgb.dtype != np.uint8
        or not isinstance(instruction, str)
        or not instruction.strip()
    ):
        raise ValueError("expected finite float32 state8, RGB uint8 480x640, and instruction")


class PortableChunkModel:
    """Strict installed model load in the isolated ML process; never receives ground truth."""

    def __init__(self, export: Path, *, device: str = "cpu", cpu_threads: int = 2) -> None:
        from vlarobot.vla.checkpoint_export import load_portable_policy

        self.torch = import_module("torch")
        self.torch.set_num_threads(cpu_threads)
        self.policy, self.pre, self.post = load_portable_policy(export, device=device)
        config = self.policy.config
        if (
            set(config.input_features) != {"observation.state", CAMERA_KEY}
            or tuple(config.input_features["observation.state"].shape) != (8,)
            or tuple(config.input_features[CAMERA_KEY].shape) != (3, 480, 640)
            or tuple(config.output_features["action"].shape) != (7,)
        ):
            raise ValueError("portable policy is not the verified TIAGo embodiment")
        self.metadata = {
            "service": SERVICE,
            "camera_key": CAMERA_KEY,
            "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES,
            "chunk_size": config.chunk_size,
            "device": device,
            "export_manifest_sha256": hashlib.sha256(
                (export / "export-manifest.json").read_bytes()
            ).hexdigest(),
        }

    def reset(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        self.policy.reset()

    def predict(self, state: np.ndarray, rgb: np.ndarray, instruction: str) -> np.ndarray:
        validate_inputs(state, rgb, instruction)
        batch = self.pre(
            {
                "observation.state": self.torch.from_numpy(state.copy()),
                CAMERA_KEY: self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
                "task": instruction,
            }
        )
        with self.torch.inference_mode():
            actions = self.post(self.policy.predict_action_chunk(batch))
        result = actions.detach().cpu().numpy()[0]
        if result.shape != (self.policy.config.chunk_size, 7) or not np.all(np.isfinite(result)):
            raise ValueError("policy produced an invalid physical action chunk")
        return result


def serve_model(endpoint: str, model: ChunkModel) -> None:
    """Serve one sequential episode stream; seeded reset precedes every episode."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.linger = 0
    socket.bind(endpoint)
    seeded = False
    try:
        while True:
            request = decode_message(socket.recv())
            payload = request["payload"]
            op = request["type"]
            response: dict[str, Any] = {"service": SERVICE}
            error = None
            shutdown = False
            try:
                if payload.get("service") != SERVICE:
                    raise ValueError("model service/version mismatch")
                if op == "vla_ping":
                    response["metadata"] = model.metadata
                elif op == "vla_reset":
                    seed = payload["seed"]
                    if not isinstance(seed, int) or isinstance(seed, bool):
                        raise ValueError("model reset requires explicit integer seed")
                    model.reset(seed)
                    seeded = True
                elif op == "vla_predict":
                    if not seeded:
                        raise ValueError("model reset required before inference")
                    state, rgb, instruction = (
                        payload["state"],
                        payload["rgb"],
                        payload["instruction"],
                    )
                    validate_inputs(state, rgb, instruction)
                    started = perf_counter()
                    chunk = np.asarray(model.predict(state, rgb, instruction))
                    if (
                        chunk.ndim != 2
                        or chunk.shape[1] != 7
                        or chunk.dtype.kind not in "fiu"
                        or not np.all(np.isfinite(chunk))
                    ):
                        raise ValueError("invalid model action chunk")
                    response.update(actions=chunk, inference_wall_s=perf_counter() - started)
                elif op == "vla_shutdown":
                    shutdown = True
                else:
                    raise ValueError("unsupported model request")
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            socket.send(
                encode_message(
                    make_message(
                        "vla_reply",
                        request["seq"],
                        response,
                        sim_time=request["sim_time"],
                        error=error,
                    )
                )
            )
            if shutdown:
                break
    finally:
        socket.close(0)
        context.term()


class ModelClient:
    """Brain-side synchronous client: input construction cannot include oracle scene state."""

    def __init__(self, endpoint: str, *, timeout_ms: int = 300_000) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.sndtimeo = self.socket.rcvtimeo = timeout_ms
        self.socket.connect(endpoint)
        self.seq = 0
        try:
            self.metadata = self.request("vla_ping", {})["metadata"]
        except Exception:
            self.close()
            raise
        if (
            self.metadata["camera_key"] != CAMERA_KEY
            or tuple(self.metadata["action_names"]) != ACTION_NAMES
            or tuple(self.metadata["state_names"]) != STATE_NAMES
        ):
            self.close()
            raise ValueError("model response embodiment mismatch")

    def request(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.seq += 1
        self.socket.send(
            encode_message(make_message(op, self.seq, {"service": SERVICE, **payload}))
        )
        reply = decode_message(self.socket.recv())
        if (
            reply["seq"] != self.seq
            or reply["type"] != "vla_reply"
            or reply["payload"].get("service") != SERVICE
        ):
            raise ValueError("model response identity mismatch")
        if reply["error"]:
            raise RuntimeError(reply["error"])
        return reply["payload"]

    def reset(self, seed: int) -> None:
        self.request("vla_reset", {"seed": seed})

    def predict(
        self, observation: Observation, instruction: str
    ) -> tuple[np.ndarray, dict[str, float]]:
        started = perf_counter()
        result = self.request(
            "vla_predict",
            {
                "state": observation_state(
                    observation.robot.joint_pos, observation.robot.gripper_opening
                ),
                "rgb": observation.rgb,
                "instruction": instruction,
            },
        )
        return np.asarray(result["actions"]), {
            "inference_wall_s": result["inference_wall_s"],
            "roundtrip_wall_s": perf_counter() - started,
        }

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()
