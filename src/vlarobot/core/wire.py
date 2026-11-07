"""Versioned msgpack serialization for the simulator/brain protocol."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import Any

import msgpack
import numpy as np

from vlarobot.core import types

PROTOCOL_VERSION = 1
_ARRAY_EXT = 1
_TYPE_REGISTRY = {
    item.__name__: item
    for item in vars(types).values()
    if isinstance(item, type) and is_dataclass(item)
}


class ProtocolError(ValueError):
    """Raised when a wire message is malformed or incompatible."""


class ProtocolVersionError(ProtocolError):
    """Raised when peers use different wire protocol versions."""


def _encode_array(array: np.ndarray) -> msgpack.ExtType:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject or contiguous.dtype.kind not in "biuf":
        raise ProtocolError(f"Unsupported NumPy dtype for wire protocol: {contiguous.dtype}")
    header = msgpack.packb(
        (contiguous.dtype.str, contiguous.shape),
        use_bin_type=True,
    )
    return msgpack.ExtType(_ARRAY_EXT, header + contiguous.tobytes())


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            "fields": {item.name: _encode(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ProtocolError("Wire mappings may only use string keys.")
        return {key: _encode(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise ProtocolError(f"Unsupported wire value: {type(value).__name__}")


def _ext_hook(code: int, data: bytes) -> np.ndarray | msgpack.ExtType:
    if code != _ARRAY_EXT:
        return msgpack.ExtType(code, data)
    try:
        unpacker = msgpack.Unpacker(raw=False)
        unpacker.feed(data)
        dtype_str, shape = unpacker.unpack()
        offset = unpacker.tell()
        dtype = np.dtype(dtype_str)
        if dtype.hasobject or dtype.kind not in "biuf":
            raise ProtocolError(f"Unsupported NumPy dtype for wire protocol: {dtype}")
        shape_tuple = tuple(int(size) for size in shape)
        if any(size < 0 for size in shape_tuple):
            raise ProtocolError("Array dimensions must be non-negative.")
        expected_size = int(np.prod(shape_tuple, dtype=np.int64)) * dtype.itemsize
        raw = data[offset:]
        if len(raw) != expected_size:
            raise ProtocolError("Array payload size does not match its shape and dtype.")
        return np.frombuffer(raw, dtype=dtype).reshape(shape_tuple).copy()
    except ProtocolError:
        raise
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError("Invalid NumPy array payload.") from exc


def _decode(value: Any) -> Any:
    if isinstance(value, msgpack.ExtType):
        if value.code != _ARRAY_EXT:
            raise ProtocolError(f"Unknown msgpack extension type: {value.code}")
        return _ext_hook(value.code, value.data)
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(_decode(item) for item in value["__tuple__"])
        if "__dataclass__" in value:
            class_name = value["__dataclass__"]
            data = value.get("fields")
            cls = _TYPE_REGISTRY.get(class_name)
            if cls is None or not isinstance(data, dict):
                raise ProtocolError(f"Unknown or malformed wire dataclass: {class_name!r}")
            allowed_fields = {item.name for item in fields(cls)}
            if set(data) != allowed_fields:
                raise ProtocolError(f"Invalid fields for wire dataclass: {class_name!r}")
            return cls(**{key: _decode(item) for key, item in data.items()})
        return {key: _decode(item) for key, item in value.items()}
    return value


def encode_message(message: dict[str, Any]) -> bytes:
    """Encode a protocol message; the version field is required."""
    version = message.get("protocol_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise ProtocolVersionError(
            f"Unsupported protocol version {version!r}; expected {PROTOCOL_VERSION}."
        )
    return msgpack.packb(_encode(message), use_bin_type=True)


def decode_message(payload: bytes) -> dict[str, Any]:
    """Decode and validate a protocol message, rejecting version mismatches."""
    if len(payload) > 512 * 1024 * 1024:
        raise ProtocolError("Protocol message exceeds the 512 MiB size limit.")
    try:
        unpacked = msgpack.unpackb(
            payload,
            raw=False,
            ext_hook=_ext_hook,
            strict_map_key=True,
        )
        message = _decode(unpacked)
    except ProtocolError:
        raise
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError("Malformed protocol message.") from exc

    if not isinstance(message, dict):
        raise ProtocolError("Protocol message must be a mapping.")
    version = message.get("protocol_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise ProtocolVersionError(
            f"Unsupported protocol version {version!r}; expected {PROTOCOL_VERSION}."
        )
    required = {"protocol_version", "type", "seq", "sim_time", "payload", "error"}
    if set(message) != required:
        raise ProtocolError(f"Protocol message fields must be exactly {sorted(required)}.")
    if not isinstance(message["type"], str):
        raise ProtocolError("Protocol message type must be a string.")
    if (
        not isinstance(message["seq"], int)
        or isinstance(message["seq"], bool)
        or message["seq"] < 0
    ):
        raise ProtocolError("Protocol message sequence must be a non-negative integer.")
    if message["sim_time"] is not None and (
        not isinstance(message["sim_time"], (int, float))
        or isinstance(message["sim_time"], bool)
        or not math.isfinite(message["sim_time"])
    ):
        raise ProtocolError("Protocol message sim_time must be numeric or null.")
    if not isinstance(message["payload"], dict):
        raise ProtocolError("Protocol message payload must be a mapping.")
    if message["error"] is not None and not isinstance(message["error"], str):
        raise ProtocolError("Protocol message error must be a string or null.")
    return message


def make_message(
    message_type: str,
    seq: int,
    payload: dict[str, Any] | None = None,
    sim_time: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a protocol envelope with the current version and all required fields."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type,
        "seq": seq,
        "sim_time": sim_time,
        "payload": {} if payload is None else payload,
        "error": error,
    }
