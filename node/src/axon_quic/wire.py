"""
AXON wire format: 32-byte little-endian header followed by shape bytes then payload.

Header layout:
  0-3   uint32  magic = 0x41584f4e ("AXON")
  4-5   uint16  version (currently 1)
  6-7   uint16  dtype enum
  8-11  uint32  ndim
  12-19 uint64  stream_id / tag
  20-23 uint32  payload_len
  24-31 uint64  flags  (bit 0 = FP8 cast applied, rest reserved)

dtype enum:
  1 = float16, 2 = bfloat16, 3 = float32, 4 = float8_e4m3fn, 255 = bytes (pickle)
"""
from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

MAGIC = 0x41584F4E
VERSION = 1
HEADER_SIZE = 32

# dtype enum ↔ torch dtype
_DTYPE_TO_ENUM: dict[object, int] = {}
_ENUM_TO_DTYPE: dict[int, object] = {}

DTYPE_BYTES = 255  # pickle payload

_FP8_DTYPE_NAMES = ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")

_HEADER_STRUCT = struct.Struct("<IHHIQIQ")  # 4+2+2+4+8+4+8 = 32 bytes; flags is unsigned Q


def _init_dtype_maps() -> None:
    try:
        import torch

        mapping = [
            (1, torch.float16),
            (2, torch.bfloat16),
            (3, torch.float32),
            (4, getattr(torch, "float8_e4m3fn", None)),
        ]
        for enum_val, dt in mapping:
            if dt is not None:
                _DTYPE_TO_ENUM[dt] = enum_val
                _ENUM_TO_DTYPE[enum_val] = dt
    except ImportError:
        pass


_init_dtype_maps()


def dtype_to_enum(dtype: object) -> int:
    if not _DTYPE_TO_ENUM:
        _init_dtype_maps()
    val = _DTYPE_TO_ENUM.get(dtype)
    if val is None:
        raise ValueError(f"unsupported dtype for AXON wire: {dtype}")
    return val


def enum_to_dtype(enum_val: int) -> object:
    if not _ENUM_TO_DTYPE:
        _init_dtype_maps()
    if enum_val == DTYPE_BYTES:
        return None
    dt = _ENUM_TO_DTYPE.get(enum_val)
    if dt is None:
        raise ValueError(f"unknown AXON dtype enum: {enum_val}")
    return dt


def encode_header(
    dtype_enum: int,
    ndim: int,
    stream_id: int,
    payload_len: int,
    flags: int = 0,
) -> bytes:
    return _HEADER_STRUCT.pack(MAGIC, VERSION, dtype_enum, ndim, stream_id, payload_len, flags)


def decode_header(buf: bytes) -> tuple[int, int, int, int, int]:
    """Returns (dtype_enum, ndim, stream_id, payload_len, flags)."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"AXON header too short: {len(buf)} < {HEADER_SIZE}")
    magic, _ver, dtype_enum, ndim, stream_id, payload_len, flags = _HEADER_STRUCT.unpack(
        buf[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"AXON bad magic: 0x{magic:08X}")
    return dtype_enum, ndim, stream_id, payload_len, flags


def encode_shape(shape: list[int]) -> bytes:
    return struct.pack(f"<{len(shape)}I", *shape)


def decode_shape(buf: bytes, ndim: int) -> list[int]:
    return list(struct.unpack(f"<{ndim}I", buf[: ndim * 4]))


# ── Tensor ↔ bytes helpers ──────────────────────────────────────────────────


def tensor_to_bytes(
    tensor: "torch.Tensor",
    stream_id: int,
    *,
    wire_dtype: str = "native",
) -> bytes:
    """Encode a tensor as a complete AXON frame (header + shape + payload).

    wire_dtype: "native" preserves original dtype; "fp8" downcasts bf16/fp16 to fp8.
    Returns raw bytes ready to write to QUIC / UDS.
    """
    import torch

    flags = 0
    original_dtype = tensor.dtype

    if wire_dtype == "fp8" and original_dtype in (torch.bfloat16, torch.float16):
        fp8_type = getattr(torch, "float8_e4m3fn", None)
        if fp8_type is not None:
            tensor = tensor.to(fp8_type)
            flags |= 1  # bit 0: FP8 cast applied

    dtype_enum = dtype_to_enum(tensor.dtype)
    shape = list(tensor.shape)
    t_cpu = tensor.contiguous().cpu()
    # float8 has no numpy-compatible dtype; view as uint8 for byte-level access.
    # bfloat16 numpy support is fragile across PyTorch versions; use uint8 view too.
    _fp8_type = getattr(torch, "float8_e4m3fn", None)
    if t_cpu.dtype == torch.bfloat16 or (_fp8_type is not None and t_cpu.dtype == _fp8_type):
        payload = t_cpu.view(torch.uint8).numpy().tobytes()
    else:
        payload = t_cpu.numpy().tobytes()
    header = encode_header(dtype_enum, len(shape), stream_id, len(payload), flags)
    shape_bytes = encode_shape(shape)
    return header + shape_bytes + payload


def bytes_to_tensor(
    data: bytes,
    *,
    target_device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Decode an AXON frame back to a tensor. Handles FP8 upcast if flag set."""
    import torch
    import numpy as np

    dtype_enum, ndim, _stream_id, payload_len, flags = decode_header(data)
    offset = HEADER_SIZE
    shape = decode_shape(data[offset:], ndim)
    offset += ndim * 4
    payload = data[offset : offset + payload_len]

    wire_dtype = enum_to_dtype(dtype_enum)
    np_dtype = _torch_dtype_to_numpy(wire_dtype)
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(arr.copy())

    # Upcast FP8 → BF16 if flag set. Payload bytes represent float8 memory; view as
    # float8_e4m3fn first (same bit width as uint8) then cast to bfloat16.
    if flags & 1:
        fp8_type = getattr(torch, "float8_e4m3fn", None)
        if fp8_type is not None:
            tensor = tensor.view(fp8_type).to(torch.bfloat16)
        else:
            tensor = tensor.to(torch.bfloat16)

    if target_device is not None:
        tensor = tensor.to(target_device)
    return tensor


def _torch_dtype_to_numpy(dtype: object) -> "np.dtype":
    import numpy as np

    _map = {
        1: np.float16,
        2: np.float32,  # bfloat16 has no numpy; load as float32 then view
        3: np.float32,
        4: np.uint8,    # float8 treated as raw bytes in numpy
    }
    try:
        import torch
        if dtype == torch.bfloat16:
            return np.dtype("uint16")  # same bit-width; reinterpret later
        for enum_val, np_dt in _map.items():
            if _ENUM_TO_DTYPE.get(enum_val) == dtype:
                return np.dtype(np_dt)
    except ImportError:
        pass
    return np.dtype("uint8")


def bytes_to_tensor_bf16(data: bytes, *, target_device=None) -> "torch.Tensor":
    """Like bytes_to_tensor but reinterprets uint16 → bfloat16 after numpy load."""
    import torch

    dtype_enum, ndim, _stream_id, payload_len, flags = decode_header(data)
    offset = HEADER_SIZE
    import struct as _struct
    shape = list(_struct.unpack(f"<{ndim}I", data[offset : offset + ndim * 4]))
    offset += ndim * 4
    payload = data[offset : offset + payload_len]

    import numpy as np

    wire_dtype = enum_to_dtype(dtype_enum)

    try:
        import torch as _torch
        if wire_dtype == _torch.bfloat16:
            arr = np.frombuffer(payload, dtype=np.uint16).reshape(shape)
            tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16)
            if target_device is not None:
                tensor = tensor.to(target_device)
            return tensor
    except ImportError:
        pass

    return bytes_to_tensor(data, target_device=target_device)


# ── Pickle frame helpers ──────────────────────────────────────────────────


def encode_object_frame(obj: object, stream_id: int) -> bytes:
    """Pickle-encode an object into an AXON frame with dtype=BYTES."""
    import pickle

    payload = pickle.dumps(obj)
    header = encode_header(DTYPE_BYTES, 0, stream_id, len(payload), 0)
    return header + payload


def decode_object_frame(data: bytes) -> object:
    """Decode a DTYPE_BYTES AXON frame back to a Python object."""
    import pickle

    dtype_enum, _ndim, _stream_id, payload_len, _flags = decode_header(data)
    if dtype_enum != DTYPE_BYTES:
        raise ValueError(f"expected DTYPE_BYTES frame, got dtype_enum={dtype_enum}")
    offset = HEADER_SIZE
    return pickle.loads(data[offset : offset + payload_len])
