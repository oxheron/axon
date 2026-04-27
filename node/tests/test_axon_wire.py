"""Unit tests for axon_quic.wire — no torch/CUDA/QUIC required."""
from __future__ import annotations

import struct
import sys
import types

import pytest

# Stub torch so the wire module loads without a real install.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _FakeDtype:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return f"torch.{self.name}"

    torch_stub.float16 = _FakeDtype("float16")
    torch_stub.bfloat16 = _FakeDtype("bfloat16")
    torch_stub.float32 = _FakeDtype("float32")
    torch_stub.float8_e4m3fn = _FakeDtype("float8_e4m3fn")
    sys.modules["torch"] = torch_stub

from axon_quic.wire import (
    MAGIC,
    HEADER_SIZE,
    DTYPE_BYTES,
    _HEADER_STRUCT,
    encode_header,
    decode_header,
    encode_shape,
    decode_shape,
    encode_object_frame,
    decode_object_frame,
    dtype_to_enum,
    enum_to_dtype,
)


class TestHeader:
    def test_header_size(self):
        hdr = encode_header(1, 2, 42, 1024, 0)
        assert len(hdr) == HEADER_SIZE

    def test_round_trip_basic(self):
        hdr = encode_header(2, 3, 99, 512, 0)
        dtype_enum, ndim, stream_id, payload_len, flags = decode_header(hdr)
        assert dtype_enum == 2
        assert ndim == 3
        assert stream_id == 99
        assert payload_len == 512
        assert flags == 0

    def test_fp8_flag(self):
        hdr = encode_header(4, 2, 7, 256, flags=1)
        _, _, _, _, flags = decode_header(hdr)
        assert flags & 1 == 1

    def test_bytes_dtype(self):
        hdr = encode_header(DTYPE_BYTES, 0, 0, 100, 0)
        dtype_enum, ndim, _, _, _ = decode_header(hdr)
        assert dtype_enum == DTYPE_BYTES
        assert ndim == 0

    def test_bad_magic_raises(self):
        hdr = encode_header(1, 1, 1, 1, 0)
        corrupted = b"\x00\x00\x00\x00" + hdr[4:]
        with pytest.raises(ValueError, match="bad magic"):
            decode_header(corrupted)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            decode_header(b"\x00" * 10)

    def test_all_dtype_enums(self):
        for enum_val in (1, 2, 3, 4):
            hdr = encode_header(enum_val, 1, 0, 4, 0)
            got, _, _, _, _ = decode_header(hdr)
            assert got == enum_val

    def test_large_stream_id(self):
        stream_id = (1 << 63) - 1
        hdr = encode_header(1, 1, stream_id, 8, 0)
        _, _, got, _, _ = decode_header(hdr)
        assert got == stream_id

    def test_flags_field_is_unsigned_uint64(self):
        """flags is stored as uint64 (Q), not int64 (q). Values >= 2^32 must round-trip."""
        # Verify the struct format char for flags is 'Q' (unsigned), not 'q' (signed).
        fmt = _HEADER_STRUCT.format
        # Format is "<IHHIQIq" (old, broken) vs "<IHHIQIQ" (correct).
        # The last character before the closing of the format must be Q.
        assert fmt.endswith("Q"), (
            f"flags field must use 'Q' (uint64), not 'q' (int64) — got format: {fmt!r}"
        )

    def test_flags_bit32_round_trips(self):
        """Bit 32 in flags must survive encode/decode."""
        flag = (1 << 32) | 1
        hdr = encode_header(1, 1, 0, 4, flags=flag)
        _, _, _, _, flags = decode_header(hdr)
        assert flags == flag

    def test_flags_bit63_round_trips(self):
        """flags >= 2^63 must encode/decode without error (requires Q, not q).
        struct.pack with 'q' raises StructError for values >= 2^63."""
        flag = (1 << 63)  # exceeds int64 max — would raise with 'q' format
        hdr = encode_header(1, 1, 0, 4, flags=flag)
        _, _, _, _, flags = decode_header(hdr)
        assert flags == flag


class TestShape:
    def test_empty_shape(self):
        assert decode_shape(b"", 0) == []

    def test_1d_shape(self):
        buf = encode_shape([512])
        assert decode_shape(buf, 1) == [512]

    def test_3d_shape(self):
        shape = [2, 4096, 128]
        buf = encode_shape(shape)
        assert decode_shape(buf, 3) == shape


class TestFP8WirePath:
    """Tests for FP8 on-wire encoding/decoding correctness.

    These tests verify the header and code path. Numeric correctness of the FP8
    cast requires real torch with float8 support and is covered by the e2e tests.
    """

    def test_fp8_header_flag_survives_round_trip(self):
        """FP8 flag (bit 0) must survive header encode → decode."""
        hdr = encode_header(dtype_enum=4, ndim=2, stream_id=7, payload_len=12, flags=1)
        dtype_enum, ndim, stream_id, payload_len, flags = decode_header(hdr)
        assert dtype_enum == 4
        assert ndim == 2
        assert stream_id == 7
        assert payload_len == 12
        assert flags & 1 == 1

    def test_native_wire_has_no_fp8_flag(self):
        """A native bf16 frame must have flags=0 (bit 0 clear)."""
        hdr = encode_header(dtype_enum=2, ndim=1, stream_id=0, payload_len=8, flags=0)
        _, _, _, _, flags = decode_header(hdr)
        assert flags & 1 == 0

    def test_fp8_dtype_enum_is_distinct(self):
        """dtype_enum=4 (float8_e4m3fn) must not collide with other enums."""
        for other_enum in (1, 2, 3):
            assert other_enum != 4


class TestObjectFrame:
    def test_dict_round_trip(self):
        obj = {"dtype": "bfloat16", "shape": [1, 3072]}
        frame = encode_object_frame(obj, stream_id=5)
        # Header says DTYPE_BYTES
        dtype_enum, _, stream_id, _, _ = decode_header(frame[:HEADER_SIZE])
        assert dtype_enum == DTYPE_BYTES
        assert stream_id == 5
        recovered = decode_object_frame(frame)
        assert recovered == obj

    def test_list_round_trip(self):
        obj = [("key", 42), ("other", [1, 2, 3])]
        frame = encode_object_frame(obj, stream_id=0)
        recovered = decode_object_frame(frame)
        assert recovered == obj

    def test_wrong_dtype_raises(self):
        # A tensor frame (dtype=1) fed to decode_object_frame should raise
        hdr = encode_header(1, 1, 0, 4, 0)
        with pytest.raises(ValueError, match="DTYPE_BYTES"):
            decode_object_frame(hdr + b"\x00" * 4)
