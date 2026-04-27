"""Unit tests for AxonQuicProcessGroup using a mock sidecar client."""
from __future__ import annotations

import os
import queue
import struct
import sys
import threading
import types

import pytest

# ── Stub torch ──────────────────────────────────────────────────────────────

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _FakeDtype:
        def __init__(self, name):
            self.name = name

    torch_stub.float16 = _FakeDtype("float16")
    torch_stub.bfloat16 = _FakeDtype("bfloat16")
    torch_stub.float32 = _FakeDtype("float32")
    torch_stub.float8_e4m3fn = _FakeDtype("float8_e4m3fn")

    class _FakeDevice:
        def __init__(self, type="cpu"):
            self.type = type

    class _FakeTensor:
        def __init__(self, data=b"", shape=(1,), dtype=None):
            self._data = data
            self.shape = shape
            self.dtype = dtype or torch_stub.bfloat16
            self.device = _FakeDevice()

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def to(self, *a, **kw):
            return self

        def view(self, *a, **kw):
            return self

        def copy_(self, other):
            self._data = other._data
            return self

        def numpy(self):
            import numpy as np
            return np.zeros(self.shape, dtype=np.uint16)

    torch_stub.Tensor = _FakeTensor
    torch_stub.device = _FakeDevice
    sys.modules["torch"] = torch_stub

if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.uint16 = "uint16"
    np_stub.float16 = "float16"
    np_stub.float32 = "float32"
    np_stub.uint8 = "uint8"
    np_stub.dtype = lambda x: x

    class _FakeArray:
        def __init__(self, shape):
            self.shape = shape

        def reshape(self, *a):
            return self

        def tobytes(self):
            return b"\x00" * 4

    np_stub.zeros = lambda shape, dtype=None: _FakeArray(shape if isinstance(shape, tuple) else (shape,))
    np_stub.frombuffer = lambda *a, **kw: _FakeArray((1,))
    sys.modules["numpy"] = np_stub


# ── Mock sidecar client ─────────────────────────────────────────────────────

class _QueueSidecarClient:
    """In-process mock: sends go to send_q, recvs come from recv_q."""

    def __init__(self, send_q, recv_q):
        self._send_q = send_q
        self._recv_q = recv_q

    def send(self, peer_idx, stream_id, frame):
        self._send_q.put(("tensor", peer_idx, stream_id, frame))

    def recv(self, peer_idx, stream_id):
        return self._recv_q.get(timeout=1.0)

    def send_obj(self, peer_idx, stream_id, frame):
        self._send_q.put(("obj", peer_idx, stream_id, frame))

    def recv_obj(self, peer_idx, stream_id):
        return self._recv_q.get(timeout=1.0)

    def close(self):
        pass


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("AXON_TRANSPORT_UDS", "/tmp/test-axon.sock")
    monkeypatch.setenv("AXON_PP_RANK", "0")
    monkeypatch.setenv("AXON_PP_SIZE", "2")
    monkeypatch.setenv("AXON_WIRE_DTYPE", "native")


class TestAxonQuicProcessGroup:
    def _make_pg(self, send_q=None, recv_q=None):
        from axon_quic.process_group import AxonQuicProcessGroup
        pg = AxonQuicProcessGroup.__new__(AxonQuicProcessGroup)
        pg._rank = 0
        pg._size = 2
        pg._wire_dtype = "native"
        pg._client = _QueueSidecarClient(
            send_q or queue.Queue(),
            recv_q or queue.Queue(),
        )
        return pg

    def test_isend_puts_frame_in_queue(self):
        send_q = queue.Queue()
        pg = self._make_pg(send_q=send_q)
        import torch
        tensor = torch.Tensor()
        work = pg.isend([tensor], dst=1, tag=42)
        assert work.is_completed()
        kind, peer_idx, stream_id, frame = send_q.get_nowait()
        assert kind == "tensor"
        assert peer_idx == 1
        assert stream_id == 42

    def test_irecv_fills_tensor(self):
        from axon_quic.wire import encode_header, encode_shape, DTYPE_BYTES, HEADER_SIZE
        # Build a minimal valid AXON frame (dtype=1=float16, shape=[1])
        hdr = encode_header(dtype_enum=1, ndim=1, stream_id=42, payload_len=2, flags=0)
        shape_bytes = encode_shape([1])
        payload = b"\x00\x00"
        frame = hdr + shape_bytes + payload

        recv_q = queue.Queue()
        recv_q.put(frame)
        pg = self._make_pg(recv_q=recv_q)

        import torch
        tensor = torch.Tensor()
        work = pg.irecv([tensor], src=1, tag=42)
        assert work.is_completed()

    def test_allreduce_raises(self):
        pg = self._make_pg()
        with pytest.raises(NotImplementedError, match="allreduce"):
            pg.allreduce()

    def test_broadcast_raises(self):
        pg = self._make_pg()
        with pytest.raises(NotImplementedError, match="broadcast"):
            pg.broadcast()

    def test_allgather_raises(self):
        pg = self._make_pg()
        with pytest.raises(NotImplementedError, match="allgather"):
            pg.allgather()

    def test_isend_connection_error_returns_failed_work(self, monkeypatch):
        pg = self._make_pg()

        def _bad_send(*a, **kw):
            raise OSError("connection refused")

        pg._client.send = _bad_send

        import torch
        work = pg.isend([torch.Tensor()], dst=1, tag=0)
        assert not work.is_success()
        with pytest.raises(ConnectionError):
            work.wait()

    def test_send_object_puts_obj_frame(self):
        send_q = queue.Queue()
        pg = self._make_pg(send_q=send_q)
        pg.send_object({"dtype": "bf16", "shape": [1, 3]}, dst=1, stream_id=0)
        kind, peer_idx, stream_id, frame = send_q.get_nowait()
        assert kind == "obj"

    def test_recv_object_returns_obj(self):
        from axon_quic.wire import encode_object_frame
        obj = {"dtype": "bf16", "shape": [1, 3]}
        frame = encode_object_frame(obj, stream_id=0)
        recv_q = queue.Queue()
        recv_q.put(frame)
        pg = self._make_pg(recv_q=recv_q)
        result = pg.recv_object(src=1, stream_id=0)
        assert result == obj

    def test_rank_and_size(self):
        pg = self._make_pg()
        assert pg.rank() == 0
        assert pg.size() == 2
