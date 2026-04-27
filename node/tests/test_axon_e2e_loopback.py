"""Integration tests: QuicConn ↔ TransportSidecar ↔ SidecarClient over loopback.

Requires aioquic. Skipped automatically if not installed.
"""
from __future__ import annotations

import asyncio
import socket
import sys
import time
import types
from types import SimpleNamespace

import pytest

# ── Stub torch if not installed ──────────────────────────────────────────────

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _FakeDtype:
        def __init__(self, name):
            self.name = name

    torch_stub.float16 = _FakeDtype("float16")
    torch_stub.bfloat16 = _FakeDtype("bfloat16")
    torch_stub.float32 = _FakeDtype("float32")
    torch_stub.float8_e4m3fn = _FakeDtype("float8_e4m3fn")
    torch_stub.uint8 = _FakeDtype("uint8")

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

# ── aioquic availability guard ───────────────────────────────────────────────

aioquic = pytest.importorskip("aioquic", reason="aioquic not installed")

pytestmark = pytest.mark.asyncio


# ── Config tests ─────────────────────────────────────────────────────────────


def test_quic_client_config_skips_cert_verify_by_default(monkeypatch):
    """Phase B: client must have verify_mode=CERT_NONE when AXON_QUIC_VERIFY is unset.
    This allows cross-node QUIC where each node has its own self-signed cert."""
    import ssl
    monkeypatch.delenv("AXON_QUIC_VERIFY", raising=False)
    from transport.quic_conn import _make_quic_config
    config = _make_quic_config(is_client=True)
    assert config.verify_mode == ssl.CERT_NONE, (
        "QUIC client must skip cert verification by default (Phase B uses per-node "
        "self-signed certs). Set AXON_QUIC_VERIFY=1 to enable after certs are shared."
    )


def test_quic_server_config_has_cert():
    """Server must always set its self-signed certificate."""
    from transport.quic_conn import _make_quic_config
    config = _make_quic_config(is_client=False)
    assert config.certificate is not None
    assert config.private_key is not None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_loopback_sock_pair() -> tuple[socket.socket, int, socket.socket, int]:
    """Return (server_sock, server_port, client_sock, client_port)."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind(("127.0.0.1", 0))
    server_port = server_sock.getsockname()[1]

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock.bind(("127.0.0.1", 0))
    client_port = client_sock.getsockname()[1]

    return server_sock, server_port, client_sock, client_port


async def _make_quic_pair():
    """Return (server_conn, client_conn) connected over 127.0.0.1 loopback."""
    from transport.quic_conn import QuicConn

    server_sock, server_port, client_sock, client_port = _make_loopback_sock_pair()

    server_conn = QuicConn(
        peer_id="client-node",
        local_sock=server_sock,
        remote_addr=("127.0.0.1", client_port),
        is_client=False,
    )
    client_conn = QuicConn(
        peer_id="server-node",
        local_sock=client_sock,
        remote_addr=("127.0.0.1", server_port),
        is_client=True,
    )

    # Both must start concurrently — server waits for client to initiate.
    await asyncio.gather(
        server_conn.connect(timeout=10.0),
        client_conn.connect(timeout=10.0),
    )
    return server_conn, client_conn


def _fake_transport(conn_map: dict) -> SimpleNamespace:
    """Minimal P2PTransport stand-in that holds a _conns dict."""
    return SimpleNamespace(_conns=conn_map)


# ── Tests ────────────────────────────────────────────────────────────────────


async def test_quic_conn_send_recv_loopback():
    """Basic send_data/recv_data round-trip over loopback QUIC."""
    server_conn, client_conn = await _make_quic_pair()
    try:
        payload = b"hello from client"
        await client_conn.send_data(payload)
        received = await asyncio.wait_for(server_conn.recv_data(), timeout=5.0)
        assert received == payload
    finally:
        client_conn.close()
        server_conn.close()


async def test_quic_conn_bidirectional():
    """Data flows in both directions simultaneously."""
    server_conn, client_conn = await _make_quic_pair()
    try:
        msg_c2s = b"client->server"
        msg_s2c = b"server->client"

        await asyncio.gather(
            client_conn.send_data(msg_c2s),
            server_conn.send_data(msg_s2c),
        )

        got_s, got_c = await asyncio.gather(
            asyncio.wait_for(server_conn.recv_data(), timeout=5.0),
            asyncio.wait_for(client_conn.recv_data(), timeout=5.0),
        )

        assert got_s == msg_c2s
        assert got_c == msg_s2c
    finally:
        client_conn.close()
        server_conn.close()


async def test_sidecar_send_recv_via_uds():
    """SidecarClient.send → TransportSidecar → QuicConn → recv end-to-end."""
    import shutil
    import tempfile
    from transport.sidecar import TransportSidecar
    from axon_quic.sidecar_client import SidecarClient
    from axon_quic.wire import encode_object_frame, decode_object_frame

    server_conn, client_conn = await _make_quic_pair()

    tmpdir = tempfile.mkdtemp(dir="/tmp", prefix="axon-t-")
    uds_a = tmpdir + "/a.sock"
    uds_b = tmpdir + "/b.sock"

    # Sidecar A owns client_conn (talks to server-node)
    # Sidecar B owns server_conn (talks to client-node)
    transport_a = _fake_transport({"server-node": client_conn})
    transport_b = _fake_transport({"client-node": server_conn})

    # peers_by_rank[0] = "server-node" for sidecar A, vice versa for B
    sidecar_a = TransportSidecar(transport_a, uds_a, ["server-node"])
    sidecar_b = TransportSidecar(transport_b, uds_b, ["client-node"])

    await sidecar_a.start()
    await sidecar_b.start()

    try:
        obj_payload = {"dtype": "bf16", "shape": [1, 4096]}
        frame = encode_object_frame(obj_payload, stream_id=7)

        def _send():
            sc = SidecarClient(uds_a, timeout=5.0)
            sc.send_obj(peer_idx=0, stream_id=7, frame=frame)
            sc.close()

        def _recv():
            sc = SidecarClient(uds_b, timeout=5.0)
            raw = sc.recv_obj(peer_idx=0, stream_id=7)
            sc.close()
            return decode_object_frame(raw)

        # asyncio.to_thread keeps the event loop alive so sidecar coroutines can run.
        _, result = await asyncio.gather(
            asyncio.to_thread(_send),
            asyncio.to_thread(_recv),
        )

        assert result == obj_payload

    finally:
        await sidecar_a.stop()
        await sidecar_b.stop()
        client_conn.close()
        server_conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_sidecar_bidirectional():
    """Both sides send simultaneously; each receives from the other."""
    import shutil
    import tempfile
    from transport.sidecar import TransportSidecar
    from axon_quic.sidecar_client import SidecarClient

    server_conn, client_conn = await _make_quic_pair()

    tmpdir = tempfile.mkdtemp(dir="/tmp", prefix="axon-t-")
    uds_a = tmpdir + "/a.sock"
    uds_b = tmpdir + "/b.sock"

    transport_a = _fake_transport({"server-node": client_conn})
    transport_b = _fake_transport({"client-node": server_conn})

    sidecar_a = TransportSidecar(transport_a, uds_a, ["server-node"])
    sidecar_b = TransportSidecar(transport_b, uds_b, ["client-node"])

    await sidecar_a.start()
    await sidecar_b.start()

    payload_a = b"from-a" * 100
    payload_b = b"from-b" * 100

    try:
        def _side_a():
            sc = SidecarClient(uds_a, timeout=5.0)
            sc.send(peer_idx=0, stream_id=1, frame=payload_a)
            recv = sc.recv(peer_idx=0, stream_id=2)
            sc.close()
            return recv

        def _side_b():
            sc = SidecarClient(uds_b, timeout=5.0)
            sc.send(peer_idx=0, stream_id=2, frame=payload_b)
            recv = sc.recv(peer_idx=0, stream_id=1)
            sc.close()
            return recv

        a_received, b_received = await asyncio.gather(
            asyncio.to_thread(_side_a),
            asyncio.to_thread(_side_b),
        )

        assert a_received == payload_b
        assert b_received == payload_a
    finally:
        await sidecar_a.stop()
        await sidecar_b.stop()
        client_conn.close()
        server_conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_loopback_latency_under_10ms():
    """Round-trip latency for a ~12 KB BF16 tensor frame is < 10 ms on loopback."""
    import shutil
    import tempfile
    from transport.sidecar import TransportSidecar
    from axon_quic.sidecar_client import SidecarClient

    server_conn, client_conn = await _make_quic_pair()
    tmpdir = tempfile.mkdtemp(dir="/tmp", prefix="axon-t-")
    uds_a = tmpdir + "/a.sock"
    uds_b = tmpdir + "/b.sock"

    transport_a = _fake_transport({"server-node": client_conn})
    transport_b = _fake_transport({"client-node": server_conn})

    sidecar_a = TransportSidecar(transport_a, uds_a, ["server-node"])
    sidecar_b = TransportSidecar(transport_b, uds_b, ["client-node"])

    await sidecar_a.start()
    await sidecar_b.start()

    # ~12 KB BF16 tensor payload (6144 bfloat16 values = 12288 bytes)
    payload = b"\xab\xcd" * 6144

    try:
        elapsed: list[float] = []

        def _send():
            sc = SidecarClient(uds_a, timeout=5.0)
            t0 = time.perf_counter()
            sc.send(peer_idx=0, stream_id=0, frame=payload)
            elapsed.append(time.perf_counter() - t0)
            sc.close()

        def _recv():
            sc = SidecarClient(uds_b, timeout=5.0)
            sc.recv(peer_idx=0, stream_id=0)
            sc.close()

        await asyncio.gather(
            asyncio.to_thread(_send),
            asyncio.to_thread(_recv),
        )

        assert elapsed, "sender did not record elapsed time"
        elapsed_ms = elapsed[0] * 1000
        assert elapsed_ms < 10.0, f"send took {elapsed_ms:.1f} ms (> 10 ms)"
    finally:
        await sidecar_a.stop()
        await sidecar_b.stop()
        client_conn.close()
        server_conn.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
