"""Tests for the UDS binary framing in sidecar_client.py."""
from __future__ import annotations

import struct

import pytest


# ── Helpers that mirror the sidecar_client framing ─────────────────────────

_OP_SEND = 0x01
_OP_RECV = 0x02
_OP_SEND_OBJ = 0x03
_OP_RECV_OBJ = 0x04


def make_send_request(peer_idx: int, stream_id: int, frame: bytes) -> bytes:
    return struct.pack("<BBII", _OP_SEND, peer_idx, stream_id, len(frame)) + frame


def make_recv_request(peer_idx: int, stream_id: int) -> bytes:
    return struct.pack("<BBI", _OP_RECV, peer_idx, stream_id)


def make_recv_response(frame: bytes) -> bytes:
    return b"\x00" + struct.pack("<I", len(frame)) + frame


def parse_send_request(data: bytes):
    opcode, peer_idx, stream_id, frame_len = struct.unpack("<BBII", data[:10])
    frame = data[10 : 10 + frame_len]
    return opcode, peer_idx, stream_id, frame


def parse_recv_response(data: bytes):
    status = data[0]
    (frame_len,) = struct.unpack("<I", data[1:5])
    frame = data[5 : 5 + frame_len]
    return status, frame


class TestSendFraming:
    def test_send_round_trip(self):
        frame = b"AXON" + b"\x00" * 28 + b"payload"
        req = make_send_request(peer_idx=1, stream_id=42, frame=frame)
        opcode, peer_idx, stream_id, got_frame = parse_send_request(req)
        assert opcode == _OP_SEND
        assert peer_idx == 1
        assert stream_id == 42
        assert got_frame == frame

    def test_empty_frame(self):
        req = make_send_request(peer_idx=0, stream_id=0, frame=b"")
        _, _, _, got_frame = parse_send_request(req)
        assert got_frame == b""

    def test_large_frame(self):
        frame = b"X" * (1024 * 1024)  # 1 MiB
        req = make_send_request(peer_idx=0, stream_id=99, frame=frame)
        _, _, _, got_frame = parse_send_request(req)
        assert got_frame == frame


class TestRecvFraming:
    def test_recv_request_size(self):
        req = make_recv_request(peer_idx=0, stream_id=5)
        # [1B opcode] [1B peer_idx] [4B stream_id] = 6 bytes
        assert len(req) == 6
        opcode, peer_idx = struct.unpack("<BB", req[:2])
        assert opcode == _OP_RECV
        assert peer_idx == 0

    def test_recv_response_round_trip(self):
        frame = b"hello tensor"
        resp = make_recv_response(frame)
        status, got_frame = parse_recv_response(resp)
        assert status == 0
        assert got_frame == frame

    def test_error_status(self):
        resp = b"\x01" + struct.pack("<I", 0)
        status, got_frame = parse_recv_response(resp)
        assert status == 1
        assert got_frame == b""


class TestSidecarClientImport:
    """Smoke test that SidecarClient can be imported without a running socket."""

    def test_import(self):
        from axon_quic.sidecar_client import SidecarClient  # noqa: F401

    def test_close_without_connect(self):
        from axon_quic.sidecar_client import SidecarClient
        client = SidecarClient("/tmp/nonexistent-axon-test.sock")
        client.close()  # Should not raise

    def test_send_fails_gracefully_when_no_socket(self):
        from axon_quic.sidecar_client import SidecarClient
        client = SidecarClient("/tmp/nonexistent-axon-test.sock", timeout=0.1)
        with pytest.raises(Exception):
            client.send(peer_idx=0, stream_id=0, frame=b"test")
