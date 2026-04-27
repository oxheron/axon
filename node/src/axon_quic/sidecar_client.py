"""SidecarClient — blocking UDS client used by AxonQuicProcessGroup.

Runs in the vLLM worker thread (synchronous context). Each call blocks until
the node-process sidecar server acks (SEND) or delivers the frame (RECV).

UDS wire protocol (all little-endian):

  SEND request:
    [1B opcode=0x01] [1B peer_idx] [4B stream_id] [4B frame_len] [frame_len B frame]
  SEND response:
    [1B status]  0=ok 1=error

  RECV request:
    [1B opcode=0x02] [1B peer_idx] [4B stream_id]
  RECV response:
    [1B status] [4B frame_len] [frame_len B frame]

  SEND_OBJ request (opcode=0x03) / RECV_OBJ (opcode=0x04): same layout as SEND/RECV.
"""
from __future__ import annotations

import socket
import struct
import threading
import logging

LOGGER = logging.getLogger(__name__)

_OP_SEND = 0x01
_OP_RECV = 0x02
_OP_SEND_OBJ = 0x03
_OP_RECV_OBJ = 0x04

_SEND_TIMEOUT = 5.0   # seconds — raises ConnectionError if sidecar doesn't respond


class SidecarClient:
    """Thread-safe blocking UDS client."""

    def __init__(self, uds_path: str, timeout: float = _SEND_TIMEOUT) -> None:
        self._path = uds_path
        self._timeout = timeout
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._path)
        return sock

    def _sock_or_connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = self._connect()
        return self._sock

    def _send_all(self, sock: socket.socket, data: bytes) -> None:
        total = 0
        while total < len(data):
            sent = sock.send(data[total:])
            if sent == 0:
                raise ConnectionError("sidecar socket closed during send")
            total += sent

    def _recv_exactly(self, sock: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("sidecar socket closed during recv")
            buf += chunk
        return buf

    def _call(self, request: bytes, expect_frame: bool) -> bytes | None:
        with self._lock:
            try:
                sock = self._sock_or_connect()
                self._send_all(sock, request)
                status = self._recv_exactly(sock, 1)[0]
                if status != 0:
                    raise ConnectionError(f"axon sidecar returned error status 0x{status:02x}")
                if expect_frame:
                    (frame_len,) = struct.unpack("<I", self._recv_exactly(sock, 4))
                    return self._recv_exactly(sock, frame_len)
                return None
            except Exception:
                # Reset socket on any error so next call reconnects.
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                raise

    def send(self, peer_idx: int, stream_id: int, frame: bytes) -> None:
        request = struct.pack("<BBII", _OP_SEND, peer_idx, stream_id, len(frame)) + frame
        self._call(request, expect_frame=False)

    def recv(self, peer_idx: int, stream_id: int) -> bytes:
        request = struct.pack("<BBI", _OP_RECV, peer_idx, stream_id)
        result = self._call(request, expect_frame=True)
        assert result is not None
        return result

    def send_obj(self, peer_idx: int, stream_id: int, frame: bytes) -> None:
        request = struct.pack("<BBII", _OP_SEND_OBJ, peer_idx, stream_id, len(frame)) + frame
        self._call(request, expect_frame=False)

    def recv_obj(self, peer_idx: int, stream_id: int) -> bytes:
        request = struct.pack("<BBI", _OP_RECV_OBJ, peer_idx, stream_id)
        result = self._call(request, expect_frame=True)
        assert result is not None
        return result

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
