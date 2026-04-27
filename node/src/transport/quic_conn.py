from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import tempfile
from typing import Optional

LOGGER = logging.getLogger(__name__)

_ALPN_PROTOCOL = "axon-pp"

# Module-level self-signed cert shared by all QuicConn instances in this process.
# Server uses it to authenticate; client trusts it via cafile.
_cert_obj = None
_key_obj = None
_cert_pem_path: str = ""
_cert_tmpdir: str = ""


def _ensure_cert():
    """Generate (once) a self-signed EC cert and write it to a temp PEM file."""
    global _cert_obj, _key_obj, _cert_pem_path, _cert_tmpdir
    if _cert_obj is not None:
        return

    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "axon")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("axon")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    _cert_obj = cert
    _key_obj = key

    _cert_tmpdir = tempfile.mkdtemp(prefix="axon-quic-")
    _cert_pem_path = os.path.join(_cert_tmpdir, "cert.pem")
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    with open(_cert_pem_path, "wb") as f:
        f.write(cert_pem)

    import shutil
    atexit.register(lambda: shutil.rmtree(_cert_tmpdir, ignore_errors=True))


def _make_quic_config(is_client: bool):
    import ssl
    from aioquic.quic.configuration import QuicConfiguration

    _ensure_cert()
    config = QuicConfiguration(
        is_client=is_client,
        alpn_protocols=[_ALPN_PROTOCOL],
    )
    if not is_client:
        config.certificate = _cert_obj
        config.private_key = _key_obj
    else:
        config.server_name = "axon"
        # Phase B: each node generates its own self-signed cert, so the client
        # cannot verify the server cert against a local CA. Skip verification by
        # default. Set AXON_QUIC_VERIFY=1 to enable cert checking (requires certs
        # to be distributed via coordinator signaling — deferred to post-B-5).
        if os.environ.get("AXON_QUIC_VERIFY") == "1":
            config.cafile = _cert_pem_path
        else:
            config.verify_mode = ssl.CERT_NONE
    return config


def _make_protocol_class():
    """Return a fresh QuicConnectionProtocol subclass with handshake and data tracking."""
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.quic.events import ConnectionTerminated, HandshakeCompleted, StreamDataReceived

    class _AxonProtocol(QuicConnectionProtocol):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.handshake_complete = asyncio.Event()
            self.terminated = False
            self._recv_queue: asyncio.Queue[bytes] = asyncio.Queue()
            self._stream_bufs: dict[int, bytes] = {}

        def quic_event_received(self, event):
            super().quic_event_received(event)
            if isinstance(event, HandshakeCompleted):
                self.handshake_complete.set()
            elif isinstance(event, ConnectionTerminated):
                self.terminated = True
                self.handshake_complete.set()
            elif isinstance(event, StreamDataReceived):
                sid = event.stream_id
                self._stream_bufs[sid] = self._stream_bufs.get(sid, b"") + event.data
                if event.end_stream:
                    payload = self._stream_bufs.pop(sid)
                    self._recv_queue.put_nowait(payload)

    return _AxonProtocol


class QuicConn:
    """
    One persistent QUIC connection to a single peer.

    Accepts a pre-bound UDP socket. The socket's address is extracted and the
    socket is closed before handing control to aioquic (which rebinds the same
    port via local_port= / host:port). This is required by aioquic 1.x which
    removed the sock= parameter from connect() and serve().

    After connect() returns, use send_data()/recv_data() to exchange frames.
    """

    def __init__(
        self,
        peer_id: str,
        local_sock: socket.socket,
        remote_addr: tuple[str, int],
        is_client: bool,
    ) -> None:
        self._peer_id = peer_id
        self._local_sock = local_sock
        self._remote_addr = remote_addr
        self._is_client = is_client
        self._connected = False
        self._closed = False
        self._close_event = asyncio.Event()
        self._server = None   # QuicServer handle (server side only)
        self._task: Optional[asyncio.Task] = None
        self._protocol: Optional[object] = None  # _AxonProtocol instance

    async def connect(self, timeout: float = 30.0) -> None:
        if self._is_client:
            await self._connect_client(timeout)
        else:
            await self._connect_server(timeout)
        self._connected = True
        LOGGER.info(
            "[quic] handshake complete: peer=%s addr=%s:%d role=%s",
            self._peer_id,
            self._remote_addr[0],
            self._remote_addr[1],
            "client" if self._is_client else "server",
        )

    async def _connect_client(self, timeout: float) -> None:
        """Client side: use aioquic.asyncio.connect() as an async context manager."""
        from aioquic.asyncio import connect

        config = _make_quic_config(True)
        ProtocolClass = _make_protocol_class()
        host, port = self._remote_addr

        # Extract local port, close the socket so aioquic can rebind to it.
        _, local_port = self._local_sock.getsockname()
        self._local_sock.close()

        handshake_done: asyncio.Future = asyncio.get_event_loop().create_future()
        protocol_ref: list = []

        async def _run() -> None:
            try:
                async with connect(
                    host,
                    port,
                    configuration=config,
                    create_protocol=ProtocolClass,
                    local_port=local_port,
                ) as protocol:
                    protocol_ref.append(protocol)
                    if protocol.terminated:
                        handshake_done.set_exception(
                            ConnectionError("QUIC terminated before handshake")
                        )
                        return
                    handshake_done.set_result(None)
                    # Keep context alive until close() is called.
                    await self._close_event.wait()
            except Exception as exc:  # noqa: BLE001
                if not handshake_done.done():
                    handshake_done.set_exception(exc)

        self._task = asyncio.create_task(_run())

        try:
            await asyncio.wait_for(asyncio.shield(handshake_done), timeout=timeout)
        except asyncio.TimeoutError:
            self._close_event.set()
            raise

        if protocol_ref:
            self._protocol = protocol_ref[0]

    async def _connect_server(self, timeout: float) -> None:
        """Server side: use aioquic.asyncio.serve() and wait for first handshake."""
        from aioquic.asyncio import serve

        config = _make_quic_config(False)
        handshake_done: asyncio.Future = asyncio.get_event_loop().create_future()
        protocol_ref: list = []

        # Extract local address, close the socket so aioquic can rebind to it.
        local_host, local_port = self._local_sock.getsockname()
        self._local_sock.close()

        def _create_protocol(*args, **kwargs):
            ProtocolClass = _make_protocol_class()
            proto = ProtocolClass(*args, **kwargs)
            original_event_received = proto.quic_event_received

            def _patched(event):
                original_event_received(event)
                from aioquic.quic.events import HandshakeCompleted, ConnectionTerminated
                if isinstance(event, HandshakeCompleted) and not handshake_done.done():
                    protocol_ref.append(proto)
                    handshake_done.set_result(None)
                elif isinstance(event, ConnectionTerminated) and not handshake_done.done():
                    handshake_done.set_exception(
                        ConnectionError("QUIC terminated before handshake")
                    )

            proto.quic_event_received = _patched
            return proto

        self._server = await serve(
            local_host,
            local_port,
            configuration=config,
            create_protocol=_create_protocol,
        )

        try:
            await asyncio.wait_for(asyncio.shield(handshake_done), timeout=timeout)
        except asyncio.TimeoutError:
            self._server.close()
            raise

        if protocol_ref:
            self._protocol = protocol_ref[0]

    # ── Data channel ──────────────────────────────────────────────────────

    async def send_data(self, data: bytes) -> None:
        """Open a new unidirectional QUIC stream and send data as one frame."""
        if self._protocol is None:
            raise ConnectionError("QuicConn.send_data called before connect()")
        proto = self._protocol
        stream_id = proto._quic.get_next_available_stream_id(is_unidirectional=True)
        proto._quic.send_stream_data(stream_id, data, end_stream=True)
        proto.transmit()

    async def recv_data(self) -> bytes:
        """Wait for the next complete incoming frame from the peer."""
        if self._protocol is None:
            raise ConnectionError("QuicConn.recv_data called before connect()")
        return await self._protocol._recv_queue.get()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False
        self._close_event.set()
        if self._server is not None:
            try:
                self._server.close()
            except Exception:  # noqa: BLE001
                pass
        if self._task is not None and not self._task.done():
            self._task.cancel()
        # Socket was already closed in _connect_*, but guard against double-close.
        try:
            self._local_sock.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def is_connected(self) -> bool:
        return self._connected
