from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from transport.hole_punch import HolePunchError, hole_punch
from transport.models import PeerEndpoint, TransportMode
from transport.port_forward import connect_port_forward

if TYPE_CHECKING:
    import torch
    from transport.quic_conn import QuicConn

LOGGER = logging.getLogger(__name__)

_SEND_RECV_TIMEOUT = 30.0  # seconds


class P2PTransport:
    """
    Top-level P2P transport. Dispatches to port_forward or hole_punch mode,
    completes QUIC handshakes with all peers, and holds QuicConn per peer.
    send_tensor / recv_tensor route over the established QUIC connections.
    """

    def __init__(
        self,
        node_id: str,
        bind_host: str,
        bind_port: int,
        advertise_port: int,
        transport_mode: TransportMode,
    ) -> None:
        self._node_id = node_id
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._advertise_port = advertise_port
        self._transport_mode = transport_mode
        self._conns: dict[str, QuicConn] = {}
        self._connected = False
        # Captured at connect() time so synchronous callers (B-3 sidecar) can
        # dispatch coroutines back onto the node's event loop.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, peers: list[PeerEndpoint], timeout: float = 30.0) -> None:
        """
        Establish QUIC connections to all peers.
        Raises ConnectionError on failure.
        """
        self._loop = asyncio.get_running_loop()

        if self._transport_mode == TransportMode.PORT_FORWARD:
            conns = await connect_port_forward(
                node_id=self._node_id,
                bind_host=self._bind_host,
                bind_port=self._bind_port,
                peers=peers,
                timeout=timeout,
            )
            self._conns = conns
            self._connected = True
            LOGGER.info(
                "[transport] port_forward mode: QUIC handshake complete with %d peer(s)",
                len(conns),
            )
        else:
            try:
                conns = await hole_punch(
                    node_id=self._node_id,
                    bind_host=self._bind_host,
                    bind_port=self._bind_port,
                    peers=peers,
                    timeout=timeout,
                )
            except HolePunchError as exc:
                raise ConnectionError(str(exc)) from exc
            self._conns = conns
            self._connected = True
            LOGGER.info(
                "[transport] hole_punch mode: QUIC handshake complete with %d peer(s)",
                len(conns),
            )

    def send_tensor(self, peer_id: str, tensor: "torch.Tensor", *, stream_id: int) -> None:
        """Synchronously send a tensor to peer over QUIC (blocks until sent)."""
        if self._loop is None:
            raise ConnectionError("P2PTransport.send_tensor called before connect()")
        from axon_quic.wire import tensor_to_bytes
        import os
        wire_dtype = os.environ.get("AXON_WIRE_DTYPE", "fp8")
        data = tensor_to_bytes(tensor, stream_id, wire_dtype=wire_dtype)
        fut = asyncio.run_coroutine_threadsafe(
            self._conns[peer_id].send_data(data), self._loop
        )
        fut.result(timeout=_SEND_RECV_TIMEOUT)

    def recv_tensor(self, peer_id: str, *, stream_id: int) -> "torch.Tensor":
        """Synchronously receive a tensor from peer over QUIC (blocks until received)."""
        if self._loop is None:
            raise ConnectionError("P2PTransport.recv_tensor called before connect()")
        from axon_quic.wire import bytes_to_tensor_bf16
        fut = asyncio.run_coroutine_threadsafe(
            self._conns[peer_id].recv_data(), self._loop
        )
        data = fut.result(timeout=_SEND_RECV_TIMEOUT)
        return bytes_to_tensor_bf16(data)

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
