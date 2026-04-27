from __future__ import annotations

import asyncio
import logging
import socket

from transport.models import PeerEndpoint
from transport.quic_conn import QuicConn

LOGGER = logging.getLogger(__name__)


def _bind_udp_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    return sock


async def connect_port_forward(
    node_id: str,
    bind_host: str,
    bind_port: int,
    peers: list[PeerEndpoint],
    timeout: float,
) -> dict[str, QuicConn]:
    """
    Bind one UDP socket and establish QUIC connections to all peers.
    Role (client/server) determined by lexicographic node_id comparison.
    """
    sock = _bind_udp_socket(bind_host, bind_port)
    LOGGER.info("[transport] port_forward: bound udp=%s:%d", bind_host, bind_port)

    conns: dict[str, QuicConn] = {}
    for peer in peers:
        is_client = node_id < peer.node_id
        LOGGER.info(
            "[transport] port_forward: peer=%s addr=%s:%d role=%s",
            peer.node_id,
            peer.external_addr,
            peer.external_port,
            "client" if is_client else "server",
        )
        conn = QuicConn(
            peer_id=peer.node_id,
            local_sock=sock,
            remote_addr=(peer.external_addr, peer.external_port),
            is_client=is_client,
        )
        conns[peer.node_id] = conn

    try:
        await asyncio.gather(
            *(conn.connect(timeout=timeout) for conn in conns.values())
        )
    except Exception as exc:
        for conn in conns.values():
            conn.close()
        raise ConnectionError(f"port_forward QUIC handshake failed: {exc}") from exc

    return conns
