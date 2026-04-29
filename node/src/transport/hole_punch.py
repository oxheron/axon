from __future__ import annotations

import asyncio
import logging
import socket
from typing import TYPE_CHECKING

from transport.models import PeerEndpoint, StunResult
from transport.quic_conn import QuicConn

if TYPE_CHECKING:
    from runtime.state import NodeRuntimeState

LOGGER = logging.getLogger(__name__)

STUN_PRIMARY = ("stun.l.google.com", 19302)
STUN_SECONDARY = ("stun1.l.google.com", 19302)

PUNCH_INITIAL_DELAY = 0.1
PUNCH_BACKOFF_BASE = 0.2
PUNCH_BACKOFF_MAX = 2.0
PUNCH_PAYLOAD = b"AXON-PUNCH"
PUNCH_TIMEOUT = 30.0


async def _sock_recvfrom(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, nbytes: int
) -> tuple[bytes, tuple]:
    """Async recvfrom compatible with Python < 3.11 (loop.sock_recvfrom added in 3.11)."""
    fut: asyncio.Future = loop.create_future()

    def _on_readable() -> None:
        loop.remove_reader(sock.fileno())
        try:
            result = sock.recvfrom(nbytes)
        except Exception as exc:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(exc)
        else:
            if not fut.done():
                fut.set_result(result)

    loop.add_reader(sock.fileno(), _on_readable)
    try:
        return await fut
    except asyncio.CancelledError:
        loop.remove_reader(sock.fileno())
        raise


async def _sock_sendto(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, data: bytes, addr: tuple
) -> None:
    """Async sendto compatible with Python < 3.11 (loop.sock_sendto added in 3.11).
    UDP sends almost never block; falls back to add_writer only if EAGAIN."""
    try:
        sock.sendto(data, addr)
        return
    except BlockingIOError:
        pass

    fut: asyncio.Future = loop.create_future()

    def _on_writable() -> None:
        loop.remove_writer(sock.fileno())
        try:
            sock.sendto(data, addr)
        except Exception as exc:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(exc)
        else:
            if not fut.done():
                fut.set_result(None)

    loop.add_writer(sock.fileno(), _on_writable)
    try:
        await fut
    except asyncio.CancelledError:
        loop.remove_writer(sock.fileno())
        raise


class HolePunchError(Exception):
    """Raised when UDP hole punching fails."""


def _run_stun_probe(host: str, port: int, source_port: int) -> tuple[str, str, int]:
    """Blocking STUN probe; run via asyncio.to_thread."""
    import stun  # pystun3

    nat_type, ext_ip, ext_port = stun.get_ip_info(
        stun_host=host,
        stun_port=port,
        source_ip="0.0.0.0",
        source_port=source_port,
    )
    return nat_type, ext_ip, ext_port


async def discover_external_addr(bind_port: int) -> StunResult:
    """
    Query both STUN servers sequentially and detect symmetric NAT.
    Both probes use the same source port (required for symmetric NAT detection).
    Sequential because pystun3 binds that port for each query — parallel would conflict.
    Never raises — on any failure returns StunResult(is_symmetric=True) for safe fallback.
    """
    try:
        # Sequential: pystun3 binds source_port for each probe; can't run both at once.
        nat_type1, ext_ip1, ext_port1 = await asyncio.to_thread(
            _run_stun_probe, STUN_PRIMARY[0], STUN_PRIMARY[1], bind_port
        )
        _nat_type2, _ext_ip2, ext_port2 = await asyncio.to_thread(
            _run_stun_probe, STUN_SECONDARY[0], STUN_SECONDARY[1], bind_port
        )

        is_symmetric = ext_port1 != ext_port2
        if is_symmetric:
            LOGGER.warning(
                "[transport] hole_punch: symmetric NAT detected "
                "(%s mapped port=%d, %s mapped port=%d). "
                "UDP hole punching will not work with this NAT. "
                "Set --advertise-host <external_ip> --advertise-port <forwarded_port> "
                "to use port forwarding mode, or wait for TURN relay in a future release.",
                STUN_PRIMARY[0], ext_port1,
                STUN_SECONDARY[0], ext_port2,
            )
        else:
            LOGGER.info(
                "[transport] STUN: external addr=%s:%d nat_type=%s",
                ext_ip1, ext_port1, nat_type1,
            )

        return StunResult(
            external_ip=ext_ip1,
            external_port=ext_port1,
            nat_type=nat_type1,
            is_symmetric=is_symmetric,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "[transport] STUN discovery failed: %s — treating as symmetric NAT for safety",
            exc,
        )
        return StunResult(external_ip="", external_port=0, nat_type="unknown", is_symmetric=True)


async def _probe_one_peer(
    sock: socket.socket,
    peer: PeerEndpoint,
    self_node_id: str,
) -> None:
    """
    Simultaneous open for one peer.
    Sends AXON-PUNCH probes with exponential backoff while listening for an
    incoming probe from the peer. Raises asyncio.TimeoutError if not completed
    within PUNCH_TIMEOUT.
    """
    loop = asyncio.get_running_loop()
    remote = (peer.external_addr, peer.external_port)
    received = asyncio.Event()

    async def _send_loop() -> None:
        delay = PUNCH_INITIAL_DELAY
        while not received.is_set():
            try:
                await _sock_sendto(loop, sock, PUNCH_PAYLOAD, remote)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(delay)
            delay = min(delay * 2 if delay > 0 else PUNCH_BACKOFF_BASE, PUNCH_BACKOFF_MAX)

    async def _recv_loop() -> None:
        while True:
            try:
                data, addr = await _sock_recvfrom(loop, sock, 65535)
                if data == PUNCH_PAYLOAD and addr[0] == remote[0]:
                    if addr[1] != remote[1]:
                        LOGGER.warning(
                            "[transport] hole_punch: received AXON-PUNCH from %s:%d "
                            "but expected port %d — NAT remapped the punch socket "
                            "(signaled port differs from actual send port)",
                            addr[0], addr[1], remote[1],
                        )
                    received.set()
                    return
                if data == PUNCH_PAYLOAD:
                    LOGGER.debug(
                        "[transport] hole_punch: AXON-PUNCH from unexpected source %s:%d "
                        "(expected %s:%d)",
                        addr[0], addr[1], remote[0], remote[1],
                    )
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.05)

    send_task = asyncio.create_task(_send_loop())
    recv_task = asyncio.create_task(_recv_loop())
    try:
        await asyncio.wait_for(received.wait(), timeout=PUNCH_TIMEOUT)
    finally:
        send_task.cancel()
        recv_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    LOGGER.info(
        "[transport] hole_punch: path open to peer=%s addr=%s:%d",
        peer.node_id, peer.external_addr, peer.external_port,
    )


async def hole_punch(
    node_id: str,
    bind_host: str,
    bind_port: int,
    peers: list[PeerEndpoint],
    timeout: float,
) -> dict[str, QuicConn]:
    """
    Bind UDP socket. Perform simultaneous open for all peers, then QUIC handshake.
    Role determined by lexicographic node_id comparison (same as port_forward).
    Raises HolePunchError on failure.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((bind_host, bind_port))
    sock.setblocking(False)

    LOGGER.info("[transport] hole_punch: bound udp=%s:%d", bind_host, bind_port)

    # Simultaneous open for all peers concurrently.
    try:
        await asyncio.gather(
            *(_probe_one_peer(sock, peer, node_id) for peer in peers)
        )
    except asyncio.TimeoutError:
        sock.close()
        peer_strs = ", ".join(
            f"{p.node_id} ({p.external_addr}:{p.external_port})" for p in peers
        )
        msg = (
            f"[transport] hole_punch: timed out after {PUNCH_TIMEOUT}s waiting for peers: "
            f"{peer_strs}. NAT punch failed. Consider using --advertise-port with a manually "
            "forwarded UDP port, or wait for TURN relay in a future release."
        )
        LOGGER.error(msg)
        raise HolePunchError(msg)
    except Exception as exc:
        sock.close()
        raise HolePunchError(f"hole punch error: {exc}") from exc

    # UDP path is open — proceed to QUIC handshake on the same socket.
    conns: dict[str, QuicConn] = {}
    for peer in peers:
        is_client = node_id < peer.node_id
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
        raise HolePunchError(f"QUIC handshake after hole punch failed: {exc}") from exc

    return conns


async def run_stun_discovery_for_state(state: NodeRuntimeState) -> None:
    """
    Run STUN discovery eagerly at app startup.
    Port-forward mode: sets stun_ready_event immediately (no STUN needed).
    Hole-punch mode: runs STUN, stores result in state, then sets stun_ready_event.
    Always sets stun_ready_event in the finally block so ws_client is never blocked.
    """
    is_port_forward = state.advertise_port != state.bind_port
    if is_port_forward:
        state.stun_ready_event.set()
        return

    try:
        result = await discover_external_addr(state.bind_port)
        if not result.is_symmetric:
            state.stun_external_addr = result.external_ip
            state.stun_external_port = result.external_port
            LOGGER.info(
                "[transport] STUN discovery complete: external=%s:%d",
                result.external_ip, result.external_port,
            )
        else:
            LOGGER.warning(
                "[transport] STUN indicates symmetric NAT — "
                "signal will use LAN address; hole punch will likely fail"
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[transport] STUN discovery error: %s — will signal LAN address", exc)
    finally:
        state.stun_ready_event.set()
