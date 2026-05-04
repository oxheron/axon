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
PUNCH_POST_RECV_PROBES = 8   # extra probes after receiving peer's probe so the peer can complete its own punch
PUNCH_MAX_ATTEMPTS = 3
PUNCH_RETRY_DELAY = 2.0


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


async def _probe_send_loop(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    peer: PeerEndpoint,
    received: asyncio.Event,
) -> None:
    """
    Send AXON-PUNCH probes to one peer with exponential backoff until the peer's probe
    arrives, then send PUNCH_POST_RECV_PROBES more probes so the peer can complete its
    own punch (avoids the race where we cancel our send loop before the peer has received
    any of our probes).
    """
    remote = (peer.external_addr, peer.external_port)
    delay = PUNCH_INITIAL_DELAY
    probe_count = 0
    while not received.is_set():
        try:
            await _sock_sendto(loop, sock, PUNCH_PAYLOAD, remote)
            probe_count += 1
            if probe_count == 1 or probe_count % 10 == 0:
                LOGGER.info(
                    "[transport] hole_punch: sent probe #%d to peer=%s target=%s:%d",
                    probe_count, peer.node_id, peer.external_addr, peer.external_port,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[transport] hole_punch: send error to %s:%d — %s",
                peer.external_addr, peer.external_port, exc,
            )
        await asyncio.sleep(delay)
        delay = min(delay * 2 if delay > 0 else PUNCH_BACKOFF_BASE, PUNCH_BACKOFF_MAX)
    # Drain: keep sending a short burst so the peer's recv loop sees at least one probe.
    for _ in range(PUNCH_POST_RECV_PROBES):
        try:
            await _sock_sendto(loop, sock, PUNCH_PAYLOAD, remote)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(PUNCH_INITIAL_DELAY)


async def _probe_recv_loop(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    peers: list[PeerEndpoint],
    received: dict[str, asyncio.Event],
) -> None:
    """
    Single shared recv loop for all peers.
    Having one reader on the socket avoids the fd-reader conflict that arises when
    multiple concurrent _probe tasks each call loop.add_reader on the same fd —
    asyncio silently replaces the previous handler, starving all but the last peer.
    Accepts AXON-PUNCH from an unexpected source IP (single-peer CGNAT fallback).
    """
    addr_to_peer: dict[str, PeerEndpoint] = {p.external_addr: p for p in peers}
    # Peers still waiting for their first probe (used for the single-peer CGNAT fallback).
    unresolved = [p for p in peers]
    while True:
        try:
            data, addr = await _sock_recvfrom(loop, sock, 65535)
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.05)
            continue

        if data != PUNCH_PAYLOAD:
            LOGGER.debug(
                "[transport] hole_punch: non-punch packet from %s:%d len=%d",
                addr[0], addr[1], len(data),
            )
            continue

        peer = addr_to_peer.get(addr[0])
        if peer is None:
            # Source IP doesn't match any signaled address — could be CGNAT remapping.
            # If there is exactly one peer still unresolved, accept it optimistically.
            still_waiting = [p for p in unresolved if not received[p.node_id].is_set()]
            if len(still_waiting) == 1:
                peer = still_waiting[0]
                LOGGER.warning(
                    "[transport] hole_punch: AXON-PUNCH from unexpected source %s:%d "
                    "(expected %s:%d) — accepting as peer=%s (possible CGNAT remapping)",
                    addr[0], addr[1], peer.external_addr, peer.external_port, peer.node_id,
                )
            else:
                LOGGER.warning(
                    "[transport] hole_punch: AXON-PUNCH from unknown source %s:%d "
                    "(known peers: %s) — ignoring",
                    addr[0], addr[1],
                    ", ".join(f"{p.external_addr}:{p.external_port}" for p in peers),
                )
                continue
        elif addr[1] != peer.external_port:
            LOGGER.warning(
                "[transport] hole_punch: AXON-PUNCH from %s:%d but expected port %d "
                "— NAT remapped the punch socket (signaled port differs from actual send port)",
                addr[0], addr[1], peer.external_port,
            )

        if not received[peer.node_id].is_set():
            received[peer.node_id].set()
            LOGGER.info(
                "[transport] hole_punch: path open to peer=%s addr=%s:%d",
                peer.node_id, addr[0], addr[1],
            )


async def _attempt_hole_punch(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    peers: list[PeerEndpoint],
) -> None:
    """One attempt at simultaneous-open for all peers. Raises asyncio.TimeoutError on timeout."""
    for peer in peers:
        LOGGER.info(
            "[transport] hole_punch: targeting peer=%s external=%s:%d",
            peer.node_id, peer.external_addr, peer.external_port,
        )
    received: dict[str, asyncio.Event] = {p.node_id: asyncio.Event() for p in peers}

    send_tasks = [
        asyncio.create_task(_probe_send_loop(loop, sock, peer, received[peer.node_id]))
        for peer in peers
    ]
    recv_task = asyncio.create_task(_probe_recv_loop(loop, sock, peers, received))

    try:
        await asyncio.wait_for(
            asyncio.gather(*(received[p.node_id].wait() for p in peers)),
            timeout=PUNCH_TIMEOUT,
        )
    finally:
        recv_task.cancel()
        for t in send_tasks:
            t.cancel()
        for t in [recv_task, *send_tasks]:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


async def hole_punch(
    node_id: str,
    bind_host: str,
    bind_port: int,
    peers: list[PeerEndpoint],
    timeout: float,
) -> dict[str, QuicConn]:
    """
    Bind UDP socket. Perform simultaneous open for all peers (with retries), then QUIC handshake.
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

    loop = asyncio.get_running_loop()
    peer_strs = ", ".join(f"{p.node_id} ({p.external_addr}:{p.external_port})" for p in peers)

    for attempt in range(1, PUNCH_MAX_ATTEMPTS + 1):
        try:
            await _attempt_hole_punch(loop, sock, peers)
            break  # all peers punched
        except asyncio.TimeoutError:
            if attempt < PUNCH_MAX_ATTEMPTS:
                LOGGER.warning(
                    "[transport] hole_punch: attempt %d/%d timed out for peers: %s "
                    "— retrying in %.1fs",
                    attempt, PUNCH_MAX_ATTEMPTS, peer_strs, PUNCH_RETRY_DELAY,
                )
                await asyncio.sleep(PUNCH_RETRY_DELAY)
            else:
                msg = (
                    f"[transport] hole_punch: all {PUNCH_MAX_ATTEMPTS} attempts timed out "
                    f"after {PUNCH_TIMEOUT}s waiting for peers: {peer_strs}. "
                    "NAT punch failed. Consider using --advertise-port with a manually "
                    "forwarded UDP port, or wait for TURN relay in a future release."
                )
                LOGGER.error(msg)
                sock.close()
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
