"""Tests for P2PTransport dispatch and interface."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from transport.models import PeerEndpoint, TransportMode
from transport.p2p import P2PTransport


def _make_transport(mode: TransportMode) -> P2PTransport:
    return P2PTransport(
        node_id="node-a",
        bind_host="0.0.0.0",
        bind_port=9000,
        advertise_port=9000 if mode == TransportMode.HOLE_PUNCH else 9001,
        transport_mode=mode,
    )


def _peer(node_id: str = "node-b") -> PeerEndpoint:
    return PeerEndpoint(
        node_id=node_id,
        external_addr="127.0.0.1",
        external_port=9001,
        transport_mode=TransportMode.PORT_FORWARD,
    )


@pytest.mark.asyncio
async def test_dispatch_port_forward():
    """PORT_FORWARD mode calls connect_port_forward, not hole_punch."""
    fake_conn = MagicMock()
    fake_conn.is_connected = True

    with (
        patch(
            "transport.p2p.connect_port_forward",
            new=AsyncMock(return_value={"node-b": fake_conn}),
        ) as mock_pf,
        patch("transport.p2p.discover_external_addr") as mock_stun,
    ):
        t = _make_transport(TransportMode.PORT_FORWARD)
        await t.connect([_peer()], timeout=5.0)

    mock_pf.assert_awaited_once()
    mock_stun.assert_not_called()
    assert t.is_connected is True


@pytest.mark.asyncio
async def test_dispatch_hole_punch_cone_nat():
    """HOLE_PUNCH mode with non-symmetric STUN calls hole_punch()."""
    from transport.models import StunResult

    fake_conn = MagicMock()
    fake_conn.is_connected = True
    non_sym = StunResult(external_ip="5.5.5.5", external_port=54321,
                         nat_type="Full Cone NAT", is_symmetric=False)

    with (
        patch("transport.p2p.discover_external_addr", new=AsyncMock(return_value=non_sym)),
        patch(
            "transport.p2p.hole_punch",
            new=AsyncMock(return_value={"node-b": fake_conn}),
        ) as mock_hp,
    ):
        t = _make_transport(TransportMode.HOLE_PUNCH)
        await t.connect([_peer()], timeout=5.0)

    mock_hp.assert_awaited_once()
    assert t.is_connected is True


@pytest.mark.asyncio
async def test_dispatch_hole_punch_symmetric_raises():
    """Symmetric NAT detected → ConnectionError raised immediately."""
    from transport.models import StunResult

    sym = StunResult(external_ip="", external_port=0,
                     nat_type="Symmetric NAT", is_symmetric=True)

    with patch("transport.p2p.discover_external_addr", new=AsyncMock(return_value=sym)):
        t = _make_transport(TransportMode.HOLE_PUNCH)
        with pytest.raises(ConnectionError, match="Symmetric NAT"):
            await t.connect([_peer()], timeout=5.0)

    assert t.is_connected is False


def test_close_idempotent():
    t = _make_transport(TransportMode.PORT_FORWARD)
    fake_conn = MagicMock()
    t._conns = {"node-b": fake_conn}
    t._connected = True

    t.close()
    t.close()  # second call must not raise

    fake_conn.close.assert_called_once()
    assert t.is_connected is False


def test_send_tensor_raises_before_connect():
    t = _make_transport(TransportMode.PORT_FORWARD)
    with pytest.raises(ConnectionError):
        t.send_tensor("node-b", MagicMock(), stream_id=0)


def test_recv_tensor_raises_before_connect():
    t = _make_transport(TransportMode.PORT_FORWARD)
    with pytest.raises(ConnectionError):
        t.recv_tensor("node-b", stream_id=0)
