"""Tests for transport.models — data types only, no IO."""
from __future__ import annotations

import pytest

from transport.models import PeerEndpoint, StunResult, TransportMode


def test_transport_mode_values():
    assert TransportMode.PORT_FORWARD == "port_forward"
    assert TransportMode.HOLE_PUNCH == "hole_punch"


def test_transport_mode_round_trip():
    assert TransportMode("port_forward") is TransportMode.PORT_FORWARD
    assert TransportMode("hole_punch") is TransportMode.HOLE_PUNCH


def test_peer_endpoint_frozen():
    ep = PeerEndpoint(
        node_id="node-a",
        external_addr="1.2.3.4",
        external_port=5000,
        transport_mode=TransportMode.PORT_FORWARD,
    )
    assert ep.node_id == "node-a"
    with pytest.raises((AttributeError, TypeError)):
        ep.node_id = "other"  # type: ignore[misc]


def test_stun_result_symmetric_flag():
    sym = StunResult(external_ip="1.2.3.4", external_port=5000, nat_type="Symmetric NAT", is_symmetric=True)
    cone = StunResult(external_ip="1.2.3.4", external_port=5000, nat_type="Full Cone NAT", is_symmetric=False)
    assert sym.is_symmetric is True
    assert cone.is_symmetric is False
