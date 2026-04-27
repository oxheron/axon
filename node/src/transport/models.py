from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TransportMode(str, Enum):
    PORT_FORWARD = "port_forward"
    HOLE_PUNCH = "hole_punch"


@dataclass(frozen=True)
class PeerEndpoint:
    node_id: str
    external_addr: str
    external_port: int
    transport_mode: TransportMode


@dataclass
class StunResult:
    external_ip: str
    external_port: int
    nat_type: str       # raw pystun3 string, e.g. "Full Cone NAT"
    is_symmetric: bool  # True if two STUN probes returned different mapped ports
