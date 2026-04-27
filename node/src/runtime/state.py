from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from topology.models import NodeAssignment, StartupConfig

if TYPE_CHECKING:
    from coordinator.models import SignalReadyData
    from transport.p2p import P2PTransport
    from transport.sidecar import TransportSidecar


@dataclass
class NodeRuntimeState:
    coordinator_url: str
    node_id: str
    bind_host: str
    bind_port: int
    advertise_host: str
    advertise_port: int
    vllm_worker_port: int
    launch_vllm_worker: bool
    vllm_gpu_memory_utilization: float
    vllm_max_model_len: int
    vllm_dtype: str

    # Optional pre-shared token for coordinator WebSocket auth (B-1).
    cluster_token: str = ""

    # B-2: STUN-discovered external address for hole-punch mode.
    # Empty string means STUN has not run or port-forward mode is active.
    stun_external_addr: str = ""
    stun_external_port: int = 0
    # Set by run_stun_discovery_for_state() before ws_client sends the signal.
    stun_ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Set in lifecycle.py after transport.connect() succeeds.
    p2p_transport: Optional["P2PTransport"] = None
    # Set in lifecycle.py after the UDS sidecar starts (B-3).
    transport_sidecar: Optional["TransportSidecar"] = None
    transport_uds_path: str = ""

    startup_config: Optional[StartupConfig] = None
    assignment: Optional[NodeAssignment] = None
    startup_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Set when coordinator broadcasts signal_ready (B-1).
    signal_ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    signal_ready: Optional[SignalReadyData] = None
    launch_task: Optional[asyncio.Task] = None
    vllm_proc: Optional[asyncio.subprocess.Process] = None
    vram_gb: float = 0.0
    vllm_launch_error: Optional[str] = None
    execution_mode: str = ""
    launch_strategy: str = ""
    lifecycle_state: str = "created"
    lifecycle_detail: str = ""

    def worker_url(self) -> str:
        return f"http://{self.advertise_host}:{self.vllm_worker_port}"
