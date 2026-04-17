from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from topology.models import NodeAssignment, StartupConfig


@dataclass
class NodeRuntimeState:
    coordinator_url: str
    node_id: str
    bind_host: str
    bind_port: int
    advertise_host: str
    vllm_worker_port: int
    launch_vllm_worker: bool
    vllm_gpu_memory_utilization: float
    vllm_max_model_len: int
    vllm_dtype: str

    startup_config: Optional[StartupConfig] = None
    assignment: Optional[NodeAssignment] = None
    startup_event: asyncio.Event = field(default_factory=asyncio.Event)
    launch_task: Optional[asyncio.Task] = None
    vllm_proc: Optional[asyncio.subprocess.Process] = None
    ray_joined: bool = False
    vram_gb: float = 0.0
    vllm_launch_error: Optional[str] = None
    execution_mode: str = ""
    launch_strategy: str = ""
    lifecycle_state: str = "created"
    lifecycle_detail: str = ""

    def worker_url(self) -> str:
        return f"http://{self.advertise_host}:{self.vllm_worker_port}"
