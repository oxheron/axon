from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class UserServiceState:
    coordinator_url: str
    cluster_ready: bool = False
    entry_node_ready: bool = False
    all_nodes_ready: bool = False
    backend_ready: bool = False
    pipeline_ready: bool = False
    inference_ready: bool = False
    execution_mode: Optional[str] = None
    model_name: Optional[str] = None
    selected_node_id: Optional[str] = None
    http_client: Optional[httpx.AsyncClient] = None
