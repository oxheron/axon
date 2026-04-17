from typing import Optional

from pydantic import BaseModel

from topology.models import NodeAssignment


class NodeRegistration(BaseModel):
    node_id: str
    host: str
    port: int
    vram_gb: float
    callback_url: str
    worker_url: str


class NodeStatusUpdate(BaseModel):
    node_id: str
    cluster_id: str = ""
    lifecycle_state: str
    lifecycle_detail: str = ""
    worker_url: str = ""
    assignment: Optional[NodeAssignment] = None
