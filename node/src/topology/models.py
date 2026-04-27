from typing import Optional

from pydantic import BaseModel, Field


class StartupConfig(BaseModel):
    cluster_id: str = ""
    model_name: str
    execution_mode: str = "vllm_slice"
    pipeline_parallel_size: int = Field(..., ge=1)
    stage_count: int = Field(1, ge=1)
    entry_node_id: str = ""
    nodes: list["TopologyNode"] = Field(default_factory=list)
    assignment: Optional["NodeAssignment"] = None
    backend_config: "BackendConfig" = Field(default_factory=lambda: BackendConfig())


class TopologyNode(BaseModel):
    node_id: str
    host: str
    port: int
    vram_gb: float = 0.0
    callback_url: str
    worker_url: str
    stage_index: Optional[int] = None
    stage_role: str = ""


class PeerNode(BaseModel):
    node_id: str
    stage_index: int = Field(..., ge=0)
    stage_role: str
    worker_url: str


class SliceSpec(BaseModel):
    kind: str = "stage_index"
    stage_index: int = Field(0, ge=0)
    stage_count: int = Field(1, ge=1)
    partition_label: str = ""
    executable: bool = True


class NodeAssignment(BaseModel):
    node_id: str
    stage_index: int = Field(..., ge=0)
    stage_count: int = Field(..., ge=1)
    stage_role: str
    load_strategy: str
    slice_spec: SliceSpec = Field(default_factory=SliceSpec)
    peer_nodes: list[PeerNode] = Field(default_factory=list)
    worker_endpoint: str = ""


class BackendConfig(BaseModel):
    env_overrides: dict[str, str] = Field(default_factory=dict)
    launch_args: list[str] = Field(default_factory=list)


StartupConfig.model_rebuild()
