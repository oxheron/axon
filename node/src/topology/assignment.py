from __future__ import annotations

from runtime.state import NodeRuntimeState
from topology.models import NodeAssignment, PeerNode, SliceSpec, StartupConfig


def infer_stage_role(stage_index: int, stage_count: int) -> str:
    if stage_count <= 1 or stage_index == 0:
        return "entry"
    if stage_index == stage_count - 1:
        return "final"
    return "middle"


def default_execution_mode(_stage_count: int) -> str:
    return "vllm_slice"


def default_load_strategy(execution_mode: str, _stage_count: int) -> str:
    if execution_mode == "coordinator_slice":
        return "coordinator_slice"
    if execution_mode == "dry_run":
        return "dry_run"
    return "vllm_slice"


def resolve_stage_count(config: StartupConfig) -> int:
    return max(1, config.stage_count or config.pipeline_parallel_size)


def build_default_assignment(
    config: StartupConfig, state: NodeRuntimeState
) -> NodeAssignment:
    stage_count = resolve_stage_count(config)
    execution_mode = config.execution_mode or default_execution_mode(stage_count)
    stage_index = 0
    stage_role = infer_stage_role(stage_index, stage_count)
    peer_nodes: list[PeerNode] = []

    if config.nodes:
        for idx, node in enumerate(config.nodes):
            if node.node_id == state.node_id:
                stage_index = node.stage_index if node.stage_index is not None else idx
                stage_role = node.stage_role or infer_stage_role(stage_index, stage_count)
                if idx > 0:
                    prev = config.nodes[idx - 1]
                    peer_nodes.append(
                        PeerNode(
                            node_id=prev.node_id,
                            stage_index=prev.stage_index
                            if prev.stage_index is not None
                            else idx - 1,
                            stage_role=prev.stage_role
                            or infer_stage_role(idx - 1, stage_count),
                            worker_url=prev.worker_url,
                        )
                    )
                if idx + 1 < len(config.nodes):
                    nxt = config.nodes[idx + 1]
                    peer_nodes.append(
                        PeerNode(
                            node_id=nxt.node_id,
                            stage_index=nxt.stage_index
                            if nxt.stage_index is not None
                            else idx + 1,
                            stage_role=nxt.stage_role
                            or infer_stage_role(idx + 1, stage_count),
                            worker_url=nxt.worker_url,
                        )
                    )
                break

    return NodeAssignment(
        node_id=state.node_id,
        stage_index=stage_index,
        stage_count=stage_count,
        stage_role=stage_role,
        load_strategy=default_load_strategy(execution_mode, stage_count),
        slice_spec=SliceSpec(
            stage_index=stage_index,
            stage_count=stage_count,
            partition_label=f"{stage_index}/{stage_count}",
            executable=execution_mode != "dry_run",
        ),
        peer_nodes=peer_nodes,
        worker_endpoint=f"http://{state.advertise_host}:{state.vllm_worker_port}",
    )


def resolve_assignment(config: StartupConfig, state: NodeRuntimeState) -> NodeAssignment:
    if config.assignment is not None:
        return config.assignment
    return build_default_assignment(config, state)
