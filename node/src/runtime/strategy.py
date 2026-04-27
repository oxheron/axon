from __future__ import annotations

from dataclasses import dataclass

from runtime.state import NodeRuntimeState
from topology.assignment import (
    default_execution_mode,
    resolve_assignment,
    resolve_stage_count,
)
from topology.models import StartupConfig


@dataclass(frozen=True)
class LaunchStrategy:
    assignment_node_id: str
    assignment_stage_index: int
    assignment_stage_count: int
    assignment_role: str
    load_strategy: str
    execution_mode: str
    stage_count: int
    distributed_backend: str
    launches_worker: bool
    worker_health_required: bool
    final_lifecycle_state: str


def resolve_launch_strategy(
    config: StartupConfig, state: NodeRuntimeState
) -> LaunchStrategy:
    assignment = resolve_assignment(config, state)
    stage_count = resolve_stage_count(config)
    execution_mode = config.execution_mode or default_execution_mode(stage_count)
    if stage_count == 1:
        distributed_backend = "mp"
    else:
        distributed_backend = "external_launcher"

    if execution_mode == "dry_run":
        final_lifecycle_state = "dry_run_ready"
    else:
        final_lifecycle_state = "pipeline_ready"

    return LaunchStrategy(
        assignment_node_id=assignment.node_id,
        assignment_stage_index=assignment.stage_index,
        assignment_stage_count=assignment.stage_count,
        assignment_role=assignment.stage_role,
        load_strategy=assignment.load_strategy,
        execution_mode=execution_mode,
        stage_count=stage_count,
        distributed_backend=distributed_backend,
        launches_worker=state.launch_vllm_worker and execution_mode != "dry_run",
        worker_health_required=state.launch_vllm_worker and execution_mode != "dry_run",
        final_lifecycle_state=final_lifecycle_state,
    )
