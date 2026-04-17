from __future__ import annotations

import asyncio
import logging
import os

from coordinator.client import report_node_status, wait_for_worker_health
from hardware import (
    _preflight_vllm_for_accelerator,
    build_vllm_worker_environ,
    detect_torch_accelerator,
)
from runtime.state import NodeRuntimeState
from runtime.strategy import LaunchStrategy, resolve_launch_strategy
from topology.assignment import resolve_assignment
from workers.ray_worker import join_ray_cluster, start_vllm_worker_process

LOGGER = logging.getLogger(__name__)


async def _run_dry_run_strategy(
    state: NodeRuntimeState, strategy: LaunchStrategy
) -> None:
    assert state.assignment is not None
    await report_node_status(
        state,
        "slice_loaded",
        detail=(
            "Validated dry-run topology and assignment semantics without "
            "joining Ray or launching a worker."
        ),
        assignment=state.assignment,
    )
    await report_node_status(
        state,
        strategy.final_lifecycle_state,
        detail=(
            "Dry-run validation completed; no executable backend worker was started."
        ),
        assignment=state.assignment,
    )


async def _join_backend_runtime(
    state: NodeRuntimeState, strategy: LaunchStrategy
) -> bool:
    assert state.assignment is not None
    if not strategy.requires_ray:
        LOGGER.info("Single-node mode: skipping Ray cluster join, using mp backend.")
        return True
    if not strategy.ray_head_address:
        state.vllm_launch_error = "missing_ray_head_address"
        await report_node_status(
            state,
            "failed",
            detail="Startup plan requires multi-node backend but no Ray head address was provided.",
            assignment=state.assignment,
        )
        return False

    state.ray_joined = await join_ray_cluster(strategy.ray_head_address)
    if not state.ray_joined:
        state.vllm_launch_error = "ray_join_failed"
        await report_node_status(
            state,
            "failed",
            detail=f"Failed to join Ray cluster at {strategy.ray_head_address}.",
            assignment=state.assignment,
        )
        return False
    await report_node_status(
        state,
        "backend_joined",
        detail=f"Joined Ray cluster at {strategy.ray_head_address}.",
        assignment=state.assignment,
    )
    return True


async def _launch_worker_for_strategy(
    state: NodeRuntimeState, strategy: LaunchStrategy, model_name: str
) -> bool:
    assert state.assignment is not None
    probe_timeout = float(os.environ.get("AXON_TORCH_ACCEL_PROBE_TIMEOUT", "180"))
    LOGGER.info(
        "Probing PyTorch accelerator (timeout %.0fs). "
        "First HIP/CUDA driver init can be slow; this runs off the request loop.",
        probe_timeout,
    )
    try:
        accel = await asyncio.wait_for(
            asyncio.to_thread(detect_torch_accelerator),
            timeout=probe_timeout,
        )
    except asyncio.TimeoutError:
        state.vllm_launch_error = f"torch_accelerator_probe_timeout_{int(probe_timeout)}s"
        LOGGER.error(
            "Torch accelerator probe timed out after %.0fs. "
            "Fix the GPU stack, then retry.",
            probe_timeout,
        )
        await report_node_status(
            state,
            "failed",
            detail=state.vllm_launch_error,
            assignment=state.assignment,
        )
        return False

    LOGGER.info("PyTorch accelerator for vLLM subprocess: %s", accel)
    await asyncio.to_thread(_preflight_vllm_for_accelerator, accel)
    vllm_env = build_vllm_worker_environ(accel)
    vllm_env.update(state.startup_config.backend_config.env_overrides)
    if strategy.execution_mode == "slice_loaded_pipeline":
        LOGGER.info(
            "Using coordinator-assigned slice mode for stage=%s/%s; "
            "backend still realizes placement through distributed pipeline startup.",
            strategy.assignment_stage_index,
            strategy.assignment_stage_count,
        )
    state.vllm_proc = await start_vllm_worker_process(
        model_name=model_name,
        pipeline_parallel_size=strategy.stage_count,
        port=state.vllm_worker_port,
        gpu_memory_utilization=state.vllm_gpu_memory_utilization,
        max_model_len=state.vllm_max_model_len,
        dtype=state.vllm_dtype,
        distributed_backend=strategy.distributed_backend,
        env=vllm_env,
        extra_args=state.startup_config.backend_config.launch_args,
    )
    LOGGER.info("vLLM worker started with pid=%s", state.vllm_proc.pid)
    await report_node_status(
        state,
        "slice_loaded",
        detail=f"Started local worker pid={state.vllm_proc.pid}.",
        assignment=state.assignment,
    )
    if await wait_for_worker_health(state.worker_url()):
        await report_node_status(
            state,
            strategy.final_lifecycle_state,
            detail="Worker health endpoint is serving traffic.",
            assignment=state.assignment,
        )
        return True

    state.vllm_launch_error = "worker_health_check_failed"
    await report_node_status(
        state,
        "failed",
        detail=state.vllm_launch_error,
        assignment=state.assignment,
    )
    return False


async def handle_cluster_start(state: NodeRuntimeState) -> None:
    await state.startup_event.wait()
    assert state.startup_config is not None
    config = state.startup_config
    strategy = resolve_launch_strategy(config, state)
    state.assignment = resolve_assignment(config, state)
    state.execution_mode = strategy.execution_mode
    state.launch_strategy = strategy.load_strategy
    LOGGER.info(
        "Received startup signal: cluster=%s model=%s mode=%s stage=%s/%s strategy=%s ray=%s",
        config.cluster_id,
        config.model_name,
        strategy.execution_mode,
        strategy.assignment_stage_index,
        strategy.assignment_stage_count,
        strategy.load_strategy,
        strategy.ray_head_address,
    )
    await report_node_status(
        state,
        "load_started",
        detail=(
            f"Starting load for stage={strategy.assignment_stage_index}/"
            f"{strategy.assignment_stage_count} mode={strategy.execution_mode} "
            f"strategy={strategy.load_strategy}"
        ),
        assignment=state.assignment,
    )

    if strategy.execution_mode == "dry_run":
        await _run_dry_run_strategy(state, strategy)
        return

    if not await _join_backend_runtime(state, strategy):
        return

    if strategy.launches_worker:
        await _launch_worker_for_strategy(state, strategy, config.model_name)
        return

    await report_node_status(
        state,
        strategy.final_lifecycle_state,
        detail="Load plan processed without launching a local worker.",
        assignment=state.assignment,
    )
