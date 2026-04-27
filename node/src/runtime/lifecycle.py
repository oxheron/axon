from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from coordinator.client import report_node_status, wait_for_worker_health
from hardware import (
    _preflight_vllm_for_accelerator,
    build_vllm_worker_environ,
    detect_torch_accelerator,
)
from runtime.state import NodeRuntimeState
from runtime.strategy import LaunchStrategy, resolve_launch_strategy
from topology.assignment import resolve_assignment
from workers.vllm_worker import start_vllm_worker_process

if TYPE_CHECKING:
    from topology.models import StartupConfig

LOGGER = logging.getLogger(__name__)


async def _run_dry_run_strategy(
    state: NodeRuntimeState, strategy: LaunchStrategy
) -> None:
    assert state.assignment is not None
    await report_node_status(
        state,
        "slice_loaded",
        detail=(
            "Validated dry-run topology and assignment semantics without launching a worker."
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



def _build_peers_by_rank(state: NodeRuntimeState) -> list[str]:
    """Return peer node_ids ordered by PP stage_index (excludes this node)."""
    if state.startup_config is None or state.signal_ready is None:
        return []
    # Build a map from node_id → stage_index using startup_config.nodes
    stage_map: dict[str, int] = {}
    for node in state.startup_config.nodes:
        stage_map[node.node_id] = node.stage_index

    peer_ids = [
        p.node_id
        for p in state.signal_ready.peers
        if p.node_id != state.node_id
    ]
    # Sort by stage_index so peers_by_rank[i] == node_id for PP rank i
    peer_ids_by_rank = sorted(peer_ids, key=lambda nid: stage_map.get(nid, 999))
    # Insert this node's own position as a sentinel "" so that the peer_idx
    # passed by vLLM (which IS a rank, including self) maps correctly.
    own_stage = stage_map.get(state.node_id, 0)
    result: list[str] = []
    peer_it = iter(peer_ids_by_rank)
    all_stages = sorted(stage_map.keys(), key=lambda nid: stage_map[nid])
    for nid in all_stages:
        if nid == state.node_id:
            result.append("")  # self — sidecar ignores sends to itself
        else:
            result.append(next(peer_it, ""))
    return result


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
    if strategy.execution_mode == "coordinator_slice":
        LOGGER.info(
            "Using coordinator-assigned slice mode for stage=%s/%s; "
            "backend still realizes placement through distributed pipeline startup.",
            strategy.assignment_stage_index,
            strategy.assignment_stage_count,
        )

    # B-3: Build transport env for the vLLM subprocess when the P2P sidecar is active.
    transport_env: dict[str, str] | None = None
    if state.p2p_transport and state.transport_uds_path and state.assignment:
        transport_env = {
            "AXON_TRANSPORT_UDS": state.transport_uds_path,
            "AXON_PP_RANK": str(state.assignment.stage_index),
            "AXON_PP_SIZE": str(state.assignment.stage_count),
            "AXON_PP_BACKEND": "axon_quic",
            "AXON_COORDINATOR_URL": state.coordinator_url,
            "AXON_CLUSTER_ID": state.startup_config.cluster_id,
            "AXON_WIRE_DTYPE": os.environ.get("AXON_WIRE_DTYPE", "fp8"),
        }
        LOGGER.info(
            "[lifecycle] transport env: rank=%s size=%s uds=%s",
            transport_env["AXON_PP_RANK"],
            transport_env["AXON_PP_SIZE"],
            state.transport_uds_path,
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
        transport_env=transport_env,
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


async def apply_startup_config(
    state: NodeRuntimeState, config: StartupConfig
) -> bool:
    """
    Apply a received StartupConfig.  Idempotent — returns False if already applied.

    Called from both the HTTP /startup endpoint and the WS receive loop so that
    startup_config delivery works regardless of which channel arrives first.
    """
    if state.startup_config is not None:
        return False

    state.startup_config = config
    state.assignment = resolve_assignment(config, state)
    strategy = resolve_launch_strategy(config, state)
    state.execution_mode = strategy.execution_mode
    state.launch_strategy = strategy.load_strategy
    state.startup_event.set()

    asyncio.create_task(
        report_node_status(
            state,
            "assigned",
            detail=(
                f"Received load plan stage={state.assignment.stage_index}/"
                f"{state.assignment.stage_count} role={state.assignment.stage_role}"
            ),
            assignment=state.assignment,
        )
    )
    state.launch_task = asyncio.create_task(handle_cluster_start(state))
    return True


async def handle_cluster_start(state: NodeRuntimeState) -> None:
    await state.startup_event.wait()
    assert state.startup_config is not None
    config = state.startup_config
    strategy = resolve_launch_strategy(config, state)
    state.assignment = resolve_assignment(config, state)
    state.execution_mode = strategy.execution_mode
    state.launch_strategy = strategy.load_strategy
    LOGGER.info(
        "Received startup signal: cluster=%s model=%s mode=%s stage=%s/%s strategy=%s",
        config.cluster_id,
        config.model_name,
        strategy.execution_mode,
        strategy.assignment_stage_index,
        strategy.assignment_stage_count,
        strategy.load_strategy,
    )

    # Multi-node non-dry_run modes must complete P2P signaling before loading.
    needs_signaling = (
        strategy.assignment_stage_count > 1
        and strategy.execution_mode != "dry_run"
    )
    if needs_signaling:
        await report_node_status(
            state,
            "signaling",
            detail=(
                f"Waiting for P2P signal exchange "
                f"(stage={strategy.assignment_stage_index}/{strategy.assignment_stage_count})"
            ),
            assignment=state.assignment,
        )
        await state.signal_ready_event.wait()
        LOGGER.info(
            "[signaling] complete: cluster=%s peers=%d",
            config.cluster_id,
            len(state.signal_ready.peers) if state.signal_ready else 0,
        )

        # B-2: establish P2P transport connections after signaling.
        if state.signal_ready and state.signal_ready.peers:
            from transport import P2PTransport, PeerEndpoint, TransportMode  # noqa: PLC0415

            peers = [
                PeerEndpoint(
                    node_id=p.node_id,
                    external_addr=p.external_addr,
                    external_port=p.external_port,
                    transport_mode=TransportMode(p.transport_mode),
                )
                for p in state.signal_ready.peers
                if p.node_id != state.node_id
            ]
            t_mode = (
                TransportMode.PORT_FORWARD
                if state.advertise_port != state.bind_port
                else TransportMode.HOLE_PUNCH
            )
            transport = P2PTransport(
                node_id=state.node_id,
                bind_host=state.bind_host,
                bind_port=state.bind_port,
                advertise_port=state.advertise_port,
                transport_mode=t_mode,
            )
            try:
                await transport.connect(peers, timeout=30.0)
                state.p2p_transport = transport
                LOGGER.info(
                    "[transport] connected: cluster=%s peers=%d mode=%s",
                    config.cluster_id,
                    len(peers),
                    t_mode.value,
                )
            except Exception as exc:
                LOGGER.error("[transport] connect failed: %s", exc)
                await report_node_status(
                    state,
                    "failed",
                    detail=f"P2P transport connect failed: {exc}",
                    assignment=state.assignment,
                )
                return

            # B-3: Start UDS sidecar so the vLLM subprocess can use the QUIC connections.
            try:
                from transport.sidecar import TransportSidecar  # noqa: PLC0415
                uds_path = f"/tmp/axon-{config.cluster_id}.sock"
                peers_by_rank = _build_peers_by_rank(state)
                sidecar = TransportSidecar(transport, uds_path, peers_by_rank)
                await sidecar.start()
                state.transport_sidecar = sidecar
                state.transport_uds_path = uds_path
                LOGGER.info(
                    "[sidecar] started: path=%s peers_by_rank=%s",
                    uds_path,
                    peers_by_rank,
                )
            except Exception as exc:
                LOGGER.error("[sidecar] failed to start: %s", exc)
                await report_node_status(
                    state,
                    "failed",
                    detail=f"Transport sidecar start failed: {exc}",
                    assignment=state.assignment,
                )
                return

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

    if strategy.launches_worker:
        await _launch_worker_for_strategy(state, strategy, config.model_name)
        return

    await report_node_status(
        state,
        strategy.final_lifecycle_state,
        detail="Load plan processed without launching a local worker.",
        assignment=state.assignment,
    )
