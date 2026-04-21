from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from coordinator.client import register_loop, report_node_status
from hardware import detect_vram_gb
from runtime.lifecycle import handle_cluster_start
from runtime.state import NodeRuntimeState
from runtime.strategy import resolve_launch_strategy
from topology.assignment import resolve_assignment
from topology.models import StartupConfig

LOGGER = logging.getLogger(__name__)


def create_app(state: NodeRuntimeState) -> FastAPI:
    app = FastAPI(title="Axon Node Service", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        state.vram_gb = detect_vram_gb()
        LOGGER.info("Detected VRAM: %.2f GiB", state.vram_gb)
        asyncio.create_task(register_loop(state))

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        if state.launch_task is not None and not state.launch_task.done():
            state.launch_task.cancel()
            try:
                await state.launch_task
            except asyncio.CancelledError:
                pass
        if state.vllm_proc is not None and state.vllm_proc.returncode is None:
            LOGGER.info("Terminating vLLM worker pid=%s", state.vllm_proc.pid)
            state.vllm_proc.terminate()
            try:
                await asyncio.wait_for(state.vllm_proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                LOGGER.warning("vLLM worker did not exit cleanly; killing")
                state.vllm_proc.kill()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "node_id": state.node_id,
            "registered_vram_gb": state.vram_gb,
            "worker_url": state.worker_url(),
            "ray_joined": state.ray_joined,
            "startup_received": state.startup_config is not None,
            "cluster_id": state.startup_config.cluster_id if state.startup_config else "",
            "execution_mode": state.execution_mode,
            "launch_strategy": state.launch_strategy,
            "lifecycle_state": state.lifecycle_state,
            "lifecycle_detail": state.lifecycle_detail,
            "assignment": (
                state.assignment.model_dump()
                if state.assignment is not None
                else None
            ),
            "vllm_worker_running": state.vllm_proc is not None
            and state.vllm_proc.returncode is None,
        }

    @app.post("/startup")
    async def receive_startup(config: StartupConfig) -> dict:
        if state.startup_config is not None:
            return {"accepted": True, "duplicate": True}

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
        return {"accepted": True, "duplicate": False}

    @app.get("/status")
    async def status() -> dict:
        return {
            "node_id": state.node_id,
            "vram_gb": state.vram_gb,
            "worker_url": state.worker_url(),
            "ray_joined": state.ray_joined,
            "execution_mode": state.execution_mode,
            "launch_strategy": state.launch_strategy,
            "lifecycle_state": state.lifecycle_state,
            "lifecycle_detail": state.lifecycle_detail,
            "startup_config": state.startup_config.model_dump()
            if state.startup_config
            else None,
            "assignment": (
                state.assignment.model_dump()
                if state.assignment is not None
                else None
            ),
            "vllm_pid": state.vllm_proc.pid if state.vllm_proc else None,
            "vllm_launch_error": state.vllm_launch_error,
        }

    return app
