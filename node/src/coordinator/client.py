from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from coordinator.models import NodeRegistration, NodeStatusUpdate
from runtime.state import NodeRuntimeState
from topology.models import NodeAssignment

LOGGER = logging.getLogger(__name__)


async def report_node_status(
    state: NodeRuntimeState,
    lifecycle_state: str,
    *,
    detail: str = "",
    assignment: Optional[NodeAssignment] = None,
) -> None:
    state.lifecycle_state = lifecycle_state
    state.lifecycle_detail = detail
    payload = NodeStatusUpdate(
        node_id=state.node_id,
        cluster_id=state.startup_config.cluster_id if state.startup_config else "",
        lifecycle_state=lifecycle_state,
        lifecycle_detail=detail,
        worker_url=state.worker_url(),
        assignment=assignment,
    )
    endpoint = f"{state.coordinator_url}/node-status"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, 4):
            try:
                response = await client.post(
                    endpoint, json=payload.model_dump(exclude_none=True)
                )
                response.raise_for_status()
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Node status update failed (%s attempt=%d): %s",
                    lifecycle_state,
                    attempt,
                    exc,
                )
                await asyncio.sleep(attempt)


async def wait_for_worker_health(worker_url: str, timeout: float = 120.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    health_url = f"{worker_url.rstrip('/')}/health"

    async with httpx.AsyncClient(timeout=5.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(2)

    return False


async def register_loop(state: NodeRuntimeState) -> None:
    payload = NodeRegistration(
        node_id=state.node_id,
        host=state.advertise_host,
        port=state.advertise_port,
        vram_gb=state.vram_gb,
        callback_url=f"http://{state.advertise_host}:{state.advertise_port}",
        worker_url=state.worker_url(),
    )
    endpoint = f"{state.coordinator_url}/register"

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                response = await client.post(endpoint, json=payload.model_dump())
                if response.status_code == 409:
                    LOGGER.error(
                        "Registration rejected: coordinator already triggered cluster "
                        "startup (409). Another node registered first, or this coordinator "
                        "is a stale instance — stop old coordinators on this URL and retry."
                    )
                    return
                response.raise_for_status()
                LOGGER.info("Registered successfully with coordinator.")
                await report_node_status(
                    state,
                    "registered",
                    detail="Node registration completed.",
                )
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Registration failed (%s), retrying...", exc)
                await asyncio.sleep(2)
