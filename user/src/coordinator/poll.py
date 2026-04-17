import asyncio
import logging

from runtime.state import UserServiceState

LOGGER = logging.getLogger(__name__)


async def poll_coordinator(state: UserServiceState) -> None:
    attempt = 0
    while True:
        try:
            response = await state.http_client.get(f"{state.coordinator_url}/status")
            response.raise_for_status()
            data = response.json()
            state.cluster_ready = bool(data.get("cluster_ready"))
            state.entry_node_ready = bool(data.get("entry_node_ready"))
            state.all_nodes_ready = bool(data.get("all_nodes_ready"))
            state.backend_ready = bool(data.get("backend_ready"))
            state.pipeline_ready = bool(data.get("pipeline_ready"))
            state.inference_ready = bool(data.get("inference_ready"))
            state.execution_mode = data.get("execution_mode")
            state.model_name = data.get("model_name")
            state.selected_node_id = data.get("selected_node_id")
        except Exception as exc:  # noqa: BLE001
            state.cluster_ready = False
            state.entry_node_ready = False
            state.all_nodes_ready = False
            state.backend_ready = False
            state.pipeline_ready = False
            state.inference_ready = False
            if attempt % 10 == 0:
                LOGGER.info("Waiting for coordinator: %s", exc)
        attempt += 1
        await asyncio.sleep(2.0)
