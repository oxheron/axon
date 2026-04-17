from __future__ import annotations

import asyncio
import json
import logging

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from coordinator.poll import poll_coordinator
from runtime.state import UserServiceState

LOGGER = logging.getLogger(__name__)


def create_app(state: UserServiceState) -> FastAPI:
    app = FastAPI(title="Axon User Service", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
        )
        asyncio.create_task(poll_coordinator(state))

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await state.http_client.aclose()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": state.pipeline_ready and state.inference_ready,
            "cluster_ready": state.cluster_ready,
            "entry_node_ready": state.entry_node_ready,
            "all_nodes_ready": state.all_nodes_ready,
            "backend_ready": state.backend_ready,
            "pipeline_ready": state.pipeline_ready,
            "inference_ready": state.inference_ready,
            "execution_mode": state.execution_mode,
            "model_name": state.model_name,
            "selected_node_id": state.selected_node_id,
        }

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_v1(request: Request, path: str):
        if not state.pipeline_ready:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        if not state.inference_ready:
            raise HTTPException(status_code=503, detail="Inference worker not ready")

        target = f"{state.coordinator_url}/v1/{path}"
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in ("host", "content-length")
        }

        try:
            body_json = json.loads(body) if body else {}
        except Exception:  # noqa: BLE001
            body_json = {}
        is_streaming = bool(body_json.get("stream", False))

        try:
            if is_streaming:

                async def _stream_gen():
                    async with state.http_client.stream(
                        request.method,
                        target,
                        content=body,
                        headers=headers,
                        params=dict(request.query_params),
                    ) as upstream:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk

                return StreamingResponse(_stream_gen(), media_type="text/event-stream")

            upstream = await state.http_client.request(
                request.method,
                target,
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            LOGGER.error("Coordinator unreachable: %s", exc)
            raise HTTPException(status_code=502, detail="Coordinator unreachable")

    return app
