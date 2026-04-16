import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [user] %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class UserServiceState:
    coordinator_url: str
    pipeline_ready: bool = False
    inference_ready: bool = False
    model_name: Optional[str] = None
    selected_node_id: Optional[str] = None
    http_client: Optional[httpx.AsyncClient] = None


async def _poll_coordinator(state: UserServiceState) -> None:
    attempt = 0
    while True:
        try:
            response = await state.http_client.get(f"{state.coordinator_url}/status")
            response.raise_for_status()
            data = response.json()
            state.pipeline_ready = bool(data.get("pipeline_ready"))
            state.inference_ready = bool(data.get("inference_ready"))
            state.model_name = data.get("model_name")
            state.selected_node_id = data.get("selected_node_id")
        except Exception as exc:  # noqa: BLE001
            state.pipeline_ready = False
            state.inference_ready = False
            if attempt % 10 == 0:
                LOGGER.info("Waiting for coordinator: %s", exc)
        attempt += 1
        await asyncio.sleep(2.0)


def create_app(state: UserServiceState) -> FastAPI:
    app = FastAPI(title="Axon User Service", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
        )
        asyncio.create_task(_poll_coordinator(state))

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await state.http_client.aclose()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": state.pipeline_ready and state.inference_ready,
            "pipeline_ready": state.pipeline_ready,
            "inference_ready": state.inference_ready,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon user-facing API service")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_URL"),
        required=os.environ.get("COORDINATOR_URL") is None,
        help="Coordinator base URL, e.g. http://10.0.0.10:8000",
    )
    parser.add_argument("--host", default="0.0.0.0", help="User service bind host")
    parser.add_argument("--port", type=int, default=8080, help="User service bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = UserServiceState(coordinator_url=args.coordinator_url.rstrip("/"))
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
