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
    format="%(asctime)s %(levelname)s [server] %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class ServerState:
    coordinator_url: str
    vllm_worker_url: str
    pipeline_ready: bool = False
    model_name: Optional[str] = None
    http_client: Optional[httpx.AsyncClient] = None


async def _poll_coordinator(state: ServerState) -> None:
    attempt = 0
    while True:
        try:
            r = await state.http_client.get(f"{state.coordinator_url}/status")
            data = r.json()
            if data.get("pipeline_ready"):
                state.pipeline_ready = True
                state.model_name = data.get("model_name")
                LOGGER.info("Pipeline ready. Model: %s", state.model_name)
                return
        except Exception as exc:  # noqa: BLE001
            if attempt % 10 == 0:
                LOGGER.info("Waiting for coordinator: %s", exc)
        attempt += 1
        await asyncio.sleep(2.0)


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Axon Inference Server", version="0.1.0")

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
            "ok": state.pipeline_ready,
            "pipeline_ready": state.pipeline_ready,
            "model_name": state.model_name,
        }

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_v1(request: Request, path: str):
        if not state.pipeline_ready:
            raise HTTPException(status_code=503, detail="Pipeline not ready")

        target = f"{state.vllm_worker_url}/v1/{path}"
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
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
            else:
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
            LOGGER.error("vLLM worker unreachable: %s", exc)
            raise HTTPException(status_code=502, detail="vLLM worker unreachable")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon OpenAI-compatible inference server")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_URL"),
        required=os.environ.get("COORDINATOR_URL") is None,
        help="Coordinator base URL, e.g. http://10.0.0.10:8000",
    )
    parser.add_argument(
        "--vllm-worker-url",
        default=os.environ.get("VLLM_WORKER_URL", "http://localhost:8100"),
        help="vLLM worker base URL to proxy requests to (default: http://localhost:8100)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8080, help="Server bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = ServerState(
        coordinator_url=args.coordinator_url.rstrip("/"),
        vllm_worker_url=args.vllm_worker_url.rstrip("/"),
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
