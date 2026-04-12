import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx
import ray
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [server] %(message)s",
)
LOGGER = logging.getLogger(__name__)


class StartupConfig(BaseModel):
    model_name: str
    pipeline_parallel_size: int = Field(..., ge=1)
    ray_head_address: str


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: bool = False


@dataclass
class ServerState:
    coordinator_url: str
    model_override: Optional[str]
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    dtype: Optional[str]
    config: Optional[StartupConfig] = None
    engine: Optional[AsyncLLMEngine] = None


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI multi-part content (text/image) simplified for barebones setup.
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    chunks: List[str] = []
    for msg in messages:
        content = normalize_content(msg.content).strip()
        chunks.append(f"{msg.role}: {content}")
    chunks.append("assistant:")
    return "\n".join(chunks)


async def wait_for_cluster_config(
    coordinator_url: str,
    timeout_s: float = 3600.0,
    poll_interval_s: float = 2.0,
) -> StartupConfig:
    deadline = time.time() + timeout_s
    status_url = f"{coordinator_url}/status"
    config_url = f"{coordinator_url}/config"

    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            try:
                status_resp = await client.get(status_url)
                status_resp.raise_for_status()
                status = status_resp.json()
                if status.get("pipeline_ready"):
                    cfg_resp = await client.get(config_url)
                    cfg_resp.raise_for_status()
                    return StartupConfig(**cfg_resp.json())
            except Exception as exc:  # noqa: BLE001 - continue polling until timeout.
                LOGGER.info("Waiting for coordinator readiness: %s", exc)
            await asyncio.sleep(poll_interval_s)

    raise TimeoutError("Timed out waiting for coordinator startup config.")


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Axon Inference Server", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        state.config = await wait_for_cluster_config(state.coordinator_url)
        model_name = state.model_override or state.config.model_name

        if not ray.is_initialized():
            ray.init(address=state.config.ray_head_address, ignore_reinit_error=True)
            LOGGER.info("Connected to Ray cluster at %s", state.config.ray_head_address)

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            pipeline_parallel_size=state.config.pipeline_parallel_size,
            distributed_executor_backend="ray",
            gpu_memory_utilization=state.gpu_memory_utilization,
            max_model_len=state.max_model_len,
            dtype=state.dtype,
        )
        state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        LOGGER.info(
            "AsyncLLMEngine initialized with model=%s pipeline_parallel_size=%s",
            model_name,
            state.config.pipeline_parallel_size,
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": state.engine is not None,
            "pipeline_ready": state.config is not None,
            "model_name": (state.model_override or state.config.model_name)
            if state.config
            else None,
            "pipeline_parallel_size": state.config.pipeline_parallel_size
            if state.config
            else None,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if state.engine is None or state.config is None:
            raise HTTPException(status_code=503, detail="Inference engine not ready.")

        model_name = state.model_override or state.config.model_name
        if req.model and req.model != model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model '{req.model}' does not match active model '{model_name}'.",
            )

        prompt = messages_to_prompt(req.messages)
        sampling_params = SamplingParams(
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 1.0,
            max_tokens=req.max_tokens if req.max_tokens is not None else 256,
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        if req.stream:
            return StreamingResponse(
                stream_chat_response(state.engine, prompt, sampling_params, request_id, model_name),
                media_type="text/event-stream",
            )
        return await non_stream_chat_response(
            state.engine, prompt, sampling_params, request_id, model_name
        )

    return app


async def stream_chat_response(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model_name: str,
):
    created = int(time.time())
    previous_text = ""
    final_finish_reason = "stop"
    yielded_role = False

    async for output in engine.generate(prompt, sampling_params, request_id):
        if not output.outputs:
            continue
        text = output.outputs[0].text
        delta = text[len(previous_text) :]
        previous_text = text
        final_finish_reason = output.outputs[0].finish_reason or "stop"

        if not yielded_role:
            role_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"
            yielded_role = True

        if delta:
            content_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"

    end_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": final_finish_reason}],
    }
    yield f"data: {json.dumps(end_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def non_stream_chat_response(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model_name: str,
) -> dict:
    created = int(time.time())
    final_text = ""
    final_finish_reason = "stop"

    async for output in engine.generate(prompt, sampling_params, request_id):
        if not output.outputs:
            continue
        final_text = output.outputs[0].text
        final_finish_reason = output.outputs[0].finish_reason or "stop"

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": final_finish_reason,
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon OpenAI-compatible inference server")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_URL"),
        required=os.environ.get("COORDINATOR_URL") is None,
        help="Coordinator base URL, e.g. http://10.0.0.10:8000",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8080, help="Server bind port")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional explicit model name override (defaults to coordinator config).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.72,
        help="vLLM GPU memory utilization ratio (lower for 10-12GB testing).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="vLLM max model context length for memory-constrained testing.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="vLLM model dtype (e.g. float16, bfloat16, auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = ServerState(
        coordinator_url=args.coordinator_url.rstrip("/"),
        model_override=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
