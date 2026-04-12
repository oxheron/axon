import argparse
import asyncio
import logging
import os
import socket
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [node-agent] %(message)s",
)
LOGGER = logging.getLogger(__name__)


class StartupConfig(BaseModel):
    model_name: str
    pipeline_parallel_size: int = Field(..., ge=1)
    ray_head_address: str


class NodeRegistration(BaseModel):
    node_id: str
    host: str
    port: int
    vram_gb: float
    callback_url: str


@dataclass
class NodeRuntimeState:
    coordinator_url: str
    node_id: str
    bind_host: str
    bind_port: int
    advertise_host: str
    vllm_worker_port: int
    launch_vllm_worker: bool
    vllm_gpu_memory_utilization: float
    vllm_max_model_len: int
    vllm_dtype: str

    startup_config: Optional[StartupConfig] = None
    startup_event: asyncio.Event = field(default_factory=asyncio.Event)
    launch_task: Optional[asyncio.Task] = None
    vllm_proc: Optional[asyncio.subprocess.Process] = None
    ray_joined: bool = False
    vram_gb: float = 0.0


def detect_vram_gb() -> float:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.total / (1024**3)
    except Exception:  # noqa: BLE001 - fallback path below.
        pass

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
    except Exception:  # noqa: BLE001
        pass

    return 0.0


def detect_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


async def join_ray_cluster(ray_head_address: str) -> None:
    cmd = [
        "ray",
        "start",
        "--address",
        ray_head_address,
        "--disable-usage-stats",
        "--num-gpus",
        "1",
    ]
    LOGGER.info("Joining Ray cluster: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        combined = (stderr or stdout).decode().strip()
        # If a Ray runtime is already active this may fail but the node can still be usable.
        LOGGER.warning("Ray join returned %s: %s", proc.returncode, combined)
    else:
        LOGGER.info("Ray join completed successfully.")


async def start_vllm_worker_process(
    model_name: str,
    pipeline_parallel_size: int,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
) -> asyncio.subprocess.Process:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model",
        model_name,
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
        "--distributed-executor-backend",
        "ray",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
        "--disable-log-requests",
    ]
    LOGGER.info("Starting local vLLM worker API: %s", " ".join(cmd))
    return await asyncio.create_subprocess_exec(*cmd)


def create_app(state: NodeRuntimeState) -> FastAPI:
    app = FastAPI(title="Axon Node Agent", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        state.vram_gb = detect_vram_gb()
        LOGGER.info("Detected VRAM: %.2f GiB", state.vram_gb)
        asyncio.create_task(register_loop(state))

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "node_id": state.node_id,
            "registered_vram_gb": state.vram_gb,
            "ray_joined": state.ray_joined,
            "startup_received": state.startup_config is not None,
            "vllm_worker_running": state.vllm_proc is not None
            and state.vllm_proc.returncode is None,
        }

    @app.post("/startup")
    async def receive_startup(config: StartupConfig) -> dict:
        if state.startup_config is not None:
            return {"accepted": True, "duplicate": True}

        state.startup_config = config
        state.startup_event.set()
        state.launch_task = asyncio.create_task(handle_cluster_start(state))
        return {"accepted": True, "duplicate": False}

    @app.get("/status")
    async def status() -> dict:
        return {
            "node_id": state.node_id,
            "vram_gb": state.vram_gb,
            "ray_joined": state.ray_joined,
            "startup_config": state.startup_config.model_dump()
            if state.startup_config
            else None,
            "vllm_pid": state.vllm_proc.pid if state.vllm_proc else None,
        }

    return app


async def register_loop(state: NodeRuntimeState) -> None:
    payload = NodeRegistration(
        node_id=state.node_id,
        host=state.advertise_host,
        port=state.bind_port,
        vram_gb=state.vram_gb,
        callback_url=f"http://{state.advertise_host}:{state.bind_port}",
    )
    endpoint = f"{state.coordinator_url}/register"

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.post(endpoint, json=payload.model_dump())
                if resp.status_code == 409:
                    LOGGER.error("Registration rejected: cluster already started.")
                    return
                resp.raise_for_status()
                LOGGER.info("Registered successfully with coordinator.")
                return
            except Exception as exc:  # noqa: BLE001 - loop until coordinator is ready.
                LOGGER.warning("Registration failed (%s), retrying...", exc)
                await asyncio.sleep(2)


async def handle_cluster_start(state: NodeRuntimeState) -> None:
    await state.startup_event.wait()
    assert state.startup_config is not None
    cfg = state.startup_config
    LOGGER.info(
        "Received startup signal: model=%s pipeline_parallel_size=%s ray=%s",
        cfg.model_name,
        cfg.pipeline_parallel_size,
        cfg.ray_head_address,
    )

    await join_ray_cluster(cfg.ray_head_address)
    state.ray_joined = True

    if state.launch_vllm_worker:
        state.vllm_proc = await start_vllm_worker_process(
            model_name=cfg.model_name,
            pipeline_parallel_size=cfg.pipeline_parallel_size,
            port=state.vllm_worker_port,
            gpu_memory_utilization=state.vllm_gpu_memory_utilization,
            max_model_len=state.vllm_max_model_len,
            dtype=state.vllm_dtype,
        )
        LOGGER.info("vLLM worker started with pid=%s", state.vllm_proc.pid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon node agent")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_URL"),
        required=os.environ.get("COORDINATOR_URL") is None,
        help="Coordinator base URL, e.g. http://10.0.0.10:8000",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Node agent bind host")
    parser.add_argument("--port", type=int, default=9000, help="Node agent bind port")
    parser.add_argument(
        "--advertise-host",
        default=os.environ.get("ADVERTISE_HOST") or detect_local_ip(),
        help="Reachable host/IP advertised to coordinator callbacks",
    )
    parser.add_argument(
        "--node-id",
        default=f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
        help="Unique node ID for registration",
    )
    parser.add_argument(
        "--vllm-worker-port",
        type=int,
        default=8100,
        help="Local port for node vLLM worker API",
    )
    parser.add_argument(
        "--no-vllm-worker",
        action="store_true",
        help="Only join Ray cluster; skip starting local vLLM API process.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.72,
        help="GPU memory utilization ratio for local worker process.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=1024,
        help="Max model context length for local worker memory control.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default="float16",
        help="Model dtype for local worker (e.g. float16, bfloat16, auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = NodeRuntimeState(
        coordinator_url=args.coordinator_url.rstrip("/"),
        node_id=args.node_id,
        bind_host=args.host,
        bind_port=args.port,
        advertise_host=args.advertise_host,
        vllm_worker_port=args.vllm_worker_port,
        launch_vllm_worker=not args.no_vllm_worker,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=args.vllm_dtype,
    )
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
