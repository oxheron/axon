import argparse
import asyncio
import logging
import os
import re
import socket
import subprocess
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
    format="%(asctime)s %(levelname)s [node] %(message)s",
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
    worker_url: str


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
    vllm_launch_error: Optional[str] = None


def _detect_vram_gb_rocm_smi() -> float:
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "-d", "0"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0
    text = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"VRAM Total Memory \(MiB\):\s*([0-9]+)", text)
    if match:
        return float(match.group(1)) / 1024.0
    match = re.search(r"VRAM Total Memory \(B\):\s*([0-9]+)", text)
    if match:
        return float(match.group(1)) / (1024**3)
    return 0.0


def detect_vram_gb() -> float:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.total / (1024**3)
    except Exception:  # noqa: BLE001
        pass

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
    except Exception:  # noqa: BLE001
        pass

    rocm_gb = _detect_vram_gb_rocm_smi()
    if rocm_gb > 0.0:
        return rocm_gb
    return 0.0


def detect_torch_accelerator() -> str:
    try:
        import torch
    except ImportError:
        return "none"
    if getattr(torch.version, "hip", None):
        return "rocm"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "none"


def _preflight_vllm_for_accelerator(accel: str) -> None:
    if accel != "rocm":
        return
    try:
        import vllm._rocm_C  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "ROCm PyTorch is active, but this vLLM build does not load `vllm._rocm_C` (%s). "
            "Install a ROCm-matched vLLM build; the default PyPI CUDA wheel will not run on this host.",
            exc,
        )
        return
    try:
        import amdsmi  # noqa: F401, PLC0415
    except ImportError:
        LOGGER.warning(
            "ROCm detected but Python package `amdsmi` is not installed; "
            "vLLM may not auto-detect the ROCm platform."
        )


def build_vllm_worker_environ(accel: str) -> dict[str, str]:
    env = os.environ.copy()
    if accel == "rocm":
        env.setdefault("VLLM_TARGET_DEVICE", "rocm")
    elif accel == "cuda":
        env.setdefault("VLLM_TARGET_DEVICE", "cuda")
    return env


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
        LOGGER.warning("Ray join returned %s: %s", proc.returncode, combined)
    else:
        LOGGER.info("Ray join completed successfully.")


async def _pipe_subprocess_output(stream: asyncio.StreamReader, prefix: str) -> None:
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            LOGGER.info("%s %s", prefix, line.decode(errors="replace").rstrip())
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        LOGGER.exception("%s log reader failed", prefix)


async def start_vllm_worker_process(
    model_name: str,
    pipeline_parallel_size: int,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    distributed_backend: str = "ray",
    env: Optional[dict[str, str]] = None,
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
        distributed_backend,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
    ]
    LOGGER.info("Starting local vLLM worker API: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env if env is not None else os.environ,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout is not None:
        asyncio.create_task(_pipe_subprocess_output(proc.stdout, "[vllm]"))
    return proc


def create_app(state: NodeRuntimeState) -> FastAPI:
    app = FastAPI(title="Axon Node Service", version="0.1.0")

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
            "worker_url": f"http://{state.advertise_host}:{state.vllm_worker_port}",
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
            "worker_url": f"http://{state.advertise_host}:{state.vllm_worker_port}",
            "ray_joined": state.ray_joined,
            "startup_config": state.startup_config.model_dump()
            if state.startup_config
            else None,
            "vllm_pid": state.vllm_proc.pid if state.vllm_proc else None,
            "vllm_launch_error": state.vllm_launch_error,
        }

    return app


async def register_loop(state: NodeRuntimeState) -> None:
    payload = NodeRegistration(
        node_id=state.node_id,
        host=state.advertise_host,
        port=state.bind_port,
        vram_gb=state.vram_gb,
        callback_url=f"http://{state.advertise_host}:{state.bind_port}",
        worker_url=f"http://{state.advertise_host}:{state.vllm_worker_port}",
    )
    endpoint = f"{state.coordinator_url}/register"

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                response = await client.post(endpoint, json=payload.model_dump())
                if response.status_code == 409:
                    LOGGER.error("Registration rejected: cluster already started.")
                    return
                response.raise_for_status()
                LOGGER.info("Registered successfully with coordinator.")
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Registration failed (%s), retrying...", exc)
                await asyncio.sleep(2)


async def handle_cluster_start(state: NodeRuntimeState) -> None:
    await state.startup_event.wait()
    assert state.startup_config is not None
    config = state.startup_config
    LOGGER.info(
        "Received startup signal: model=%s pipeline_parallel_size=%s ray=%s",
        config.model_name,
        config.pipeline_parallel_size,
        config.ray_head_address,
    )

    if config.pipeline_parallel_size == 1:
        LOGGER.info("Single-node mode: skipping Ray cluster join, using mp backend.")
    else:
        await join_ray_cluster(config.ray_head_address)
        state.ray_joined = True

    if state.launch_vllm_worker:
        backend = "mp" if config.pipeline_parallel_size == 1 else "ray"
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
            state.vllm_launch_error = (
                f"torch_accelerator_probe_timeout_{int(probe_timeout)}s"
            )
            LOGGER.error(
                "Torch accelerator probe timed out after %.0fs. "
                "Fix the GPU stack, then retry.",
                probe_timeout,
            )
            return
        LOGGER.info("PyTorch accelerator for vLLM subprocess: %s", accel)
        await asyncio.to_thread(_preflight_vllm_for_accelerator, accel)
        vllm_env = build_vllm_worker_environ(accel)
        state.vllm_proc = await start_vllm_worker_process(
            model_name=config.model_name,
            pipeline_parallel_size=config.pipeline_parallel_size,
            port=state.vllm_worker_port,
            gpu_memory_utilization=state.vllm_gpu_memory_utilization,
            max_model_len=state.vllm_max_model_len,
            dtype=state.vllm_dtype,
            distributed_backend=backend,
            env=vllm_env,
        )
        LOGGER.info("vLLM worker started with pid=%s", state.vllm_proc.pid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon node service")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_URL"),
        required=os.environ.get("COORDINATOR_URL") is None,
        help="Coordinator base URL, e.g. http://10.0.0.10:8000",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Node service bind host")
    parser.add_argument("--port", type=int, default=9000, help="Node service bind port")
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
