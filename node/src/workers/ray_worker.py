import asyncio
import logging
import os
import sys
from typing import Optional

LOGGER = logging.getLogger(__name__)


async def join_ray_cluster(ray_head_address: str) -> bool:
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
        return False
    LOGGER.info("Ray join completed successfully.")
    return True


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
    extra_args: Optional[list[str]] = None,
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
    if extra_args:
        cmd.extend(extra_args)
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
