from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_NODE_SRC = str(Path(__file__).parent.parent)

LOGGER = logging.getLogger(__name__)


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
    distributed_backend: str = "mp",
    env: Optional[dict[str, str]] = None,
    extra_args: Optional[list[str]] = None,
    transport_env: Optional[dict[str, str]] = None,
) -> asyncio.subprocess.Process:
    cmd = [
        sys.executable,
        "-m",
        "workers.vllm_launcher",
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
    merged_env = dict(env if env is not None else os.environ)
    existing_path = merged_env.get("PYTHONPATH", "")
    merged_env["PYTHONPATH"] = f"{_NODE_SRC}:{existing_path}" if existing_path else _NODE_SRC
    if transport_env:
        merged_env.update(transport_env)
    LOGGER.info("Starting local vLLM worker API: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=merged_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout is not None:
        asyncio.create_task(_pipe_subprocess_output(proc.stdout, "[vllm]"))
    return proc
