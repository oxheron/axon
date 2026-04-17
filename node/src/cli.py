from __future__ import annotations

import argparse
import logging
import os
import socket
import uuid

import uvicorn

from api.app import create_app
from hardware import detect_local_ip
from runtime.state import NodeRuntimeState

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [node] %(message)s",
)


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
