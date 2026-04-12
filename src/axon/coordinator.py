import argparse
import asyncio
import logging
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [coordinator] %(message)s",
)
LOGGER = logging.getLogger(__name__)


class NodeRegistration(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    host: str = Field(..., description="Node host/IP reachable by coordinator")
    port: int = Field(..., ge=1, le=65535, description="Node agent HTTP port")
    vram_gb: float = Field(..., ge=0, description="Detected VRAM in GiB")
    callback_url: Optional[str] = Field(
        default=None, description="Optional callback endpoint for startup signal"
    )


class StartupConfig(BaseModel):
    model_name: str
    pipeline_parallel_size: int
    ray_head_address: str


class RegistrationResponse(BaseModel):
    accepted: bool
    registered_nodes: int
    startup_triggered: bool


@dataclass
class NodeInfo:
    node_id: str
    host: str
    port: int
    vram_gb: float
    callback_url: str


@dataclass
class CoordinatorState:
    min_nodes: int
    model_name: str
    ray_head_address: str
    nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    startup_config: Optional[StartupConfig] = None
    broadcast_task: Optional[asyncio.Task] = None


def build_callback_url(registration: NodeRegistration) -> str:
    if registration.callback_url:
        return registration.callback_url.rstrip("/")
    return f"http://{registration.host}:{registration.port}"


def detect_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def maybe_start_ray_head(enabled: bool, port: int) -> None:
    if not enabled:
        return

    cmd = [
        "ray",
        "start",
        "--head",
        "--port",
        str(port),
        "--disable-usage-stats",
    ]
    LOGGER.info("Starting Ray head: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        LOGGER.warning(
            "Ray head start exited with code %s (this can be expected if already running): %s",
            completed.returncode,
            completed.stderr.strip() or completed.stdout.strip(),
        )
    else:
        LOGGER.info("Ray head started successfully.")


def create_app(state: CoordinatorState) -> FastAPI:
    app = FastAPI(title="Axon Coordinator", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/status")
    async def status() -> dict:
        return {
            "min_nodes": state.min_nodes,
            "registered_nodes": len(state.nodes),
            "pipeline_ready": state.startup_config is not None,
            "model_name": state.model_name,
            "ray_head_address": state.ray_head_address,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "vram_gb": n.vram_gb,
                    "callback_url": n.callback_url,
                }
                for n in state.nodes.values()
            ],
        }

    @app.get("/config", response_model=StartupConfig)
    async def config() -> StartupConfig:
        if state.startup_config is None:
            raise HTTPException(status_code=404, detail="Pipeline has not started yet.")
        return state.startup_config

    @app.post("/register", response_model=RegistrationResponse)
    async def register_node(registration: NodeRegistration) -> RegistrationResponse:
        if state.startup_config is not None:
            raise HTTPException(
                status_code=409,
                detail="Cluster startup already triggered; dynamic resize is disabled.",
            )

        callback_url = build_callback_url(registration)
        state.nodes[registration.node_id] = NodeInfo(
            node_id=registration.node_id,
            host=registration.host,
            port=registration.port,
            vram_gb=registration.vram_gb,
            callback_url=callback_url,
        )
        LOGGER.info(
            "Registered node=%s host=%s:%s vram=%.2fGiB (%s/%s)",
            registration.node_id,
            registration.host,
            registration.port,
            registration.vram_gb,
            len(state.nodes),
            state.min_nodes,
        )

        startup_triggered = False
        if len(state.nodes) >= state.min_nodes and state.startup_config is None:
            state.startup_config = StartupConfig(
                model_name=state.model_name,
                pipeline_parallel_size=len(state.nodes),
                ray_head_address=state.ray_head_address,
            )
            state.broadcast_task = asyncio.create_task(broadcast_startup(state))
            startup_triggered = True
            LOGGER.info(
                "Startup triggered with pipeline_parallel_size=%s",
                state.startup_config.pipeline_parallel_size,
            )

        return RegistrationResponse(
            accepted=True,
            registered_nodes=len(state.nodes),
            startup_triggered=startup_triggered,
        )

    return app


async def send_startup_signal(
    client: httpx.AsyncClient, node: NodeInfo, config: StartupConfig
) -> bool:
    url = f"{node.callback_url}/startup"
    payload = config.model_dump()
    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        LOGGER.info("Startup signal delivered to node=%s (%s)", node.node_id, url)
        return True
    except Exception as exc:  # noqa: BLE001 - keep broad to continue broadcasting.
        LOGGER.error("Failed startup signal for node=%s (%s): %s", node.node_id, url, exc)
        return False


async def broadcast_startup(state: CoordinatorState) -> None:
    if state.startup_config is None:
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        for node in state.nodes.values():
            # Retry a few times because node agent may still be settling after registration.
            for attempt in range(1, 6):
                success = await send_startup_signal(client, node, state.startup_config)
                if success:
                    break
                await asyncio.sleep(1.5 * attempt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon coordinator service")
    parser.add_argument("--host", default="0.0.0.0", help="Coordinator bind host")
    parser.add_argument("--port", type=int, default=8000, help="Coordinator bind port")
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=2,
        help="Minimum nodes required before startup signal",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name/path used by all nodes and inference server",
    )
    parser.add_argument(
        "--ray-port", type=int, default=6379, help="Ray GCS port for head node"
    )
    parser.add_argument(
        "--ray-head-address",
        default=None,
        help="Ray head address host:port. If omitted, inferred from local IP + --ray-port",
    )
    parser.add_argument(
        "--autostart-ray-head",
        action="store_true",
        help="Start `ray start --head` before serving HTTP API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_nodes < 1:
        raise ValueError("--min-nodes must be >= 1")

    maybe_start_ray_head(enabled=args.autostart_ray_head, port=args.ray_port)
    ray_head_address = args.ray_head_address or f"{detect_local_ip()}:{args.ray_port}"

    state = CoordinatorState(
        min_nodes=args.min_nodes,
        model_name=args.model_name,
        ray_head_address=ray_head_address,
    )
    app = create_app(state)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
