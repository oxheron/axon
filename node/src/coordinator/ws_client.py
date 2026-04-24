"""WebSocket client for the coordinator signaling channel (B-1)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.state import NodeRuntimeState

LOGGER = logging.getLogger(__name__)

_RECONNECT_DELAY = 5.0


def _ws_url(coordinator_url: str, node_id: str, token: str) -> str:
    base = coordinator_url.rstrip("/")
    base = base.replace("https://", "wss://").replace("http://", "ws://")
    params = f"node_id={node_id}"
    if token:
        params += f"&token={token}"
    return f"{base}/ws?{params}"


async def ws_session_loop(state: NodeRuntimeState) -> None:
    """
    Background task: maintains a persistent WebSocket session with the coordinator.

    Receives startup_config over the WS (if the coordinator delivers it that way),
    sends the local node's signal, and waits for signal_ready.  Reconnects
    automatically on any error.
    """
    try:
        import websockets  # noqa: PLC0415
        import websockets.exceptions  # noqa: PLC0415
    except ImportError:
        LOGGER.error("[ws] websockets library not installed; WS signaling unavailable")
        return

    url = _ws_url(state.coordinator_url, state.node_id, state.cluster_token)
    LOGGER.info("[ws] will connect to %s", url)

    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=None,   # coordinator drives keepalive pings
                open_timeout=10,
            ) as ws:
                LOGGER.info("[ws] connected to coordinator")
                await _run_ws_session(state, ws)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[ws] session error (reconnecting in %.0fs): %s", _RECONNECT_DELAY, exc)
        await asyncio.sleep(_RECONNECT_DELAY)


async def _run_ws_session(state: NodeRuntimeState, ws: object) -> None:
    """Handle one live WS session: concurrent receive loop + signal sender."""
    send_task = asyncio.create_task(_send_signal_when_ready(state, ws))
    try:
        await _recv_loop(state, ws)
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass


async def _recv_loop(state: NodeRuntimeState, ws: object) -> None:
    """Process inbound coordinator messages until the connection closes."""
    from coordinator.models import NodeSignalData, SignalReadyData
    from topology.models import StartupConfig

    async for raw in ws:  # type: ignore[attr-defined]
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = msg.get("type")

        if msg_type == "startup_config":
            try:
                config = StartupConfig.model_validate(msg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[ws] invalid startup_config: %s", exc)
                continue
            await _apply_startup_config_from_ws(state, config)

        elif msg_type == "signal_ready":
            peers = [
                NodeSignalData(
                    node_id=p["node_id"],
                    external_addr=p["external_addr"],
                    external_port=p["external_port"],
                    transport_mode=p["transport_mode"],
                )
                for p in msg.get("peers", [])
            ]
            state.signal_ready = SignalReadyData(
                cluster_id=msg.get("cluster_id", ""),
                peers=peers,
            )
            state.signal_ready_event.set()
            LOGGER.info(
                "[ws] signal_ready received: cluster=%s peers=%d",
                state.signal_ready.cluster_id,
                len(peers),
            )

        elif msg_type == "error":
            LOGGER.error(
                "[ws] coordinator error: code=%s message=%s",
                msg.get("code"),
                msg.get("message"),
            )


async def _send_signal_when_ready(state: NodeRuntimeState, ws: object) -> None:
    """Wait for startup_config to arrive then send this node's signal."""
    await state.startup_event.wait()

    transport_mode = (
        "port_forward" if state.advertise_port != state.bind_port else "hole_punch"
    )
    signal = json.dumps(
        {
            "type": "signal",
            "external_addr": state.advertise_host,
            "external_port": state.advertise_port,
            "transport_mode": transport_mode,
        }
    )
    await ws.send(signal)  # type: ignore[attr-defined]
    LOGGER.info(
        "[ws] signal sent: addr=%s:%d mode=%s",
        state.advertise_host,
        state.advertise_port,
        transport_mode,
    )


async def _apply_startup_config_from_ws(state: NodeRuntimeState, config) -> None:
    """Idempotent: apply startup_config received over WS (mirrors HTTP /startup)."""
    if state.startup_config is not None:
        return
    # Delegate to the shared apply helper in runtime.lifecycle to avoid duplication.
    from runtime.lifecycle import apply_startup_config  # noqa: PLC0415
    await apply_startup_config(state, config)
