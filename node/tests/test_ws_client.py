"""Unit tests for the coordinator WebSocket client (B-1)."""
from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

# Stub heavy optional deps so tests run without them installed.
if "httpx" not in sys.modules:
    stub = types.ModuleType("httpx")
    stub.AsyncClient = object
    sys.modules["httpx"] = stub

from coordinator.ws_client import _send_signal_when_ready, _ws_url, ws_session_loop
from runtime.state import NodeRuntimeState
from topology.models import BackendConfig, NodeAssignment, SliceSpec, StartupConfig, TopologyNode


def build_state(advertise_port: int = 9000, bind_port: int = 9000) -> NodeRuntimeState:
    return NodeRuntimeState(
        coordinator_url="http://127.0.0.1:8000",
        node_id="node-a",
        bind_host="0.0.0.0",
        bind_port=bind_port,
        advertise_host="1.2.3.4",
        advertise_port=advertise_port,
        vllm_worker_port=8100,
        launch_vllm_worker=False,
        vllm_gpu_memory_utilization=0.72,
        vllm_max_model_len=1024,
        vllm_dtype="float16",
    )


def build_startup_config() -> StartupConfig:
    return StartupConfig(
        cluster_id="cluster-ws",
        model_name="test-model",
        execution_mode="dry_run",
        pipeline_parallel_size=1,
        stage_count=1,
        entry_node_id="node-a",
        ray_head_address="",
        backend_config=BackendConfig(),
        nodes=[
            TopologyNode(
                node_id="node-a",
                host="127.0.0.1",
                port=9000,
                callback_url="http://127.0.0.1:9000",
                worker_url="http://127.0.0.1:8100",
                stage_index=0,
                stage_role="entry",
            )
        ],
        assignment=NodeAssignment(
            node_id="node-a",
            stage_index=0,
            stage_count=1,
            stage_role="entry",
            load_strategy="dry_run",
            slice_spec=SliceSpec(stage_index=0, stage_count=1, partition_label="0/1", executable=False),
            worker_endpoint="http://127.0.0.1:8100",
        ),
    )


class WSUrlTests(IsolatedAsyncioTestCase):
    def test_converts_http_to_ws(self) -> None:
        url = _ws_url("http://coordinator:8000", "node-a", "")
        assert url.startswith("ws://coordinator:8000/ws?node_id=node-a"), url

    def test_converts_https_to_wss(self) -> None:
        url = _ws_url("https://coordinator:8000", "node-a", "")
        assert url.startswith("wss://"), url

    def test_appends_token_when_set(self) -> None:
        url = _ws_url("http://coordinator:8000", "node-a", "secret")
        assert "token=secret" in url, url

    def test_omits_token_when_empty(self) -> None:
        url = _ws_url("http://coordinator:8000", "node-a", "")
        assert "token=" not in url, url


class SendSignalTests(IsolatedAsyncioTestCase):
    async def test_sends_signal_after_startup_event(self) -> None:
        state = build_state()
        state.startup_event.set()

        mock_ws = AsyncMock()
        await _send_signal_when_ready(state, mock_ws)

        mock_ws.send.assert_awaited_once()
        payload = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(payload["type"], "signal")
        self.assertEqual(payload["external_addr"], "1.2.3.4")
        self.assertEqual(payload["external_port"], 9000)

    async def test_signal_uses_hole_punch_when_ports_match(self) -> None:
        state = build_state(advertise_port=9000, bind_port=9000)
        state.startup_event.set()
        mock_ws = AsyncMock()
        await _send_signal_when_ready(state, mock_ws)
        payload = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(payload["transport_mode"], "hole_punch")

    async def test_signal_uses_port_forward_when_ports_differ(self) -> None:
        state = build_state(advertise_port=12345, bind_port=9000)
        state.startup_event.set()
        mock_ws = AsyncMock()
        await _send_signal_when_ready(state, mock_ws)
        payload = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(payload["transport_mode"], "port_forward")

    async def test_waits_for_startup_event_before_sending(self) -> None:
        state = build_state()
        mock_ws = AsyncMock()

        async def set_event_after_delay() -> None:
            await asyncio.sleep(0.05)
            state.startup_event.set()

        task = asyncio.create_task(_send_signal_when_ready(state, mock_ws))
        asyncio.create_task(set_event_after_delay())
        await task

        mock_ws.send.assert_awaited_once()


class RecvLoopTests(IsolatedAsyncioTestCase):
    async def test_sets_signal_ready_event_on_signal_ready_message(self) -> None:
        from coordinator.ws_client import _recv_loop

        state = build_state()

        signal_ready_msg = json.dumps(
            {
                "type": "signal_ready",
                "cluster_id": "cluster-ws",
                "peers": [
                    {
                        "node_id": "node-b",
                        "external_addr": "5.6.7.8",
                        "external_port": 13000,
                        "transport_mode": "hole_punch",
                    }
                ],
            }
        )

        async def fake_aiter(ws):
            yield signal_ready_msg

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: fake_aiter(self)

        await _recv_loop(state, mock_ws)

        self.assertTrue(state.signal_ready_event.is_set())
        self.assertIsNotNone(state.signal_ready)
        self.assertEqual(state.signal_ready.cluster_id, "cluster-ws")
        self.assertEqual(len(state.signal_ready.peers), 1)
        self.assertEqual(state.signal_ready.peers[0].node_id, "node-b")

    async def test_applies_startup_config_from_ws(self) -> None:
        from coordinator.ws_client import _recv_loop

        state = build_state()
        config = build_startup_config()
        msg = json.dumps({"type": "startup_config", **config.model_dump()})

        async def fake_aiter(ws):
            yield msg

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: fake_aiter(self)

        # Patch both report_node_status and handle_cluster_start so the background
        # task created by apply_startup_config completes without doing real work.
        with (
            patch("runtime.lifecycle.report_node_status", new=AsyncMock()),
            patch("runtime.lifecycle.handle_cluster_start", new=AsyncMock()),
        ):
            await _recv_loop(state, mock_ws)
            # Ensure any create_task'd coroutines run within the patched scope.
            await asyncio.sleep(0)

        self.assertIsNotNone(state.startup_config)
        self.assertEqual(state.startup_config.cluster_id, "cluster-ws")
        self.assertTrue(state.startup_event.is_set())

    async def test_skips_malformed_messages_silently(self) -> None:
        from coordinator.ws_client import _recv_loop

        state = build_state()

        async def fake_aiter(ws):
            yield "not json {"
            yield json.dumps({"type": "unknown_type"})

        mock_ws = MagicMock()
        mock_ws.__aiter__ = lambda self: fake_aiter(self)

        # Should complete without raising.
        await _recv_loop(state, mock_ws)

    async def test_ws_session_loop_exits_gracefully_when_websockets_absent(self) -> None:
        state = build_state()
        with patch.dict(sys.modules, {"websockets": None}):
            # Should return quickly rather than looping forever.
            await asyncio.wait_for(ws_session_loop(state), timeout=1.0)
