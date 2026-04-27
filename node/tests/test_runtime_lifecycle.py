from __future__ import annotations

import asyncio
import sys
import types
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.AsyncClient = object
    sys.modules["httpx"] = httpx_stub

from runtime.lifecycle import handle_cluster_start
from runtime.state import NodeRuntimeState
from topology.assignment import build_default_assignment
from runtime.strategy import resolve_launch_strategy
from topology.models import BackendConfig, NodeAssignment, SliceSpec, StartupConfig, TopologyNode


def build_state() -> NodeRuntimeState:
    return NodeRuntimeState(
        coordinator_url="http://127.0.0.1:8000",
        node_id="node-a",
        bind_host="0.0.0.0",
        bind_port=9000,
        advertise_host="127.0.0.1",
        advertise_port=9000,
        vllm_worker_port=8100,
        launch_vllm_worker=True,
        vllm_gpu_memory_utilization=0.72,
        vllm_max_model_len=1024,
        vllm_dtype="float16",
    )


def build_dry_run_config() -> StartupConfig:
    assignment = NodeAssignment(
        node_id="node-a",
        stage_index=0,
        stage_count=2,
        stage_role="entry",
        load_strategy="dry_run",
        slice_spec=SliceSpec(
            stage_index=0,
            stage_count=2,
            partition_label="0/2",
            executable=False,
        ),
        worker_endpoint="http://127.0.0.1:8100",
    )
    return StartupConfig(
        cluster_id="cluster-test",
        model_name="test-model",
        execution_mode="dry_run",
        pipeline_parallel_size=2,
        stage_count=2,
        entry_node_id="node-a",
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
            ),
            TopologyNode(
                node_id="node-b",
                host="127.0.0.1",
                port=9001,
                callback_url="http://127.0.0.1:9001",
                worker_url="http://127.0.0.1:8101",
                stage_index=1,
                stage_role="final",
            ),
        ],
        assignment=assignment,
    )


def build_multi_node_config() -> StartupConfig:
    assignment = NodeAssignment(
        node_id="node-a",
        stage_index=0,
        stage_count=2,
        stage_role="entry",
        load_strategy="vllm_slice",
        slice_spec=SliceSpec(
            stage_index=0,
            stage_count=2,
            partition_label="0/2",
            executable=True,
        ),
        worker_endpoint="http://127.0.0.1:8100",
    )
    return StartupConfig(
        cluster_id="cluster-multi",
        model_name="test-model",
        execution_mode="vllm_slice",
        pipeline_parallel_size=2,
        stage_count=2,
        entry_node_id="node-a",
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
            ),
            TopologyNode(
                node_id="node-b",
                host="127.0.0.1",
                port=9001,
                callback_url="http://127.0.0.1:9001",
                worker_url="http://127.0.0.1:8101",
                stage_index=1,
                stage_role="final",
            ),
        ],
        assignment=assignment,
    )


class HandleClusterStartTests(IsolatedAsyncioTestCase):
    async def test_dry_run_stops_before_backend_startup(self) -> None:
        state = build_state()
        state.startup_config = build_dry_run_config()
        state.startup_event.set()

        with (
            patch("runtime.lifecycle.report_node_status", new=AsyncMock()) as report_status,
            patch(
                "runtime.lifecycle.start_vllm_worker_process", new=AsyncMock()
            ) as start_vllm_worker_process,
        ):
            await handle_cluster_start(state)

        start_vllm_worker_process.assert_not_awaited()
        self.assertEqual(
            [call.args[1] for call in report_status.await_args_list],
            ["load_started", "slice_loaded", "dry_run_ready"],
        )

    async def test_multi_node_reports_signaling_then_gates_on_signal_ready(self) -> None:
        """handle_cluster_start must emit 'signaling' and wait for signal_ready_event
        before reporting 'load_started' in a multi-node non-dry-run cluster."""
        state = build_state()
        state.launch_vllm_worker = False  # skip torch/GPU probe, only test signaling gate
        state.startup_config = build_multi_node_config()
        state.startup_event.set()

        async def release_signal_ready_after_short_delay() -> None:
            await asyncio.sleep(0.05)
            state.signal_ready_event.set()

        with (
            patch("runtime.lifecycle.report_node_status", new=AsyncMock()) as report_status,
        ):
            asyncio.create_task(release_signal_ready_after_short_delay())
            await handle_cluster_start(state)

        reported_states = [call.args[1] for call in report_status.await_args_list]
        self.assertIn("signaling", reported_states)
        self.assertIn("load_started", reported_states)
        self.assertLess(
            reported_states.index("signaling"),
            reported_states.index("load_started"),
            "signaling must be reported before load_started",
        )


class AssignmentResolutionTests(TestCase):
    def test_build_default_assignment_marks_dry_run_as_non_executable(self) -> None:
        state = build_state()
        config = StartupConfig(
            cluster_id="cluster-test",
            model_name="test-model",
            execution_mode="dry_run",
            pipeline_parallel_size=2,
            stage_count=2,
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
        )

        assignment = build_default_assignment(config, state)

        self.assertEqual(assignment.load_strategy, "dry_run")
        self.assertFalse(assignment.slice_spec.executable)

    def test_resolve_launch_strategy_reports_dry_run_terminal_state(self) -> None:
        state = build_state()
        strategy = resolve_launch_strategy(build_dry_run_config(), state)

        self.assertEqual(strategy.execution_mode, "dry_run")
        self.assertEqual(strategy.load_strategy, "dry_run")
        self.assertFalse(strategy.launches_worker)
        self.assertEqual(strategy.final_lifecycle_state, "dry_run_ready")
