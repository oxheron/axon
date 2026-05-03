from __future__ import annotations

import os

from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.executor.uniproc_executor import UniProcExecutor


class AxonExecutor(UniProcExecutor):
    """Single-worker executor for axon pipeline-parallel stages.

    Each axon node runs one instance of this executor as one PP rank.
    torch.distributed is initialized here (before super()) via AxonCoordinatorStore
    so that vLLM's init_distributed_environment() finds it already initialized
    and skips the env:// rendezvous that would require MASTER_ADDR/MASTER_PORT.
    """

    supports_pp: bool = True

    def _init_executor(self) -> None:
        from axon_quic import _maybe_patch_vllm, _register_backend
        from axon_quic.coordinator_store import AxonCoordinatorStore

        import torch.distributed as dist

        _register_backend()
        _maybe_patch_vllm()

        if not dist.is_initialized():
            store = AxonCoordinatorStore(
                coordinator_url=os.environ["AXON_COORDINATOR_URL"],
                cluster_id=os.environ["AXON_CLUSTER_ID"],
            )
            dist.init_process_group(
                backend="axon_quic",
                store=store,
                rank=int(os.environ["AXON_PP_RANK"]),
                world_size=int(os.environ["AXON_PP_SIZE"]),
            )

        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        init_method = get_distributed_init_method(get_ip(), get_open_port())
        return init_method, int(os.environ["AXON_PP_RANK"]), 0
