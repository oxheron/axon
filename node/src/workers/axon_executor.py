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

        pp_size = int(os.environ.get("AXON_PP_SIZE", "1"))
        if pp_size > 1:
            # vLLM defaults nnodes=1, making local_world_size = world_size (pp*tp)
            # which requires all GPUs to be local. Each axon node runs exactly one
            # PP stage, so nnodes=pp_size gives local_world_size = world_size/pp_size
            # = tp_size, which is the correct number of GPUs per node.
            # This scales: PP=4/TP=2 → world=8, nnodes=4, local_world_size=2 ✓
            self.vllm_config.parallel_config.nnodes = pp_size

        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        init_method = get_distributed_init_method(get_ip(), get_open_port())
        return init_method, int(os.environ["AXON_PP_RANK"]), 0
