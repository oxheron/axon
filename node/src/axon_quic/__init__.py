"""axon_quic — custom torch.distributed backend for PP tensor routing over QUIC.

Entry point called by vLLM's general plugin loader:
  vllm.general_plugins.axon_quic = axon_quic:_load_plugin

_load_plugin() runs in every vLLM worker process at startup, before
initialize_model_parallel(). It:
  1. Registers the "axon_quic" torch.distributed backend.
  2. (When AXON_PP_BACKEND=axon_quic) monkey-patches GroupCoordinator.__init__
     so that PP-sized groups use "axon_quic" as their torch_distributed_backend.
  3. (When AXON_PP_BACKEND=axon_quic and not already initialized) calls
     torch.distributed.init_process_group() using AxonCoordinatorStore for
     cross-node rendezvous via the coordinator HTTP endpoint.
"""
from __future__ import annotations

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)


def _register_backend() -> None:
    """Register the "axon_quic" torch.distributed backend."""
    try:
        import torch.distributed as dist
        from axon_quic.process_group import AxonQuicProcessGroup

        if hasattr(dist, "Backend") and hasattr(dist.Backend, "register_backend"):
            dist.Backend.register_backend(
                "axon_quic",
                lambda store, rank, size, timeout=None: AxonQuicProcessGroup(
                    store, rank, size, timeout
                ),
                devices=["cpu", "cuda"],
            )
            LOGGER.debug("[axon_quic] backend registered")
        else:
            LOGGER.warning("[axon_quic] torch.distributed.Backend.register_backend unavailable")
    except ImportError:
        LOGGER.debug("[axon_quic] torch not available, skipping backend registration")


def _maybe_patch_vllm() -> None:
    """Monkey-patch GroupCoordinator to use axon_quic for PP-sized groups."""
    if os.environ.get("AXON_PP_BACKEND") != "axon_quic":
        return

    pp_size = int(os.environ.get("AXON_PP_SIZE", "1"))
    if pp_size <= 1:
        return

    try:
        from vllm.distributed import parallel_state

        orig_init = parallel_state.GroupCoordinator.__init__

        def _patched_init(
            self: Any,
            group_ranks: list[list[int]],
            local_rank: int,
            torch_distributed_backend: str,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            # Identify PP groups by the number of ranks in each sub-group.
            if group_ranks and len(group_ranks[0]) == pp_size:
                LOGGER.info(
                    "[axon_quic] overriding torch_distributed_backend to 'axon_quic' "
                    "for %d-rank group (PP group)", pp_size
                )
                torch_distributed_backend = "axon_quic"
            orig_init(self, group_ranks, local_rank, torch_distributed_backend, *args, **kwargs)

        parallel_state.GroupCoordinator.__init__ = _patched_init
        LOGGER.info("[axon_quic] GroupCoordinator patched for pp_size=%d", pp_size)
    except ImportError:
        LOGGER.warning("[axon_quic] vllm.distributed not found; GroupCoordinator patch skipped")
    except Exception:
        LOGGER.exception("[axon_quic] GroupCoordinator patch failed")


def _maybe_init_process_group() -> None:
    """Initialize torch.distributed via AxonCoordinatorStore if not already done."""
    if os.environ.get("AXON_PP_BACKEND") != "axon_quic":
        return

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            LOGGER.debug("[axon_quic] torch.distributed already initialized, skipping")
            return

        coordinator_url = os.environ.get("AXON_COORDINATOR_URL", "")
        cluster_id = os.environ.get("AXON_CLUSTER_ID", "")
        pp_rank = int(os.environ.get("AXON_PP_RANK", "0"))
        pp_size = int(os.environ.get("AXON_PP_SIZE", "2"))

        if not coordinator_url or not cluster_id:
            LOGGER.warning(
                "[axon_quic] AXON_COORDINATOR_URL or AXON_CLUSTER_ID not set; "
                "skipping init_process_group"
            )
            return

        from axon_quic.coordinator_store import AxonCoordinatorStore
        from axon_quic.gloo_socket_env import apply_default_gloo_socket_ifname

        apply_default_gloo_socket_ifname(pp_size=pp_size)

        LOGGER.info(
            "[axon_quic] initializing torch.distributed rank=%d world=%d via coordinator %s",
            pp_rank,
            pp_size,
            coordinator_url,
        )
        store = AxonCoordinatorStore(
            coordinator_url=coordinator_url,
            cluster_id=cluster_id,
        )
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=pp_rank,
            world_size=pp_size,
        )
        LOGGER.info("[axon_quic] torch.distributed initialized (gloo, rank=%d/%d)", pp_rank, pp_size)
    except ImportError:
        LOGGER.debug("[axon_quic] torch not available, skipping init_process_group")
    except Exception:
        LOGGER.exception("[axon_quic] init_process_group failed")


def _load_plugin() -> None:
    """Called by vLLM's general plugin loader (vllm.general_plugins entry point)."""
    _register_backend()
    _maybe_patch_vllm()
    _maybe_init_process_group()
