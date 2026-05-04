"""AxonQuicProcessGroup — custom torch.distributed ProcessGroup for PP tensors.

Registered as the "axon_quic" backend. Handles the point-to-point operations
that vLLM's PP path uses:
  - isend / irecv  (async, returns AxonWork)
  - send / recv    (synchronous, returns AxonWork)
  - barrier        (coordinator-store-based, returns c10d Work)
  - broadcast_object_list  (for GroupCoordinator CPU metadata exchange)

All tensor data travels over the UDS sidecar → QUIC path established by B-2.

Inherits from torch.distributed.ProcessGroup (c10d::ProcessGroup) so that
PyTorch's _new_process_group_helper takes the PythonProcessGroup shortcut
(issubclass check) and never calls _register_backend, which requires c10d::Backend.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from torch._C._distributed_c10d import _create_work_from_future

from axon_quic.work import AxonWork
from axon_quic.wire import (
    tensor_to_bytes,
    bytes_to_tensor_bf16,
    encode_object_frame,
    decode_object_frame,
)
from axon_quic.sidecar_client import SidecarClient

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)

_SIDECAR_TIMEOUT = 5.0  # seconds — ConnectionError if no response


def _get_rank() -> int:
    return int(os.environ.get("AXON_PP_RANK", "0"))


def _get_size() -> int:
    return int(os.environ.get("AXON_PP_SIZE", "2"))


def _get_wire_dtype() -> str:
    return os.environ.get("AXON_WIRE_DTYPE", "fp8")


def _completed_work() -> Any:
    """Return a pre-completed c10d Work object (required by C++ dispatch path)."""
    from torch.futures import Future
    fut: Future = Future()
    fut.set_result(None)
    return _create_work_from_future(fut)


class AxonQuicProcessGroup(dist.ProcessGroup):
    """
    Custom torch.distributed ProcessGroup that routes PP tensors over QUIC.

    Inheriting from dist.ProcessGroup lets PyTorch use this instance directly
    as the process group without going through _register_backend (which requires
    a c10d::Backend subclass that we cannot satisfy from pure Python).

    The constructor signature (store, rank, size, timeout) is the standard
    factory signature used by torch.distributed.Backend.register_backend.
    """

    def __init__(self, store: Any, rank: int, size: int, timeout: Any = None) -> None:
        pg_rank = _get_rank()
        pg_size = _get_size()
        # c10d::ProcessGroup(rank, world_size) — store is managed separately
        super().__init__(pg_rank, pg_size)

        self._rank = pg_rank
        self._size = pg_size
        self._wire_dtype = _get_wire_dtype()
        self._store = store  # coordinator-backed store, used for barrier
        self._barrier_count = 0

        uds_path = os.environ.get("AXON_TRANSPORT_UDS", "")
        if not uds_path:
            raise RuntimeError(
                "AxonQuicProcessGroup: AXON_TRANSPORT_UDS not set. "
                "The node process must start the sidecar before launching vLLM."
            )
        self._client = SidecarClient(uds_path, timeout=_SIDECAR_TIMEOUT)
        LOGGER.info(
            "[axon_quic] ProcessGroup created: rank=%d size=%d uds=%s wire=%s",
            self._rank,
            self._size,
            uds_path,
            self._wire_dtype,
        )

    # ── Point-to-point tensor operations ─────────────────────────────────

    def isend(self, tensors: list["torch.Tensor"], dst: int, tag: int) -> AxonWork:
        try:
            frame = tensor_to_bytes(tensors[0], tag, wire_dtype=self._wire_dtype)
            self._client.send(peer_idx=dst, stream_id=tag, frame=frame)
            return AxonWork()
        except Exception as exc:
            LOGGER.error("[axon_quic] isend failed: %s", exc)
            err = exc if isinstance(exc, ConnectionError) else ConnectionError(f"axon_quic isend: {exc}")
            return AxonWork(exc=err)

    def irecv(self, tensors: list["torch.Tensor"], src: int, tag: int) -> AxonWork:
        try:
            frame = self._client.recv(peer_idx=src, stream_id=tag)
            received = bytes_to_tensor_bf16(frame, target_device=tensors[0].device)
            tensors[0].copy_(received)
            return AxonWork()
        except Exception as exc:
            LOGGER.error("[axon_quic] irecv failed: %s", exc)
            err = exc if isinstance(exc, ConnectionError) else ConnectionError(f"axon_quic irecv: {exc}")
            return AxonWork(exc=err)

    def send(self, tensors: list["torch.Tensor"], dst: int, tag: int) -> AxonWork:
        work = self.isend(tensors, dst, tag)
        work.wait()
        return work

    def recv(self, tensors: list["torch.Tensor"], src: int, tag: int) -> AxonWork:
        work = self.irecv(tensors, src, tag)
        work.wait()
        return work

    # ── Object send/recv (for GroupCoordinator CPU metadata group) ────────

    def broadcast_object_list(
        self, obj_list: list, src: int, group: Any = None
    ) -> None:
        """Broadcast a list of objects from rank `src` to all ranks in the group."""
        if self._rank == src:
            frame = encode_object_frame(obj_list, stream_id=0)
            for r in range(self._size):
                if r != src:
                    try:
                        self._client.send_obj(peer_idx=r, stream_id=0, frame=frame)
                    except Exception as exc:
                        raise ConnectionError(f"axon_quic broadcast_object_list send: {exc}") from exc
        else:
            try:
                frame = self._client.recv_obj(peer_idx=src, stream_id=0)
                result = decode_object_frame(frame)
                obj_list[:] = result
            except Exception as exc:
                raise ConnectionError(f"axon_quic broadcast_object_list recv: {exc}") from exc

    def send_object(self, obj: Any, dst: int, stream_id: int = 0) -> None:
        frame = encode_object_frame(obj, stream_id=stream_id)
        self._client.send_obj(peer_idx=dst, stream_id=stream_id, frame=frame)

    def recv_object(self, src: int, stream_id: int = 0) -> Any:
        frame = self._client.recv_obj(peer_idx=src, stream_id=stream_id)
        return decode_object_frame(frame)

    # ── Barrier — coordinator-store-based, returns c10d Work ─────────────

    def barrier(self, opts: Any = None) -> Any:
        arrive_key = f"axon_barrier_{self._barrier_count}_arrive"
        done_key = f"axon_barrier_{self._barrier_count}_done"
        self._barrier_count += 1
        n = self._store.add(arrive_key, 1)
        if n >= self._size:
            self._store.set(done_key, b"1")
        else:
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                try:
                    self._store.wait([done_key], 2.0)
                    break
                except TimeoutError:
                    continue
            else:
                raise TimeoutError(f"axon_quic barrier timed out at {arrive_key}")
        return _completed_work()

    # ── Collective stubs ──────────────────────────────────────────────────

    def allreduce(self, tensors: Any, opts: Any = None) -> Any:
        raise NotImplementedError("axon_quic: allreduce not supported")

    def broadcast(self, tensors: Any, opts: Any = None) -> Any:
        raise NotImplementedError("axon_quic: broadcast not supported")

    def allgather(self, output: Any, input: Any, opts: Any = None) -> Any:
        raise NotImplementedError("axon_quic: allgather not supported")

    def reduce_scatter(self, output: Any, input: Any, opts: Any = None) -> Any:
        raise NotImplementedError("axon_quic: reduce_scatter not supported")

    # ── ProcessGroup protocol ─────────────────────────────────────────────

    def getBackendName(self) -> str:
        return "axon_quic"

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"AxonQuicProcessGroup(rank={self._rank}, size={self._size})"
