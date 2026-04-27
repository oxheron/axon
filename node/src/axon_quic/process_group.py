"""AxonQuicProcessGroup — custom torch.distributed ProcessGroup for PP tensors.

Registered as the "axon_quic" backend. Handles only the point-to-point
operations that vLLM's PP path uses:
  - isend / irecv  (async, but returns a pre-completed AxonWork for B-3)
  - send / recv    (synchronous wrappers)
  - broadcast_object_list  (for GroupCoordinator CPU metadata exchange)

All data travels over the UDS sidecar → QUIC path established by B-2.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

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


class AxonQuicProcessGroup:
    """
    Custom torch.distributed ProcessGroup that routes PP tensors over QUIC.

    Created by PyTorch's new_group() machinery when backend="axon_quic" is
    requested. The constructor signature (store, rank, size) is standard for
    registered Python backends.
    """

    def __init__(self, store: Any, rank: int, size: int, timeout: Any = None) -> None:
        # Store rank/size for reference; actual rank/size come from env vars set
        # by the node process to match the PP assignment.
        self._rank = _get_rank()
        self._size = _get_size()
        self._wire_dtype = _get_wire_dtype()

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
            return AxonWork(exc=ConnectionError(f"axon_quic isend: {exc}") if not isinstance(exc, ConnectionError) else exc)

    def irecv(self, tensors: list["torch.Tensor"], src: int, tag: int) -> AxonWork:
        try:
            frame = self._client.recv(peer_idx=src, stream_id=tag)
            received = bytes_to_tensor_bf16(frame, target_device=tensors[0].device)
            tensors[0].copy_(received)
            return AxonWork()
        except Exception as exc:
            LOGGER.error("[axon_quic] irecv failed: %s", exc)
            return AxonWork(exc=ConnectionError(f"axon_quic irecv: {exc}") if not isinstance(exc, ConnectionError) else exc)

    def send(self, tensors: list["torch.Tensor"], dst: int, tag: int) -> None:
        self.isend(tensors, dst, tag).wait()

    def recv(self, tensors: list["torch.Tensor"], src: int, tag: int) -> None:
        self.irecv(tensors, src, tag).wait()

    # ── Object send/recv (for GroupCoordinator CPU metadata group) ────────

    def broadcast_object_list(
        self, obj_list: list, src: int, group: Any = None
    ) -> None:
        """Broadcast a list of objects from rank `src` to all ranks in the group."""
        if self._rank == src:
            # Encode and send to all other ranks
            frame = encode_object_frame(obj_list, stream_id=0)
            for rank in range(self._size):
                if rank != src:
                    try:
                        self._client.send_obj(peer_idx=rank, stream_id=0, frame=frame)
                    except Exception as exc:
                        raise ConnectionError(f"axon_quic broadcast_object_list send: {exc}") from exc
        else:
            # Receive from src
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

    # ── Collective stubs — raise if called on PP group ────────────────────

    def allreduce(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("axon_quic: allreduce not supported on PP group")

    def broadcast(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("axon_quic: broadcast not supported on PP group")

    def allgather(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("axon_quic: allgather not supported on PP group")

    def reduce_scatter(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("axon_quic: reduce_scatter not supported on PP group")

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("axon_quic: barrier not supported on PP group")

    # ── ProcessGroup protocol attributes ─────────────────────────────────

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"AxonQuicProcessGroup(rank={self._rank}, size={self._size})"
