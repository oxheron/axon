"""TransportSidecar — asyncio UDS server bridging the vLLM subprocess to QUIC.

Runs inside the node process event loop. The vLLM subprocess (AxonQuicProcessGroup
via SidecarClient) connects to this server and issues SEND/RECV requests. The
sidecar dispatches to the appropriate QuicConn.

Protocol: see sidecar_client.py for the binary framing spec.
"""
from __future__ import annotations

import asyncio
import logging
import os
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transport.p2p import P2PTransport

LOGGER = logging.getLogger(__name__)

_OP_SEND = 0x01
_OP_RECV = 0x02
_OP_SEND_OBJ = 0x03
_OP_RECV_OBJ = 0x04

_STATUS_OK = b"\x00"
_STATUS_ERR = b"\x01"


class TransportSidecar:
    """
    UDS server that bridges vLLM ↔ QUIC.

    peers_by_rank: list of peer node_ids in PP rank order. Index i = PP rank i.
    The current node's own rank is excluded; the mapping is:
      peer_idx (sent by vLLM as dst/src rank) → peers_by_rank[peer_idx] → QuicConn
    """

    def __init__(
        self,
        transport: "P2PTransport",
        uds_path: str,
        peers_by_rank: list[str],
    ) -> None:
        self._transport = transport
        self._uds_path = uds_path
        self._peers_by_rank = peers_by_rank
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        if os.path.exists(self._uds_path):
            os.unlink(self._uds_path)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self._uds_path
        )
        LOGGER.info("[sidecar] listening on %s", self._uds_path)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        try:
            os.unlink(self._uds_path)
        except OSError:
            pass

    def _peer_id(self, peer_idx: int) -> str:
        try:
            return self._peers_by_rank[peer_idx]
        except IndexError:
            raise ValueError(f"sidecar: invalid peer_idx={peer_idx}, known={self._peers_by_rank}")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername", "<vllm>")
        LOGGER.debug("[sidecar] client connected: %s", peer)
        try:
            while True:
                header = await reader.readexactly(1)
                opcode = header[0]

                if opcode in (_OP_SEND, _OP_SEND_OBJ):
                    await self._handle_send(reader, writer, opcode)
                elif opcode in (_OP_RECV, _OP_RECV_OBJ):
                    await self._handle_recv(reader, writer, opcode)
                else:
                    LOGGER.warning("[sidecar] unknown opcode 0x%02x, closing", opcode)
                    writer.write(_STATUS_ERR)
                    await writer.drain()
                    break

        except asyncio.IncompleteReadError:
            LOGGER.debug("[sidecar] client disconnected")
        except Exception:
            LOGGER.exception("[sidecar] client handler error")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _handle_send(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        opcode: int,
    ) -> None:
        # [1B peer_idx] [4B stream_id] [4B frame_len]
        meta = await reader.readexactly(9)
        peer_idx, stream_id, frame_len = struct.unpack("<BII", meta)
        frame = await reader.readexactly(frame_len)

        try:
            peer_id = self._peer_id(peer_idx)
            conn = self._transport._conns[peer_id]
            await conn.send_data(frame)
            writer.write(_STATUS_OK)
        except Exception as exc:
            LOGGER.error("[sidecar] send error: %s", exc)
            writer.write(_STATUS_ERR)
        await writer.drain()

    async def _handle_recv(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        opcode: int,
    ) -> None:
        # [1B peer_idx] [4B stream_id]
        meta = await reader.readexactly(5)
        peer_idx, _stream_id = struct.unpack("<BI", meta)

        try:
            peer_id = self._peer_id(peer_idx)
            conn = self._transport._conns[peer_id]
            frame = await conn.recv_data()
            writer.write(_STATUS_OK)
            writer.write(struct.pack("<I", len(frame)))
            writer.write(frame)
        except Exception as exc:
            LOGGER.error("[sidecar] recv error: %s", exc)
            writer.write(_STATUS_ERR)
            writer.write(struct.pack("<I", 0))
        await writer.drain()
