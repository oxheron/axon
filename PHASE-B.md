---
name: Phase B — P2P PP transport (NAT traversal)
overview: Replace Ray as the multi-node PP transport with a custom torch.distributed ProcessGroup backend (axon_quic) that routes PP tensors over direct QUIC/UDP connections. Scoped to PP=2 across two home-NAT nodes initially, with PP>2 planned as a follow-on milestone. Within-node parallelism (TP) is unconstrained. Coordinator handles signaling only. Supports three connection modes in order of preference: port forwarding (manual), UDP hole punching (automatic cone NAT), TURN relay (fallback, deferred).
status: in_progress (B-0 complete, B-1/B-2 ready, B-3 restructured)
---

# Phase B — P2P PP transport

Goal: two nodes behind home NAT, no third-party VPN service, coordinator used only for signaling, PP activations travel peer-to-peer.

## Architecture overview

```
[coordinator]  ←  signaling only (address exchange)
    ↓ ↓
[node A]  ←——— QUIC over UDP ———→  [node B]
  vLLM PP stage 0                   vLLM PP stage 1
  (AxonQuicProcessGroup)            (AxonQuicProcessGroup)
```

Ray is removed entirely. `AxonQuicProcessGroup` — a custom `torch.distributed` backend — establishes direct QUIC connections per PP peer pair during `initialize_model_parallel()`, then routes all PP tensor send/recv over those connections during inference. Nodes with routable IPs use port forwarding mode (B-2a) rather than the old Ray-based path; Ray is not required for any connection mode.

---

## Connection modes (priority order)

The transport layer attempts these in order, stopping at the first that succeeds:

| Mode | How | NAT requirement | User effort |
|------|-----|-----------------|-------------|
| **Port forwarding** | User opens a UDP port on their router; node advertises external IP:port | Any NAT | Manual, one-time |
| **UDP hole punching** | Coordinator exchanges external addresses; nodes punch simultaneously | Cone NAT (most home routers) | None |
| **TURN relay** | Coordinator or third-party TURN server relays PP traffic | Any NAT including symmetric | Deferred — future milestone |

In all three modes the wire protocol on top of UDP is **QUIC**, not raw UDP. See "Transport protocol" below. The three modes differ only in how the underlying UDP 5-tuple is established; once a UDP path exists, QUIC runs over it identically.

TURN relay is explicitly deferred. The Phase B implementation ships with the first two modes. The executor interface must be designed so TURN can be added as a third transport option without restructuring.

---

## Transport protocol: QUIC over UDP

PP activations range from a few KB per decode token to hundreds of MB for long-prompt prefill (see "Bandwidth / latency budget"). All traffic must arrive reliably and in order — a dropped or reordered tensor corrupts inference. Raw UDP provides none of that. Rather than reinvent TCP inside this project, the transport layer runs **QUIC** over the UDP socket established by whichever connection mode succeeded.

Rationale:
- **Reliability, ordering, retransmission, congestion control** — all provided by QUIC. Activations cannot tolerate loss or reordering; building these on raw UDP is a large, bug-prone subproject.
- **Fragmentation / path MTU** — QUIC handles PMTU discovery. A single 16 MB activation would otherwise require ~12k manually-framed UDP datagrams with custom reassembly.
- **Multiplexed streams** — multiple microbatches or control/data channels can share one connection without head-of-line blocking between streams.
- **Works with hole punching** — QUIC runs over a single UDP 5-tuple, which is exactly what hole punching produces. The hole punch establishes the NAT mapping; QUIC runs on top. This is the same pattern used by WebRTC data channels, libp2p, and Tailscale.
- **Encryption path** — QUIC mandates TLS 1.3, which gives us an encryption story essentially for free when we stop deferring it (see "Explicitly deferred").

Library choice: `aioquic` is the first choice. A benchmark is still required (see B-0 Q5) — if `aioquic` shows per-send latency > 5 ms on small (decode-sized) payloads or throughput < 40 Mbps on large (prefill-sized) payloads, the fallback is a Rust QUIC sidecar (`quinn` / `s2n-quic`) exposed via a local UDS.

Tensor framing: activations are sent as length-prefixed frames over a dedicated QUIC stream per tag/stream_id. Frame header carries dtype, shape, and a tag so the receiver can reconstruct the tensor without out-of-band coordination. Exact header layout specified in B-3.

---

## Bandwidth / latency budget

What actually crosses the PP boundary is the hidden state `[num_tokens, hidden_size]` in the activation dtype — **not** KV cache (that stays local to whichever stage owns those layers). This makes the budget story very different between decode and prefill.

**Decode (one token at a time).** Traffic per PP boundary crossing is `1 × hidden_size × bytes_per_element`:

| Model | hidden | BF16/token | FP8/token |
|---|---|---|---|
| MiniMax M2.7 | 3072 | 6 KB | 3 KB |
| GLM-5.1 | 6144 | 12 KB | 6 KB |

At 50 Mbps (~6.25 MB/s effective), 12 KB is ~2 ms of wire time. **Decode is not bandwidth-bound, it is latency-bound.** Each generated token pays one one-way network hop per PP boundary. On a 20 ms RTT home link that's ~10 ms per hop; with PP=2 that's a 10 ms floor per token (theoretical ceiling ~100 tok/s from network alone). The target of 5–10 tok/s fits comfortably inside this ceiling. The binding constraint on decode throughput is GPU compute, not the network.

**Prefill (whole prompt at once).** Traffic scales linearly with prompt length:

| Prompt tokens | MiniMax M2.7 @ BF16 | GLM-5.1 @ BF16 | Wire time @ 50 Mbps (M2.7 BF16) |
|---|---|---|---|
| 1 K | 6 MB | 24 MB | ~1 s |
| 4 K | 24 MB | 96 MB | ~4 s |
| 16 K | 96 MB | 384 MB | ~15 s |
| 64 K | 384 MB | 1.5 GB | ~60 s |

Prefill is where the bandwidth budget actually bites. Time-to-first-token at longer prompts is dominated by the uplink, not the GPUs. Users accustomed to hosted APIs will find 16K+ prompts feel slow on residential links without mitigation.

**Mitigations (required in scope, not deferred):**

1. **FP8 on the wire.** Halves prefill transit time. Makes the difference between tolerable and unusable TTFT at 16K prompts. This is promoted from deferred to B-3 scope — see B-3.
2. **Overlap transit with compute via chunked prefill.** Multiplexed QUIC streams allow chunk N's activations to transit to stage 2 while stage 1 is still computing chunk N+1. Pipelining the wire with the GPU hides most of the prefill network cost in the steady state.
3. **Cap PP depth.** Each additional PP stage adds one network hop per token on the decode hot path. **Phase B is explicitly scoped to PP=2 only.** PP≥3 is a planned follow-on milestone — the transport (`P2PTransport`) and ProcessGroup are both designed for N peers, not just 2, so extension should be incremental. PP=2 must ship and be validated against the performance targets before any work on PP≥3 begins.

**Budget summary (target for B-5 acceptance):**

- Hardware assumption: user chooses either a quantized or smaller variant of the target models such that weights fit on their per-node GPUs. This plan does not prescribe a specific VRAM tier.
- Network assumption: ~50 Mbps symmetric residential uplink, ≤30 ms RTT between nodes.
- Topology: PP=2 across two nodes. Within-node parallelism (TP) is the node's own business; see "Parallelism scope" below.
- **Decode target:** 5–10 tok/s sustained, GPU-bound. Network contributes ~10–15 ms/token and is absorbed into the pipeline bubble.
- **TTFT target:** ≤2 s at 1K prompt, ≤10 s at 4K prompt (with FP8-on-wire + overlapped chunked prefill). Longer prompts degrade gracefully; document the trade-off rather than treating it as a bug.
- **Red-flag thresholds:** if B-5 shows <5 tok/s decode on target hardware, or TTFT at 4K exceeds 15 s, the budget has failed and scope must shrink (smaller model, or promote TURN/relay for users whose RTT exceeds the assumption).

---

## Parallelism scope

Phase B is deliberately narrow on parallelism topology to make the P2P transport tractable:

- **PP=2 across nodes.** Exactly one inter-node boundary. This is the only cross-node topology Phase B validates. PP≥3 is a planned follow-on — the `P2PTransport` interface and `AxonQuicProcessGroup` are both written for N peers, so adding more stages should not require a redesign. The prerequisite is that PP=2 ships and passes the B-5 performance targets first.
- **Within-node parallelism is the node's own concern.** A single node may run TP internally across its local GPUs; this does not involve the P2P transport at all, since TP collectives stay on local NVLink / PCIe. The P2P transport only sees the boundary between the two nodes. Nothing in B-1 through B-4 should assume a particular intra-node parallelism shape.
- **Hardware reality.** The full BF16 weights of MiniMax M2.7 (230 B total) or GLM-5.1 / GLM-5 (357–744 B total) do not fit on typical home hardware. The expected Phase B use case is either a quantized variant (FP8/FP4), a smaller sibling in the same family, or a different model in the ~30–70 B class. This plan does not prescribe which; it just supports the PP=2 transport cleanly regardless.

---

---

## Todos

### B-0 — Spike: vLLM executor interface + PP tensor path ✓ COMPLETE

As described above. Written output required before proceeding.

**Status:** Done — see "B-0 Spike Output" appendix above the Success Criteria section. Key finding: integration point is a custom `torch.distributed` ProcessGroup backend, not a custom executor. B-3 restructured accordingly.

---

### B-1 — Coordinator: WebSocket control channel

**File:** `coordinator/internal/server/` (new handler + types)

Nodes communicate with the coordinator over a single persistent WebSocket connection opened at startup. All coordinator→node pushes (including `StartupConfig` and `SignalReady`) and all node→coordinator messages (registration, signal submission) flow over this connection. This works through NAT with no port forwarding on the node side because the node initiates the outbound connection.

#### Connection lifecycle

```
node startup
  → WS connect to wss://<coordinator>/ws?cluster_id=&node_id=&token=<cluster_token>
  ← coordinator pushes:  {type: "startup_config", ...StartupConfig fields}
  → node sends:          {type: "signal", external_addr, external_port, transport_mode}
  ← coordinator pushes:  {type: "signal_ready", cluster_id, peers: [NodeSignal, ...]}
  [connection stays open; coordinator pushes keepalive pings every 30s]
```

Authentication: the `token` query parameter carries the per-cluster pre-shared token. The coordinator rejects connections with an unknown `cluster_id`, unrecognised `node_id`, or mismatched token before upgrading to WebSocket. No bearer header needed — the token is in the URL at upgrade time and the connection is then trusted for its lifetime.

The coordinator does not interpret addresses or participate in hole punching. It stores and redistributes opaque address blobs.

#### Message types

All frames are JSON. The `type` field discriminates:

```go
// node → coordinator
type SignalMsg struct {
    Type          string `json:"type"` // "signal"
    ExternalAddr  string `json:"external_addr"`
    ExternalPort  int    `json:"external_port"`
    TransportMode string `json:"transport_mode"` // "port_forward" | "hole_punch"
}

// coordinator → node (pushed when all peers have signaled or timeout fires)
type SignalReadyMsg struct {
    Type      string       `json:"type"` // "signal_ready"
    ClusterID string       `json:"cluster_id"`
    Peers     []NodeSignal `json:"peers"`
}

type NodeSignal struct {
    NodeID        string `json:"node_id"`
    ExternalAddr  string `json:"external_addr"`
    ExternalPort  int    `json:"external_port"`
    TransportMode string `json:"transport_mode"`
}

// coordinator → node (error case)
type ErrorMsg struct {
    Type    string `json:"type"` // "error"
    Code    string `json:"code"` // e.g. "signal_timeout", "unknown_node"
    Message string `json:"message"`
}
```

`StartupConfig` is delivered as a `{type: "startup_config", ...}` frame immediately after the connection is accepted. Its field set is unchanged from the existing definition.

#### Signaling deadline

When all expected nodes have sent a `signal` message **or** the deadline elapses, the coordinator pushes `signal_ready` to every connected node in the cluster. Default deadline: 60s after the first node signals. Configurable via `--signal-timeout`. If the deadline fires with missing nodes, the coordinator pushes `{type: "error", code: "signal_timeout", message: "nodes X, Y never signaled"}` to connected nodes and closes their connections.

#### Re-signaling

UDP NAT mappings commonly expire after 30–120s of idle traffic. A node may send another `signal` frame at any time to update its record; the coordinator immediately pushes a fresh `signal_ready` to all peers. Data-path keepalive is handled by QUIC (built-in `PING` frames) so an active cluster should rarely need re-signaling, but the path exists for crash-restart and long-idle cases.

**Library:** `nhooyr.io/websocket` (zero indirect dependencies, context-aware). On the node side (Python): `websockets` library.

Lifecycle state addition: `signaling` between `assigned` and `backend_joined`. Nodes enter `signaling` after the WebSocket connection is accepted and `startup_config` is received; they exit after receiving `signal_ready` and completing B-2.

**Acceptance:** two nodes both receive `signal_ready` with each other's addresses before attempting to start the executor. Connections with invalid token or unknown node_id are rejected at the WebSocket upgrade (HTTP 401/403, never upgraded). A node that never signals causes the cluster to fail with a named-node error after the configured timeout rather than hanging forever.

---

### B-2 — Node: transport negotiation module

**New package:** `node/src/transport/`

Responsible for: determining the local node's external address and establishing a direct UDP socket to each PP peer.

#### Sub-task B-2a — Port forwarding mode

If `--advertise-port` is set (added in Phase A, A-2) and differs from `--port`, the node is assumed to be port-forwarded. In this mode:

- External addr = `--advertise-host`, external port = `--advertise-port`
- No STUN required
- Node sends `{type: "signal", external_addr, external_port, transport_mode: "port_forward"}` over the coordinator WebSocket (opened in B-1)
- Binds a UDP socket on the configured port and waits for peers to connect

This mode requires zero NAT traversal code. If both peers are port-forwarded, connection is trivial.

#### Sub-task B-2b — UDP hole punching mode

Used when `--advertise-port` is not set (default path).

1. **STUN discovery + NAT behavior probe**: use `pystun3` to discover external IP:port. In addition, query a *second* STUN server and compare the mapped port: if the two mapped ports differ, the local NAT is likely symmetric (address/port-dependent mapping) and hole punching will not work with a standard cone peer. In that case log a clear diagnostic up front and skip straight to the fallback path instead of attempting a doomed punch. This is a pragmatic substitute for full RFC 5780 behavior discovery and catches the common symmetric-NAT case deterministically.
2. **Send `signal` frame over WebSocket**: send `{type: "signal", external_addr, external_port, transport_mode: "hole_punch"}` over the existing coordinator WebSocket connection (opened in B-1).
3. **Wait for `signal_ready` frame**: the coordinator pushes this over the same WebSocket connection when all peers have signaled. No polling or separate request needed.
4. **Simultaneous open**: for each PP peer, send small UDP probe packets to their external addr:port while listening on the local port. Retry with exponential backoff for up to **30 seconds** before declaring failure. (WebRTC's ICE budget is in this range; 5s is too short for slower NATs.)
5. **Fallback signal**: if hole punch fails, log a clear message that distinguishes the diagnosis (symmetric NAT detected up front vs. punch timed out) and directs the user to `--advertise-port` with a manually forwarded port (or wait for TURN in a future release).

Libraries: `pystun3` for STUN (two STUN servers configured for the behavior probe). No additional dependencies needed for the hole punch packets themselves — raw UDP. Once a UDP path exists, `aioquic` (or sidecar — decided in B-0) runs on top; see "Transport protocol".

#### Common interface

Both modes expose the same interface to the executor (B-3). The interface is **synchronous** — vLLM's worker forward pass is synchronous PyTorch, and bridging async↔sync at the hottest loop in the system is a known source of latency and deadlocks. The transport owns an internal IO thread (and/or the QUIC sidecar process) and presents a blocking API:

```python
class P2PTransport:
    def connect(self, peers: list[PeerEndpoint], timeout: float) -> None: ...
    def send_tensor(self, peer_id: str, tensor: torch.Tensor, *, stream_id: int) -> None: ...
    def recv_tensor(self, peer_id: str, *, stream_id: int) -> torch.Tensor: ...
    def close(self) -> None: ...
```

Note `tensor: torch.Tensor`, not `bytes`: the transport owns (de)serialization including dtype/shape/device metadata so callers don't have to re-derive framing. `stream_id` maps to a QUIC stream for head-of-line-blocking isolation between concurrent microbatches. Exact frame layout specified in B-3.

`AxonQuicProcessGroup.__init__()` calls `connect()` during `initialize_model_parallel()`. TURN (future) would be a third implementation of this interface.

**Acceptance:** two nodes on a home network (cone NAT) reach each other via UDP and complete a QUIC handshake. Log clearly which mode succeeded and, on failure, which NAT class was detected.

---

### B-3 — Custom `torch.distributed` ProcessGroup backend (`axon_quic`)

**New package:** `node/src/axon_quic/`

B-0 established that PP tensors cross node boundaries via `torch.distributed.isend/irecv` inside `GroupCoordinator.isend_tensor_dict/irecv_tensor_dict` (`vllm/distributed/parallel_state.py:899–901, 1002–1004`), fired from inside `GPUWorker.execute_model()`. This is below the vLLM executor interface and cannot be intercepted there. The correct integration point is a custom `torch.distributed` backend — a `ProcessGroup` subclass whose `send()`/`recv()` primitives route over `P2PTransport` (B-2) instead of NCCL.

#### Registration

PyTorch allows custom backends via `torch.distributed.Backend` and a registered `ProcessGroup` subclass. The backend is named `"axon_quic"` and registered at import time:

```python
# node/src/axon_quic/__init__.py
import torch.distributed as dist
from .process_group import AxonQuicProcessGroup

dist.Backend.register_backend(
    "axon_quic",
    AxonQuicProcessGroup,
    devices=["cpu", "cuda"],
)
```

This package is loaded via the vLLM plugin system (`vllm.general_plugins` entry point) so it is registered before `initialize_model_parallel()` runs. No vLLM fork is required.

vLLM is then started with `VLLM_DISTRIBUTED_BACKEND=axon_quic` (or `--distributed-backend axon_quic` if the flag exists for the installed version). `initialize_model_parallel()` passes this backend string to `torch.distributed.new_group(..., backend="axon_quic")` at `parallel_state.py:339`, which constructs `AxonQuicProcessGroup` for the PP group. The TP group uses the default intra-node backend (NCCL), since TP is constrained to `--tensor-parallel-size ≤ GPUs-per-node` in Phase B.

#### ProcessGroup interface

`AxonQuicProcessGroup` subclasses `torch.distributed.ProcessGroup` and implements the subset of operations vLLM's PP path actually calls:

```python
class AxonQuicProcessGroup(dist.ProcessGroup):
    def __init__(self, store, rank, size, timeout, transport: P2PTransport):
        super().__init__(store, rank, size)
        self._transport = transport

    # Point-to-point — the only ops used on the PP group
    def send(self, tensors: list[torch.Tensor], dst: int, tag: int) -> Work: ...
    def recv(self, tensors: list[torch.Tensor], src: int, tag: int) -> Work: ...

    # isend/irecv are the async variants called by GroupCoordinator
    def isend(self, tensors: list[torch.Tensor], dst: int, tag: int) -> Work: ...
    def irecv(self, tensors: list[torch.Tensor], src: int, tag: int) -> Work: ...

    # Collective stubs — not used on PP group, raise clearly if called
    def allreduce(self, *args, **kwargs): raise NotImplementedError(...)
    def broadcast(self, *args, **kwargs): raise NotImplementedError(...)
    # ... etc
```

`tag` maps to `stream_id` in the transport layer, giving each outstanding PP send/recv its own QUIC stream. `Work` is a thin wrapper around a Python `Future` that satisfies PyTorch's `Work.wait()` contract.

`transport.connect(peers)` is called inside `__init__` (during `new_group()`), so by the time `initialize_model_parallel()` returns, the QUIC connection to the PP peer is established and the node can transition to `backend_joined`.

#### Tensor wire format

Each activation is sent as a framed message over a dedicated QUIC stream (one stream per `stream_id`/`tag`). Frame header — fixed 32 bytes, little-endian:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | magic `0x41584f4e` ("AXON") |
| 4 | 2 | version (currently 1) |
| 6 | 2 | dtype enum (f16=1, bf16=2, f32=3, f8_e4m3=4, …) |
| 8 | 4 | rank (number of shape dims, max 8) |
| 12 | 8 | tag / stream id (mirrors the `tag` argument from PyTorch) |
| 20 | 4 | payload length in bytes |
| 24 | 8 | reserved / flags (bit 0: FP8-cast applied; rest zero in v1) |

Followed by `rank × 4` bytes of shape (uint32 per dim), then `payload length` bytes of raw tensor data.

Note: `GroupCoordinator` sends shape/dtype metadata on a separate Gloo CPU group before sending the tensor on the device group (`parallel_state.py:851–906`). `AxonQuicProcessGroup` handles only the device-group (tensor data) side. The metadata group remains on Gloo; this is fine because metadata messages are tiny (a few bytes) and not latency-critical relative to the tensor transit itself. If Gloo is unavailable or undesirable in the target environment, the metadata can be embedded in the AXON frame header above and the CPU group can be stubbed out — defer this unless Gloo creates a concrete problem in B-5.

#### FP8 on-wire casting (in scope for B-3, not deferred)

Per the bandwidth budget, prefill TTFT at 16K+ prompts is unusable without halving wire size. `AxonQuicProcessGroup.isend()` MAY downcast BF16/FP16 tensors to FP8 (e4m3) on GPU before serialization, and the receiver upcasts on GPU before returning the tensor to the caller. Controlled by a per-cluster env var `AXON_WIRE_DTYPE={native,fp8}`. The cast flag in the frame header tells the receiver which path to take. FP8 downcast/upcast uses existing PyTorch/vLLM FP8 kernels — cost is ~1 kernel launch per tensor, negligible vs transit savings. Default: `fp8` for PP=2 across nodes, `native` for loopback testing.

#### Chunked-prefill pipelining

vLLM's chunked-prefill scheduler issues successive chunks with distinct `tag` values. Each chunk's `isend` call therefore gets its own QUIC stream (`tag` → `stream_id`), so chunk N's activation transits to stage 2 while stage 1 is still computing chunk N+1. This overlap is what makes prefill tolerable at realistic prompt lengths. `AxonQuicProcessGroup` does not need any special logic here — it simply honors the `tag` argument it receives.

#### Device handling

Tensors arrive at `isend()` on GPU. The transport moves them to CPU via pinned memory (one pre-allocated pinned buffer per peer per stream, allocated during `connect()`) for framing. FP8 downcast happens on GPU *before* the device→host copy, so the CPU-side buffer holds the already-smaller payload. On the recv side, the frame is written into a pinned buffer, then copied to GPU and upcast if needed. Pre-allocated pinned buffers avoid `cudaMallocHost` on the forward-pass hot path.

#### Error handling

If the QUIC connection drops mid-inference, `isend()`/`irecv()` must raise (not hang). The `Work` future has a built-in deadline: if `Work.wait()` is not satisfied within 5 seconds of the expected wire time (estimated from payload size and measured link throughput), it raises `ConnectionError`. This propagates out of `GroupCoordinator.isend_tensor_dict()` as an uncaught exception, which vLLM's engine loop surfaces as an inference error rather than a deadlock.

#### vLLM launch integration

In `node/src/strategy.py`, when `backend == "axon_p2p"` (the per-cluster field set by the coordinator per B-4), add `VLLM_DISTRIBUTED_BACKEND=axon_quic` to the vLLM process environment before launch. The `axon_quic` package is installed alongside the node and auto-registered via the `vllm.general_plugins` entry point, so no additional flags are needed. The node does not set `--distributed-executor-backend`; vLLM uses its default executor (`mp` for single-node, which is correct since each axon node is a single vLLM instance).

**Acceptance:** PP forward pass completes end-to-end through the custom ProcessGroup backend on two nodes connected via B-2 transport. `GroupCoordinator.isend_tensor_dict/irecv_tensor_dict` is confirmed (via logging) to route through `AxonQuicProcessGroup` and not NCCL. QUIC connection loss during inference surfaces as an exception within 5s, not a hang.

---

### B-4 — Remove Ray from multi-node path

Once B-3 is working, clean up:

- `node/src/workers/ray_worker.py`: delete. `join_ray_cluster` is unused on all paths.
- `node/src/runtime/lifecycle.py`: remove `join_ray_cluster` call.
- `coordinator/src/main.go`: delete `maybeStartRayHead` and `--autostart-ray-head`. Nodes with routable IPs use port forwarding mode (B-2a) with the `axon_p2p` backend — they do not need a Ray head, and opening Ray's port range (6379, 8265, 10001–19999) is exactly what we are eliminating.
- `coordinator/internal/server/server.go`: remove `ray_head_address` from `StartupConfig`. It is not used on the `axon_p2p` path and there is no longer a Ray path to fall back to.

Ray is removed from all paths. Users who previously required routable IPs and used the Ray backend should use `axon_p2p` with `--advertise-host` + `--advertise-port` (port forwarding mode, B-2a), which requires only a single forwarded UDP port — not Ray's sprawling port range.

**Gating condition (explicit).** The coordinator always sets `backend: "axon_p2p"` on `StartupConfig`. The `backend` field is kept in the struct for forward compatibility (e.g., a future TURN-only mode) but `"axon_p2p"` is the only accepted value. Any coordinator or node code that still checks `backend == "ray"` should be deleted, not gated.

---

### B-5 — Integration test: two-node NAT (hole punch path)

Run Phase B end-to-end on two machines behind real home NAT (or a simulated NAT environment — see below):

- Both nodes start with no `--advertise-port`
- Coordinator starts, nodes register, signaling completes, hole punch succeeds, QUIC handshake completes
- Executor starts, PP forward pass completes
- `inference_ready: true`, curl returns a valid response

**NAT simulation.** The test environment explicitly covers two cases, not just one:

1. **Cone-ish NAT** — Linux netns + `iptables -t nat -A POSTROUTING -j MASQUERADE` on each node's namespace, which gives endpoint-independent mapping behavior. Punch should succeed here.
2. **Symmetric NAT** — same setup but with `--random-fully` on the MASQUERADE rule (or an explicit `SNAT ... --to-ports <range>` with randomization) to force port-dependent mapping. Punch should fail here, and the NAT-behavior probe in B-2b-1 should detect this up front and report it clearly without the 30s punch attempt.

Both cases must produce the correct diagnostic output; only case 1 is expected to reach `inference_ready`.

**Acceptance criteria:** case 1 passes the same smoke test as Phase A on NAT nodes with no manual port forwarding, **and** meets the decode/TTFT targets from the bandwidth budget (≥5 tok/s sustained decode on target hardware, ≤10 s TTFT at 4 K prompt with FP8-on-wire + chunked-prefill overlap enabled). Case 2 fails fast with a NAT-class diagnostic pointing the user at `--advertise-port`.

---

### B-6 — Integration test: port forwarding path

Same as B-5 but both nodes use `--advertise-port` with manually forwarded ports.

---

## Explicitly deferred (future milestone)

- **TURN relay**: third transport mode in `P2PTransport`. Coordinator or a dedicated TURN server relays packets when hole punch fails (symmetric NAT) or when NAT behavior probing (B-2b-1) detects a doomed configuration up front. Interface is already designed to accept it (B-2). Implement after B-5 validates the core path.
- **Authenticated + verified QUIC TLS**: QUIC mandates TLS 1.3, so the data path is encrypted as soon as B-3 ships. However, Phase B will use a minimal cert setup (e.g., self-signed per-cluster certs distributed via coordinator signaling, or raw-keys mode) rather than a full PKI. Promote to proper mutual authentication before any production/untrusted-peer deployment. Note: coordinator signaling is already authenticated per B-1 via the cluster pre-shared token — that piece is not deferred.
- **TLS on coordinator + node HTTP**: same deferral as Phase A runbook noted.
- **PP ≥ 3 across nodes**: planned follow-on milestone. The transport (`P2PTransport`) and `AxonQuicProcessGroup` are both designed for N peers — B-2 already handles arbitrary peer lists and the ProcessGroup can hold connections to multiple nodes. The prerequisite is that PP=2 ships and passes B-5 performance targets first. Each added PP stage adds one network hop per decode token; at residential RTTs this costs ~10 ms/hop, so PP=3 roughly halves decode throughput vs PP=2. A latency-aware scheduler (or larger microbatch pipelining) would be needed to recover that loss before PP≥3 is practical.
- **Further activation compression beyond FP8**: FP8-on-wire is in B-3 scope. Additional compression (INT4 activations, LZ4/zstd on top of FP8, perceptual-loss-tolerant codecs) is deferred until B-5 tells us whether FP8 alone clears the TTFT target. The compression flag bits in the wire header reserve space for this.

---

## B-0 Spike Output — Decision Doc

*Completed 2026-04-23. Source: vLLM commit read from `/Users/micah/dev/axon2/axon/vllm/vllm/`. All file paths below are relative to the vllm package root (`vllm/vllm/vllm/`).*

---

### Q1 — Custom executor interface

The abstract base class is `Executor` in `v1/executor/abstract.py:37`. Key class variables:
- `uses_ray: bool = False` (line 44)
- `supports_pp: bool = False` (line 45)

Required abstract methods:
- `_init_executor()` (line 115)
- `collective_rpc()` (lines 199–202) — this is the primary dispatch point for all worker calls
- `check_health()` (line 275)

Registration: `Executor.get_class()` (line 48) accepts either a built-in string (`"ray"`, `"mp"`, `"uni"`, `"external_launcher"`) or a fully-qualified class name string, resolved via `resolve_obj_by_qualname()` (line 82). The class is type-checked as a subclass of `Executor` (lines 83–86). The CLI flag is `--distributed-executor-backend` (`engine/arg_utils.py:901`).

**Conclusion:** implementing a custom executor requires subclassing `Executor`, setting `supports_pp = True`, and passing the fully-qualified class name via `--distributed-executor-backend`. No fork required for the executor layer itself.

---

### Q2 — Where PP tensors cross node boundaries

**Full call chain:**

```
GPUWorker.execute_model()                  v1/worker/gpu_worker.py:800–835
  → get_pp_group().irecv_tensor_dict()     distributed/parallel_state.py:946
  → get_pp_group().isend_tensor_dict()     distributed/parallel_state.py:851
    → torch.distributed.isend(tensor,      parallel_state.py:899–901
           dst=self.ranks[dst],
           group=comm_group)               ← NCCL ProcessGroup
    → torch.distributed.irecv(...)         parallel_state.py:1002–1004
```

The tensor being sent is `hidden_states` (shape `[num_tokens, hidden_dim]`) assembled into an `IntermediateTensors` dict at `v1/worker/gpu_model_runner.py:4083`. Metadata (shapes, dtypes) travels on a separate Gloo CPU group; the tensor data travels on the NCCL device group.

**This all happens inside the worker's forward pass, not inside the executor's `collective_rpc()` path.** The executor dispatches `execute_model` to the worker as an opaque RPC. By the time control is inside the worker, the PP send/recv fires directly against `torch.distributed`.

---

### Q3 — Can a custom executor own the tensor transport? (CRITICAL FINDING)

**No. The integration point is not the executor layer.**

PP tensor send/recv is invoked from inside `GPUWorker.execute_model()` → `GPUModelRunner.execute_model()` → `GroupCoordinator.isend_tensor_dict/irecv_tensor_dict()` → `torch.distributed.isend/irecv()`. This entire chain fires below the executor's `collective_rpc()` dispatch and is not interceptable at the executor level.

The `GroupCoordinator` (and its underlying `ProcessGroup`) is created per-worker during `initialize_model_parallel()` (`parallel_state.py:1486–1643`). It is instantiated with the torch distributed backend (typically `"nccl"`) at `parallel_state.py:339–340`:
```python
device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
```

**The only way to route PP traffic over a custom transport (QUIC over UDP) is to register a custom `torch.distributed` backend** — i.e., a custom `ProcessGroup` implementation — and tell vLLM to use it via `--distributed-backend` (or equivalent env var). The custom ProcessGroup's `send()` and `recv()` methods would then route over QUIC instead of NCCL.

**This changes the implementation plan. B-3 is not a custom `DistributedExecutorBackend`; it is a custom `torch.distributed` `ProcessGroup` backend.**

---

### Q4 — Fork required?

**No fork required**, but the approach changes slightly given Q3's finding:

- For the *executor layer*: plugin registration via fully-qualified class name is sufficient (no fork).
- For the *custom ProcessGroup backend*: PyTorch supports registering custom backends via `torch.distributed.Backend` and `torch.distributed.ProcessGroup`. This is a Python-level extension and also does not require forking vLLM. vLLM accepts `--distributed-backend` to override the torch distributed backend string (e.g., set to `"axon_quic"` once registered).

The vLLM plugin system (`plugins/__init__.py:28–66`, entry point group `"vllm.general_plugins"`) can be used to load the custom backend registration at startup without patching vLLM source.

---

### Q5 — QUIC library viability / activation sizes

PP activations are `hidden_states` tensors of shape `[num_tokens, hidden_dim]`:

| Model class | hidden_dim | 1-token BF16 | 1K-token BF16 | 4K-token BF16 |
|---|---|---|---|---|
| ~7B (LLaMA-7B) | 4096 | 8 KB | 8 MB | 32 MB |
| ~70B (LLaMA-70B) | 8192 | 16 KB | 16 MB | 64 MB |
| MiniMax M2.7 | ~3072 | 6 KB | 6 MB | 24 MB |

For decode (1 token/step): 6–16 KB per crossing — **latency-bound, not bandwidth-bound**. QUIC overhead on small payloads is the risk; `aioquic` adds ~0.5–2 ms per small send on loopback. This must be benchmarked. If it exceeds ~5 ms per PP crossing, decode throughput degrades below 5 tok/s on 20 ms residential RTT.

For prefill: payloads reach tens to hundreds of MB — **bandwidth-bound**. QUIC's throughput ceiling over a 50 Mbps link is the constraint; congestion control behavior matters here.

**Benchmark still required** (not skipped by this spike): measure `aioquic` at 10 KB (decode) and 32 MB (4K-token prefill) payloads over a loopback with 20 ms induced RTT. If decode latency > 5 ms or prefill throughput < 40 Mbps, switch to Rust QUIC sidecar (`quinn`/`s2n-quic`) via local UDS.

---

### Q6 — TP AllReduce intra-node confinement

**TP and PP use completely separate, non-overlapping ProcessGroups.** From `parallel_state.py`:

- TP groups (line 1572): `all_ranks.view(-1, tensor_model_parallel_size).unbind(0)` — groups of TP-size ranks along the TP axis.
- PP groups (line 1630–1631): ranks transposed along the PP axis — groups of PP-size ranks.

For an 8-GPU layout with TP=2, PP=4: TP groups are `[0,1],[2,3],[4,5],[6,7]` (intra-node pairs); PP groups are `[0,2,4,6],[1,3,5,7]` (cross-node chains). Each rank belongs to exactly one TP group and one PP group; the groups never overlap.

TP AllReduce calls `get_tp_group().all_reduce()` → fires on the TP ProcessGroup only. **It cannot leak into the PP ProcessGroup by construction.**

**Caveat:** if a user sets `--tensor-parallel-size` larger than GPUs-per-node, TP groups intentionally span nodes. In that case, TP AllReduce will generate cross-node traffic on the TP ProcessGroup — but this is by explicit user configuration and is expected. The custom ProcessGroup backend (B-3) must handle TP collectives as well as PP send/recv if TP spans nodes; or alternatively, TP groups are restricted to intra-node ranks and only PP groups are routed over QUIC.

**Recommendation:** for Phase B, constrain `--tensor-parallel-size ≤ GPUs per node`. Document this. Each node may have any TP within its local GPUs; the custom ProcessGroup only handles the PP boundary.

---

### Plan restructuring required before B-1

B-3 in the original plan described a custom `DistributedExecutorBackend` registered as `"axon_p2p"`. **This is the wrong integration point.** The corrected plan:

| Original | Corrected |
|---|---|
| B-3: custom `DistributedExecutorBackend` subclass | B-3: custom `torch.distributed` `ProcessGroup` backend registered as `"axon_quic"` |
| Executor calls `transport.send_tensor/recv_tensor` | `ProcessGroup.send()/recv()` calls `transport.send_tensor/recv_tensor` |
| `strategy.py:54` sets `distributed_backend = "axon_p2p"` | vLLM is started with `--distributed-backend axon_quic` (torch distributed env var) |
| Executor owns connection setup | ProcessGroup `__init__` calls `transport.connect(peers)` during `initialize_model_parallel()` |

The `P2PTransport` interface (B-2) is unchanged. The tensor framing spec (B-3 wire format) is unchanged. The coordinator signaling (B-1) is unchanged. Only the vLLM integration layer changes from executor to ProcessGroup.

**B-1 and B-2 proceed as written. B-3 has been rewritten accordingly — see the B-3 todo section.**

---

## Success criteria

Two nodes behind **home NAT** (cone), no third-party VPN, PP=2 across nodes, arbitrary TP within each node:
- Coordinator log: `inference_ready: true`  
- Both nodes: vLLM using `axon_quic` ProcessGroup backend for PP communication; no Ray process running on either node  
- PP activations logged as traveling via direct QUIC-over-UDP connection between nodes (not coordinator)  
- Within-node TP collectives confirmed to stay local (no AllReduce leaking over the P2P link)  
- Decode sustained ≥5 tok/s on the target model at target hardware; TTFT ≤10 s at 4 K prompt  
- Valid response from user service  
- Port forwarding path also validated separately (B-6)
