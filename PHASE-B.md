---
name: Phase B — P2P PP transport (NAT traversal)
overview: Replace Ray as the multi-node PP transport with a custom vLLM executor backend that sends tensors over direct P2P connections. Scoped to PP=2 across two home-NAT nodes; within-node parallelism (TP) is unconstrained. Coordinator handles signaling only. Supports three connection modes in order of preference: port forwarding (manual), UDP hole punching (automatic cone NAT), TURN relay (fallback, deferred).
status: pending
---

# Phase B — P2P PP transport

Goal: two nodes behind home NAT, no third-party VPN service, coordinator used only for signaling, PP activations travel peer-to-peer.

## Architecture overview

```
[coordinator]  ←  signaling only (address exchange)
    ↓ ↓
[node A]  ←——— P2P transport ———→  [node B]
  vLLM PP stage 0                   vLLM PP stage 1
  (custom executor)                 (custom executor)
```

Ray is removed from the multi-node path entirely. The custom executor establishes direct sockets per PP peer pair during startup, then uses those sockets for tensor passing during inference. Ray remains available for single-node GPU management if needed but is not required.

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

Library choice (to be finalized in B-0): `aioquic` is the obvious Python option. If B-0 surfaces that Python-side QUIC throughput is inadequate for activation sizes, a Rust sidecar (`quinn` or `s2n-quic`) exposed via a local socket is the fallback. Decision recorded as part of B-0 output.

Tensor framing: activations are sent as length-prefixed frames over a dedicated QUIC stream per (peer, direction) pair. Frame header carries dtype, shape, and a microbatch/sequence ID so the receiver can reconstruct the tensor without out-of-band coordination. Exact header layout specified in B-3.

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
3. **Cap PP depth.** Each additional PP stage adds one network hop per token on the decode hot path. **Phase B is explicitly scoped to PP=2 only.** PP≥3 is deferred — it would push the decode latency budget past the 5–10 tok/s target on typical residential RTT. Users who need more stages are pointed at Phase A (routable IP + Ray) or future milestones.

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

- **PP=2 across nodes.** Exactly one inter-node boundary. This is the only cross-node topology Phase B promises to support. Higher PP depth is deferred.
- **Within-node parallelism is the node's own concern.** A single node may run TP internally across its local GPUs; this does not involve the P2P transport at all, since TP collectives stay on local NVLink / PCIe. The P2P transport only sees the boundary between the two nodes. Nothing in B-1 through B-4 should assume a particular intra-node parallelism shape.
- **Hardware reality.** The full BF16 weights of MiniMax M2.7 (230 B total) or GLM-5.1 / GLM-5 (357–744 B total) do not fit on typical home hardware. The expected Phase B use case is either a quantized variant (FP8/FP4), a smaller sibling in the same family, or a different model in the ~30–70 B class. This plan does not prescribe which; it just supports the PP=2 transport cleanly regardless.

---

## Spike (do this first — B-0)

**Before writing any executor code**, spend 1–2 days reading vLLM source to answer:

1. What interface does a custom `--distributed-executor-backend` class need to implement? Read `vllm/executor/` base classes and how `ray` backend is registered.
2. Where exactly do PP tensors cross node boundaries? Trace a PP forward pass: `vllm/worker/worker.py` → distributed send/recv ops → what sits below (Ray actor call, NCCL, custom socket?). In particular, determine whether the actual tensor send/recv happens at the executor layer or inside `torch.distributed` via `ProcessGroup`/NCCL below it. These are different integration points and the answer shapes B-3.
3. Can the custom executor fully own the tensor transport, or does NCCL get invoked below the executor interface in a way that bypasses it? If the latter, the work is a custom `torch.distributed` backend, not a custom executor.
4. Is a vLLM fork required, or is executor plugin registration sufficient?
5. QUIC library viability: benchmark `aioquic` throughput for realistic activation payloads (e.g., 1 KB decode-sized and 10 MB prefill-sized) over loopback and over a bandwidth-capped link with ~20 ms induced RTT. Confirm:
   - Small-payload latency is competitive with raw UDP (decode path is latency-sensitive — see bandwidth budget).
   - Large-payload throughput saturates the link (prefill path is bandwidth-sensitive).
   If `aioquic` fails either test, decide now whether to use a Rust QUIC sidecar (`quinn` / `s2n-quic`) exposed via a local UDS or shared-memory channel.
6. Intra-node parallelism compatibility: confirm that a vLLM instance running with `--tensor-parallel-size N` on each side of the PP boundary does not introduce cross-node collectives beyond the PP send/recv. TP AllReduce must stay intra-node; only the PP hop crosses the wire. If B-0 surfaces any cross-node collective leaking out of TP, that breaks the bandwidth budget and must be resolved before B-3.

**Output:** a short written decision doc (can live in this file as an appendix) answering the six questions above and confirming or adjusting the implementation plan.

**B-1 through B-4 are provisional pending B-0.** The types, state transitions, and code diffs prescribed below assume an executor-level integration. If the spike shows the correct integration point is a custom `torch.distributed` `ProcessGroup` backend (or something else), restructure B-1 through B-4 accordingly before implementation. Do not start B-1 through B-4 until the spike is done and any required restructuring is reflected here.

---

## Todos

### B-0 — Spike: vLLM executor interface + PP tensor path

As described above. Written output required before proceeding.

---

### B-1 — Coordinator: signaling endpoint

**File:** `coordinator/internal/server/` (new handler + types)

Add a `/signal` endpoint that nodes use to exchange transport addresses after receiving their startup config. The flow:

1. Node receives `StartupConfig` from coordinator (existing flow). `StartupConfig` includes a per-cluster pre-shared token the node will use to authenticate signal submissions.
2. Node determines its external address (via STUN or manual config — see B-2).
3. Node POSTs `{ node_id, external_addr, external_port, transport_mode }` to `/signal`, authenticated with the cluster token (bearer header) and bound to `node_id` the coordinator already expects in this cluster. Submissions for unknown `node_id` or with a mismatched token are rejected.
4. Coordinator stores signal records keyed by `cluster_id + node_id`.
5. When all expected nodes have signaled **or** the registration deadline elapses, the coordinator either broadcasts `SignalReady` to each node containing the full peer signal table, or fails the cluster with a clear error identifying which nodes never signaled. Default deadline: 60s after the first node signals. Configurable via `--signal-timeout`.

The coordinator does not interpret addresses or participate in hole punching. It stores and redistributes opaque address blobs.

Types to add:
```go
type NodeSignal struct {
    NodeID        string `json:"node_id"`
    ExternalAddr  string `json:"external_addr"`
    ExternalPort  int    `json:"external_port"`
    TransportMode string `json:"transport_mode"` // "port_forward" | "hole_punch"
}

type SignalReady struct {
    ClusterID string       `json:"cluster_id"`
    Peers     []NodeSignal `json:"peers"`
}
```

**Re-signaling.** UDP NAT mappings commonly expire after 30–120s of idle traffic, and nodes may lose their mapping mid-session. The signaling endpoint accepts `POST /signal` from an already-signaled node at any time; this updates that node's record and triggers a fresh `SignalReady` broadcast to its peers. Data-path keepalive is handled by QUIC (built-in `PING` frames) so an active cluster should never need re-signaling, but the endpoint supports it for crash-restart and long-idle cases.

Lifecycle state addition: `signaling` between `assigned` and `backend_joined`. Nodes enter `signaling` after startup, exit it after receiving `SignalReady` and completing B-2.

**Acceptance:** two nodes both receive `SignalReady` with each other's addresses before attempting to start the executor. Unauthenticated `/signal` POSTs are rejected. A node that never signals causes the cluster to fail with a named-node error after the configured timeout rather than hanging forever.

---

### B-2 — Node: transport negotiation module

**New package:** `node/src/transport/`

Responsible for: determining the local node's external address and establishing a direct UDP socket to each PP peer.

#### Sub-task B-2a — Port forwarding mode

If `--advertise-port` is set (added in Phase A, A-2) and differs from `--port`, the node is assumed to be port-forwarded. In this mode:

- External addr = `--advertise-host`, external port = `--advertise-port`
- No STUN required
- Node signals `transport_mode: "port_forward"` to coordinator
- Binds a UDP socket on the configured port and waits for peers to connect

This mode requires zero NAT traversal code. If both peers are port-forwarded, connection is trivial.

#### Sub-task B-2b — UDP hole punching mode

Used when `--advertise-port` is not set (default path).

1. **STUN discovery + NAT behavior probe**: use `pystun3` to discover external IP:port. In addition, query a *second* STUN server and compare the mapped port: if the two mapped ports differ, the local NAT is likely symmetric (address/port-dependent mapping) and hole punching will not work with a standard cone peer. In that case log a clear diagnostic up front and skip straight to the fallback path instead of attempting a doomed punch. This is a pragmatic substitute for full RFC 5780 behavior discovery and catches the common symmetric-NAT case deterministically.
2. **Send to `/signal`**: post the mapped address and detected NAT class.
3. **Wait for `SignalReady`**: receive all peers' external addresses from coordinator.
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

The executor calls `connect()` during startup. TURN (future) would be a third implementation of this interface.

**Acceptance:** two nodes on a home network (cone NAT) reach each other via UDP and complete a QUIC handshake. Log clearly which mode succeeded and, on failure, which NAT class was detected.

---

### B-3 — Custom vLLM PP executor

**New package:** `node/src/executor/`

Implements the integration identified in the spike (B-0) — either a `DistributedExecutorBackend` or a custom `torch.distributed` `ProcessGroup` backend, depending on what B-0 concludes. Uses `P2PTransport` (B-2) for all cross-node tensor passing.

Key responsibilities:
- On startup: call `transport.connect(peers)` before signaling `backend_joined`
- During a PP forward pass: call `transport.send_tensor` / `recv_tensor` with the activation tensor. The transport handles QUIC framing, serialization, and reliable delivery.
- Error handling: if a QUIC connection dies mid-inference, surface a clean error (do not hang). Connection loss must propagate as an exception out of `send_tensor`/`recv_tensor` within a bounded timeout.

**Tensor wire format.** Each activation is sent as a single framed message on a dedicated QUIC stream. Frame header (fixed 32 bytes, little-endian):

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | magic `0x41584f4e` ("AXON") |
| 4 | 2 | version (currently 1) |
| 6 | 2 | dtype enum (f16=1, bf16=2, f32=3, f8_e4m3=4, …) |
| 8 | 4 | rank (number of shape dims, max 8) |
| 12 | 8 | microbatch / sequence id |
| 20 | 4 | payload length in bytes |
| 24 | 8 | reserved / flags (compression, etc. — zero in v1 unless FP8-cast flag is set) |

Followed by `rank × 4` bytes of shape (uint32 per dim), followed by `payload length` bytes of raw tensor data in little-endian native dtype.

**FP8 on-wire casting (in scope for B-3, not deferred).** Per the bandwidth budget, prefill TTFT at 16K+ prompts is unusable without halving wire size. The transport MAY downcast BF16/FP16 activations to FP8 (e4m3) before serialization and re-upcast on receive, controlled by a per-cluster config flag (`--wire-dtype {native,fp8}`). The flag bit in the reserved header field records which was sent so the receiver upcasts correctly. Downcast/upcast is done on GPU using existing vLLM/PyTorch FP8 casting kernels — cost is ~1 kernel launch per tensor, negligible vs transit savings. Default: `fp8` for PP=2 across nodes, `native` for loopback testing.

**Chunked-prefill pipelining.** During prefill, each prompt chunk's activation is sent on its own QUIC stream (distinct `stream_id`) so chunk N's transit overlaps chunk N+1's compute on the sending stage. This is what makes prefill usable at realistic prompt lengths; without it the bandwidth budget assumes serial transit+compute and TTFT balloons. The executor is responsible for issuing chunks on distinct stream IDs; the transport just honors whatever `stream_id` is passed.

**Device handling.** Tensors are assumed to be on GPU at send time. The transport moves them to CPU (pinned memory, one pinned buffer per peer per stream to avoid allocation on the hot path) for framing, and reverses on recv. FP8 downcast happens on GPU *before* the device→host copy, so the CPU-side buffer is already the smaller size. If B-0 surfaces that GPU-direct paths (GPUDirect, `ibv_reg_mr`, CUDA IPC over sockets) are available and worthwhile over residential links, revisit — but for 50 Mbps uplinks the host-staging cost is dwarfed by transit time.

The executor/backend registers under the name `"axon_p2p"`. In `strategy.py:54`, replace the `"mp"` vs `"ray"` branch with:

```python
distributed_backend = (
    "mp" if stage_count == 1 else "axon_p2p"
)
requires_ray = False  # always — Ray removed from multi-node path
```

(If B-0 concludes the integration point is a `torch.distributed` backend rather than a vLLM executor, the string and registration mechanism will differ; update this section as part of B-0's output.)

**Acceptance:** PP forward pass completes end-to-end through the custom backend on two nodes connected via B-2 transport. QUIC connection loss during inference surfaces as an exception within 5s, not a hang.

---

### B-4 — Remove Ray from multi-node path

Once B-3 is working, clean up:

- `node/src/workers/ray_worker.py`: `join_ray_cluster` is no longer called for multi-node. Either delete or gate behind a flag.
- `node/src/runtime/lifecycle.py`: remove `join_ray_cluster` call from multi-node startup sequence.
- `coordinator/src/main.go`: `maybeStartRayHead` and `--autostart-ray-head` are no longer relevant for the P2P path. Keep them available for Phase A users (routable IP + Ray path) but document that they are bypassed when nodes negotiate `axon_p2p` backend.
- `coordinator/internal/server/server.go`: `ray_head_address` in `StartupConfig` can be omitted or marked deprecated for `axon_p2p` clusters.

Do not delete Ray support entirely — Phase A users (routable IP, Ray backend) may still want it.

**Gating condition (explicit).** The coordinator selects the backend per cluster at assignment time based on a single `backend` field on `StartupConfig`, with values `"ray"` or `"axon_p2p"`. The field is derived from cluster config — not negotiated per-node and not inferred from `execution_mode` — so there is exactly one source of truth and no ambiguity about which code path is live. All Ray-specific call sites (`join_ray_cluster`, `maybeStartRayHead`, `ray_head_address`) check `backend == "ray"` and are no-ops otherwise. Default for new clusters: `"axon_p2p"`. Phase A users opt in explicitly with `--backend ray`.

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
- **PP ≥ 3 across nodes**: each added PP stage adds one network hop per decode token. At residential RTTs this quickly pushes the decode-tok/s target out of reach. Revisit only after a multi-peer transport (B-2 already supports N peers, but hasn't been tested past 2) and after a latency-aware scheduler can hide inter-stage hops behind larger microbatch pipelining.
- **Further activation compression beyond FP8**: FP8-on-wire is in B-3 scope. Additional compression (INT4 activations, LZ4/zstd on top of FP8, perceptual-loss-tolerant codecs) is deferred until B-5 tells us whether FP8 alone clears the TTFT target. The compression flag bits in the wire header reserve space for this.

---

## Success criteria

Two nodes behind **home NAT** (cone), no third-party VPN, PP=2 across nodes, arbitrary TP within each node:
- Coordinator log: `inference_ready: true`  
- Both nodes: executor started via `axon_p2p` backend  
- PP activations logged as traveling via direct QUIC-over-UDP connection between nodes (not coordinator)  
- Within-node TP collectives confirmed to stay local (no AllReduce leaking over the P2P link)  
- Decode sustained ≥5 tok/s on the target model at target hardware; TTFT ≤10 s at 4 K prompt  
- Valid response from user service  
- Port forwarding path also validated separately (B-6)
