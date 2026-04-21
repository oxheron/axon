---
name: Phase B — P2P PP transport (NAT traversal)
overview: Replace Ray as the multi-node PP transport with a custom vLLM executor backend that sends tensors over direct P2P connections. Coordinator handles signaling only. Supports three connection modes in order of preference: port forwarding (manual), UDP hole punching (automatic cone NAT), TURN relay (fallback, deferred).
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

TURN relay is explicitly deferred. The Phase B implementation ships with the first two modes. The executor interface must be designed so TURN can be added as a third transport option without restructuring.

---

## Spike (do this first — B-0)

**Before writing any executor code**, spend 1–2 days reading vLLM source to answer:

1. What interface does a custom `--distributed-executor-backend` class need to implement? Read `vllm/executor/` base classes and how `ray` backend is registered.
2. Where exactly do PP tensors cross node boundaries? Trace a PP forward pass: `vllm/worker/worker.py` → distributed send/recv ops → what sits below (Ray actor call, NCCL, custom socket?).
3. Can the custom executor fully own the tensor transport, or does NCCL get invoked below the executor interface in a way that bypasses it?
4. Is a vLLM fork required, or is executor plugin registration sufficient?

**Output:** a short written decision doc (can live in this file as an appendix) answering the four questions above and confirming or adjusting the implementation plan. Do not start B-1 through B-4 until the spike is done.

---

## Todos

### B-0 — Spike: vLLM executor interface + PP tensor path

As described above. Written output required before proceeding.

---

### B-1 — Coordinator: signaling endpoint

**File:** `coordinator/internal/server/` (new handler + types)

Add a `/signal` endpoint that nodes use to exchange transport addresses after receiving their startup config. The flow:

1. Node receives `StartupConfig` from coordinator (existing flow).
2. Node determines its external address (via STUN or manual config — see B-2).
3. Node POSTs `{ node_id, external_addr, external_port, transport_mode }` to `/signal`.
4. Coordinator stores signal records keyed by `cluster_id + node_id`.
5. When all expected nodes have signaled, coordinator broadcasts a `SignalReady` message to each node containing the full peer signal table.

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

Lifecycle state addition: `signaling` between `assigned` and `backend_joined`. Nodes enter `signaling` after startup, exit it after receiving `SignalReady` and completing B-2.

**Acceptance:** two nodes both receive `SignalReady` with each other's addresses before attempting to start the executor.

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

1. **STUN discovery**: use `pystun3` (or equivalent) to discover external IP:port. Send to `/signal`.
2. **Wait for `SignalReady`**: receive all peers' external addresses from coordinator.
3. **Simultaneous open**: for each PP peer, send UDP packets to their external addr:port while listening on the local port. Most cone NATs will punch through within a few hundred milliseconds. Retry for up to ~5 seconds before declaring failure.
4. **Fallback signal**: if hole punch fails, log a clear message directing the user to use `--advertise-port` with a manually forwarded port (or wait for TURN in a future release).

Libraries: `pystun3` for STUN. No additional dependencies needed for the hole punch itself — it's raw UDP.

#### Common interface

Both modes expose the same interface to the executor (B-3):

```python
class P2PTransport:
    async def connect(self, peers: list[PeerEndpoint]) -> None: ...
    async def send_tensor(self, peer_id: str, tensor: bytes) -> None: ...
    async def recv_tensor(self, peer_id: str) -> bytes: ...
    async def close(self) -> None: ...
```

The executor calls `connect()` during startup. TURN (future) would be a third implementation of this interface.

**Acceptance:** two nodes on a home network (cone NAT) reach each other via UDP. Log clearly which mode succeeded.

---

### B-3 — Custom vLLM PP executor

**New package:** `node/src/executor/`

Implements the `DistributedExecutorBackend` interface identified in the spike (B-0). Uses `P2PTransport` (B-2) for all cross-node tensor passing.

Key responsibilities:
- On startup: call `transport.connect(peers)` before signaling `backend_joined`
- During a PP forward pass: serialize outgoing activation tensors, send via transport, deserialize incoming
- Error handling: if a peer socket dies mid-inference, surface a clean error (do not hang)

Tensor wire format: length-prefixed raw bytes. No compression initially — keep it simple and revisit if latency is a problem.

The executor registers under the name `"axon_p2p"`. In `strategy.py:54`, replace the `"mp"` vs `"ray"` branch with:

```python
distributed_backend = (
    "mp" if stage_count == 1 else "axon_p2p"
)
requires_ray = False  # always — Ray removed from multi-node path
```

**Acceptance:** PP forward pass completes end-to-end through the custom executor on two nodes connected via B-2 transport.

---

### B-4 — Remove Ray from multi-node path

Once B-3 is working, clean up:

- `node/src/workers/ray_worker.py`: `join_ray_cluster` is no longer called for multi-node. Either delete or gate behind a flag.
- `node/src/runtime/lifecycle.py`: remove `join_ray_cluster` call from multi-node startup sequence.
- `coordinator/src/main.go`: `maybeStartRayHead` and `--autostart-ray-head` are no longer relevant for the P2P path. Keep them available for Phase A users (routable IP + Ray path) but document that they are bypassed when nodes negotiate `axon_p2p` backend.
- `coordinator/internal/server/server.go`: `ray_head_address` in `StartupConfig` can be omitted or marked deprecated for `axon_p2p` clusters.

Do not delete Ray support entirely — Phase A users (routable IP, Ray backend) may still want it. Gate on `execution_mode` or backend negotiation.

---

### B-5 — Integration test: two-node NAT (hole punch path)

Run Phase B end-to-end on two machines behind real home NAT (or a simulated cone NAT environment):

- Both nodes start with no `--advertise-port`
- Coordinator starts, nodes register, signaling completes, hole punch succeeds
- Executor starts, PP forward pass completes
- `inference_ready: true`, curl returns a valid response

**Acceptance criteria:** same as Phase A smoke test, but on NAT nodes with no manual port forwarding.

---

### B-6 — Integration test: port forwarding path

Same as B-5 but both nodes use `--advertise-port` with manually forwarded ports.

---

## Explicitly deferred (future milestone)

- **TURN relay**: third transport mode in `P2PTransport`. Coordinator or a dedicated TURN server relays packets when hole punch fails (symmetric NAT). Interface is already designed to accept it (B-2). Implement after B-5 validates the core path.
- **Encryption**: P2P sockets are plaintext in Phase B. Add DTLS or a session key (exchanged via coordinator signaling) before any production/untrusted-peer deployment.
- **TLS on coordinator + node HTTP**: same deferral as Phase A runbook noted.

---

## Success criteria

Two nodes behind **home NAT** (cone), no third-party VPN:
- Coordinator log: `inference_ready: true`  
- Both nodes: executor started via `axon_p2p` backend  
- PP activations logged as traveling via direct UDP socket (not coordinator)  
- Valid response from user service  
- Port forwarding path also validated separately (B-6)
