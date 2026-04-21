---
name: Phase A — Routable IP validation
overview: Prove pipeline-parallel inference works across two machines with publicly routable IPs (cloud nodes, lab network, or manually port-forwarded hosts). No NAT traversal required. Mostly a runbook + two small code changes.
status: pending
---

# Phase A — Routable IP validation

Goal: get two GPU nodes on reachable IPs to reach `inference_ready` with both PP stages active. No new transport work — just correct configuration and the small gaps that make WAN configuration error-prone today.

## What already works

The codebase supports WAN today if every address is routable. The two flags that matter:

- `--advertise-host` on each node (default: `detect_local_ip()` → RFC1918 private IP, **wrong on WAN**)
- `--ray-head-address` on the coordinator (default: `detectLocalIP():6379` → same problem)

On a LAN both defaults work. On the internet, both must be set explicitly.

---

## Todos

### A-1 — Coordinator: pass WAN flags to `maybeStartRayHead`

**File:** `coordinator/src/main.go:29` — `maybeStartRayHead`

Today `ray start --head` is called with only `--port`. Ray infers its own node IP from the local interface, which is private on a cloud VM unless explicitly set.

Add `--node-ip-address` and optionally `--min-worker-port` / `--max-worker-port` to the `ray start --head` command when `--autostart-ray-head` is used:

```
ray start --head
  --port <ray-port>
  --node-ip-address <public-ip>   ← new: taken from --ray-node-ip flag or detectLocalIP()
  --disable-usage-stats
```

Add a new CLI flag `--ray-node-ip` (default: same as `detectLocalIP()`) so operators can override without touching `--ray-head-address`. This keeps the two concerns separate: where the coordinator advertises the head vs. what IP Ray binds to.

**Acceptance:** `ray start --head` with `--node-ip-address` set to the public IP; remote workers can join across the internet.

---

### A-2 — Node: add `--advertise-port` flag

**File:** `node/src/cli.py`

Today `advertise_host` controls the IP/hostname in callback and worker URLs, but the port is always the bind port. On a port-forwarded or cloud host where the external port differs from the bind port, there is no way to set an external port independently.

Add `--advertise-port` (default: same as `--port`). Use `advertise_port` instead of `bind_port` when constructing the `CallbackURL` sent to the coordinator.

This is a small change now that also unblocks the Phase B port-forwarding transport mode.

**Acceptance:** node started with `--port 9000 --advertise-port 19000` registers with `callback_url` containing port `19000`.

---

### A-3 — Write the WAN runbook

**File:** `docs/wan-routable.md` (new, or a section in README)

Document the exact flags needed for a two-node cloud setup:

```
# Node A (Ray head host)
coordinator:
  --ray-head-address <node-a-public-ip>:6379
  --ray-node-ip <node-a-public-ip>
  --autostart-ray-head

node (on Node A):
  --advertise-host <node-a-public-ip>

# Node B
node (on Node B):
  --advertise-host <node-b-public-ip>
```

Include:
- Firewall / security group rules required (Ray port 6379, Ray worker range 10000–19999 by default, node API port 9000, vLLM worker port 8100)
- How to verify each step: coordinator logs show both nodes registered; node logs show `ray join completed`; coordinator reaches `inference_ready`
- What breaks silently vs. loudly when addresses are wrong

---

### A-4 — Two-node smoke test on cloud

Run the setup from A-3 on two cloud VMs with public IPs (same region is fine). Both PP stages must reach `pipeline_ready`; coordinator must reach `inference_ready`; a curl to the user service must return a model response.

This is the Phase A success gate. No code changes — just validation that the runbook is correct.

**Acceptance criteria:**
- Coordinator log shows `inference_ready: true`
- Both node logs show `[vllm] ... pipeline_ready`
- `curl .../v1/chat/completions` returns a valid response

---

## What Phase A does NOT solve

- NAT traversal for home networks — that is Phase B
- TLS — deferred, document as "lab only" in the runbook
- 409 registration races on high-latency WAN — low priority, can be retried manually
- Symmetric NAT — Phase B, TURN relay fallback
