# WAN Setup — Routable IP (Phase A)

This guide covers running a two-node pipeline-parallel cluster where both machines have
publicly routable IPs (cloud VMs, lab network, or manually port-forwarded hosts).
No NAT traversal is required.

> **Security note:** This setup has no TLS and no authentication. Use only in a trusted
> lab environment or with firewall rules that restrict access to known IPs.

---

## Required open ports

| Port / range   | Direction        | Purpose                          |
|----------------|------------------|----------------------------------|
| 6379           | Node B → Node A  | Ray GCS (head node)              |
| 10000–19999    | Node B → Node A  | Ray worker-to-worker traffic     |
| 9000           | Coordinator → both nodes | Node API callbacks      |
| 8100           | Coordinator → both nodes | vLLM worker health check |
| 8000           | Nodes → Coordinator | Node registration/status      |

---

## Launch sequence

### Node A (Ray head host)

```bash
# 1. Start the coordinator — Ray head auto-starts here
coordinator \
  --model-name <model-name-or-path> \
  --min-nodes 2 \
  --autostart-ray-head \
  --ray-port 6379 \
  --ray-node-ip <node-a-public-ip> \
  --ray-head-address <node-a-public-ip>:6379

# 2. Start the node agent on Node A
python -m cli \
  --coordinator-url http://<coordinator-ip>:8000 \
  --advertise-host <node-a-public-ip>
```

### Node B

```bash
python -m cli \
  --coordinator-url http://<coordinator-ip>:8000 \
  --advertise-host <node-b-public-ip>
```

If a node's external port differs from its bind port (e.g. port-forwarded host):

```bash
python -m cli \
  --coordinator-url http://<coordinator-ip>:8000 \
  --port 9000 \
  --advertise-port 19000 \
  --advertise-host <node-public-ip>
```

---

## Verification steps

### 1. Both nodes registered

Coordinator log should show:

```
registered node <id-a> ...
registered node <id-b> ...
```

Or query directly:

```bash
curl http://<coordinator-ip>:8000/nodes
```

Both nodes should appear with `lifecycle_state: registered`.

### 2. Ray join completed

Each node log should include:

```
ray join completed
```

If a node stays at `joining_ray`, check that port 6379 and the 10000–19999 range are
open from Node B to Node A.

### 3. Pipeline ready

Node logs should show:

```
[vllm] ... pipeline_ready
```

Coordinator log should reach:

```
inference_ready: true
```

### 4. Smoke test

```bash
curl http://<coordinator-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-name>","messages":[{"role":"user","content":"hello"}]}'
```

---

## What breaks silently vs. loudly

| Symptom | Likely cause |
|---------|-------------|
| Node stuck at `joining_ray` | Ray port 6379 or worker range blocked; `--ray-node-ip` set to private IP on head |
| Coordinator never sends startup to node | `--advertise-host` is a private IP the coordinator can't reach |
| vLLM worker unreachable | Port 8100 blocked, or `--advertise-host` wrong |
| Node registers, then coordinator retries startup | Callback URL uses wrong port; check `--advertise-port` |
| `409` on registration | A prior registration already triggered startup; restart coordinator and retry |

---

## What this does NOT solve

- **NAT traversal** for home networks — Phase B
- **TLS / authentication** — deferred, lab-only for now
- **Symmetric NAT** — Phase B TURN relay fallback
