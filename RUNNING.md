# Running Axon — Distributed Inference Guide

## Overview

Axon runs three cooperating services. Each has its own run script so they can be deployed on separate machines:

| Script | Service | Default port |
|---|---|---|
| `scripts/run_coordinator.sh` | Coordinator (Go) | 8000 |
| `scripts/run_node.sh` | Node agent + vLLM worker (Python) | 9000 (control), 8100 (vLLM) |
| `scripts/run_user.sh` | User / API proxy (Python) | 8080 |

**Launch order:** coordinator first → nodes → user service (nodes and user can start in either order after the coordinator is up).

---

## Prerequisites

- **Coordinator:** Go 1.22+
- **Node:** Python 3.10+, `pip install -r requirements.txt`, a CUDA or ROCm GPU
- **User service:** Python 3.10+, `pip install -r requirements.txt`

---

## Port-Forwarding / Firewall Rules

### Coordinator machine

| Port | Direction | Who connects | Purpose |
|---|---|---|---|
| 8000 | Inbound | Nodes, user service, clients | HTTP control plane + inference proxy |
| 6379 | Inbound | Nodes | Ray GCS (multi-node Ray mode only) |
| 10000–19999 | Inbound + Outbound | Nodes | Ray inter-node tensor passing (multi-node only) |

### Node machines (one set of rules per node)

| Port | Direction | Who connects | Purpose |
|---|---|---|---|
| 9000 | Inbound | Coordinator | Node control API (`/startup` callback) |
| 8100 | Inbound | Coordinator | vLLM worker API (inference proxied here) |
| 10000–19999 | Inbound + Outbound | Other nodes, coordinator | Ray worker traffic (multi-node only) |

### User service machine

No port-forwarding required. The user service only makes outbound connections to the coordinator and is typically run on localhost or a LAN machine alongside the client. Port 8080 only needs to be opened if clients are on a different machine from where the user service is running.

---

## Single-node (one machine, one GPU)

```bash
# Terminal 1 — coordinator
MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_coordinator.sh

# Terminal 2 — node
COORDINATOR_URL=http://localhost:8000 bash scripts/run_node.sh

# Terminal 3 — user service
COORDINATOR_URL=http://localhost:8000 bash scripts/run_user.sh
```

Wait for the node log to show `pipeline_ready`. Then test:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'
```

---

## Multi-node (pipeline parallel, separate machines)

Replace `<COORD_IP>`, `<NODE_A_IP>`, `<NODE_B_IP>` with real IPs.

**Coordinator machine:**

```bash
MODEL=meta-llama/Llama-3-70b-Instruct \
MIN_NODES=2 \
RAY_NODE_IP=<COORD_IP> \
AUTOSTART_RAY_HEAD=1 \
bash scripts/run_coordinator.sh
```

**Node A (first machine with a GPU):**

```bash
COORDINATOR_URL=http://<COORD_IP>:8000 \
ADVERTISE_HOST=<NODE_A_IP> \
NODE_ID=node-a \
bash scripts/run_node.sh
```

**Node B (second machine with a GPU):**

```bash
COORDINATOR_URL=http://<COORD_IP>:8000 \
ADVERTISE_HOST=<NODE_B_IP> \
NODE_ID=node-b \
bash scripts/run_node.sh
```

**User service (any machine):**

```bash
COORDINATOR_URL=http://<COORD_IP>:8000 \
bash scripts/run_user.sh
```

The coordinator waits for both nodes to register before broadcasting startup. Once both nodes reach `pipeline_ready`, the user service becomes live.

---

## All environment variables

### run_coordinator.sh

| Variable | Default | Description |
|---|---|---|
| `MODEL` | *(required)* | Model name/path (e.g. `Qwen/Qwen2.5-3B-Instruct`) |
| `COORD_PORT` | `8000` | HTTP bind port |
| `MIN_NODES` | `1` | Node count required before startup |
| `EXECUTION_MODE` | auto | `single_node` \| `vllm_ray_pipeline` \| `slice_loaded_pipeline` \| `dry_run` |
| `AUTOSTART_RAY_HEAD` | `1` when `MIN_NODES>1` | Run `ray start --head` before serving |
| `RAY_PORT` | `6379` | Ray GCS port |
| `RAY_NODE_IP` | auto | IP passed to `ray start --node-ip-address`. Set to coordinator's public/LAN IP on multi-machine setups. |
| `LOG_FILE` | `coordinator.log` | Log path. Set to `""` to write to stdout. |

### run_node.sh

| Variable | Default | Description |
|---|---|---|
| `COORDINATOR_URL` | *(required)* | Coordinator URL (e.g. `http://1.2.3.4:8000`) |
| `NODE_PORT` | `9000` | Node control API port |
| `VLLM_PORT` | `8100` | vLLM worker subprocess port |
| `ADVERTISE_HOST` | auto | IP this node tells the coordinator. Must be reachable from the coordinator. |
| `ADVERTISE_PORT` | `NODE_PORT` | Advertised port (override for NAT setups) |
| `NODE_ID` | `<hostname>-<uuid8>` | Unique node identifier |
| `VLLM_GPU_MEM_UTIL` | `0.72` | `--gpu-memory-utilization` |
| `VLLM_MAX_MODEL_LEN` | `1024` | `--max-model-len` |
| `VLLM_DTYPE` | `float16` | `--dtype` |
| `NO_VLLM` | `0` | Set to `1` to skip vLLM launch (Ray join only) |
| `LOG_FILE` | `node.log` | Log path. Set to `""` for stdout. |

### run_user.sh

| Variable | Default | Description |
|---|---|---|
| `COORDINATOR_URL` | *(required)* | Coordinator URL |
| `SERVER_PORT` | `8080` | API bind port |
| `LOG_FILE` | `user.log` | Log path. Set to `""` for stdout. |

---

## Checking status

```bash
# Coordinator cluster state
curl -s http://<COORD_IP>:8000/status | python3 -m json.tool

# Node internal state
curl -s http://<NODE_IP>:9000/status | python3 -m json.tool

# User service readiness
curl -s http://<USER_IP>:8080/healthz | python3 -m json.tool
```

## Sending inference

The user service exposes a standard OpenAI-compatible API on port 8080:

```bash
# Chat completion
curl http://<USER_IP>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'

# Streaming
curl http://<USER_IP>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": true
  }'
```

Or use any OpenAI SDK by setting `base_url=http://<USER_IP>:8080/v1` and `api_key=ignored`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://<USER_IP>:8080/v1", api_key="ignored")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```
