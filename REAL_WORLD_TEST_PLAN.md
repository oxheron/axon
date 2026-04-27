# Real-World Test Plan

Two machines required for both paths. Each path is tested independently — pick one path per test run. The coordinator can run on either machine or a third machine (a VPS works well).

---

## Common Setup (Both Paths)

### Machines
- **Machine A** — node 0 (first worker, PP stage 0)
- **Machine B** — node 1 (second worker, PP stage 1)
- **Coordinator** — can be Machine A, Machine B, or a separate VPS

### Software prerequisites
```bash
# On each node machine
cd axon/node
pip install -e .

# On coordinator machine
cd axon/coordinator
go build ./src
```

### Execution modes

| Mode | What it does | GPU required |
|------|--------------|--------------|
| `dry_run` | Validates signaling and QUIC transport without loading a model | No |
| `vllm_slice` | Runs real inference with vLLM across both nodes (PP=2) | Yes |

Start with `dry_run` to confirm connectivity, then switch to `vllm_slice` when you're ready to run inference.

---

## Path 1: Port Forwarding

### What it tests
Direct QUIC connection over UDP after manual port forwarding. No NAT traversal — each node connects directly to the other's publicly advertised address.

### Network setup required

**On Machine A's router/firewall:**
- Forward UDP port `9001` (external) → Machine A LAN IP, port `9000` (internal)
- Allow inbound UDP on port 9001

**On Machine B's router/firewall:**
- Forward UDP port `9001` (external) → Machine B LAN IP, port `9000` (internal)
- Allow inbound UDP on port 9001

You can use any external port; 9001 is just an example. If running in a cloud environment (EC2, GCE, etc.) add an inbound UDP rule to the security group instead of router port forwarding.

Find your external IP:
```bash
curl -s https://api.ipify.org
```

### Dry-run (connectivity test, no GPU needed)

**Step 1 — Start coordinator**
```bash
./coordinator \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --execution-mode dry_run
```

**Step 2 — Start node on Machine A**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --advertise-host <MACHINE_A_EXTERNAL_IP> \
  --advertise-port 9001 \
  --no-vllm-worker
```

**Step 3 — Start node on Machine B**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --advertise-host <MACHINE_B_EXTERNAL_IP> \
  --advertise-port 9001 \
  --no-vllm-worker
```

Because `--advertise-port` (9001) differs from `--bind-port` (9000), the transport mode is automatically set to `port_forward`.

**What to observe:**
1. Coordinator logs — both nodes connect via WebSocket, coordinator broadcasts `signal_ready`
2. Node logs — transport mode reported as `port_forward`; no STUN probe output
3. Node logs — QUIC handshake completes between Machine A and Machine B
4. Node logs — `dry_run` execution completes without errors

### Inference (vllm_slice, GPU required)

**Step 1 — Start coordinator**
```bash
./coordinator \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --execution-mode vllm_slice
```

**Step 2 — Start node on Machine A**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --advertise-host <MACHINE_A_EXTERNAL_IP> \
  --advertise-port 9001 \
  --vllm-gpu-memory-utilization 0.85 \
  --vllm-max-model-len 4096
```

**Step 3 — Start node on Machine B**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --advertise-host <MACHINE_B_EXTERNAL_IP> \
  --advertise-port 9001 \
  --vllm-gpu-memory-utilization 0.85 \
  --vllm-max-model-len 4096
```

**Step 4 — Wait for inference-ready, then send a request**
```bash
# Poll coordinator until inference_ready: true
curl http://<COORDINATOR_IP>:8000/status

# Send a completion request through the coordinator proxy
curl http://<COORDINATOR_IP>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 32
  }'
```

**What to observe:**
1. Node logs — QUIC handshake completes; transport mode `port_forward`
2. Node logs — vLLM loads model weights across both nodes (this takes a few minutes)
3. Coordinator status — `inference_ready: true` once both nodes report `pipeline_ready`
4. Completion response — tokens generated and returned through the coordinator proxy

---

## Path 2: UDP Hole Punching

### What it tests
Automatic NAT traversal via simultaneous UDP open. Both nodes use STUN to discover their external address, exchange that info through the coordinator, then punch through their respective NATs.

### Network setup required

**No port forwarding needed.** The only requirement is:
- Both nodes have **non-symmetric (cone) NAT** — most home/office routers qualify
- Inbound UDP is not blocked by a strict firewall (no stateful block of unsolicited UDP)
- Both nodes can reach the coordinator and Google STUN servers (`stun.l.google.com:19302`)

To verify your NAT type before running:
```bash
python -c "
import stun
nat_type, ext_ip, ext_port = stun.get_ip_info()
print(f'NAT type: {nat_type}')
print(f'External: {ext_ip}:{ext_port}')
"
```
If `nat_type` is `Symmetric NAT`, hole punching will not work — use Path 1 instead.

### Dry-run (connectivity test, no GPU needed)

**Step 1 — Start coordinator**
```bash
./coordinator \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --execution-mode dry_run
```

**Step 2 — Start node on Machine A**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --no-vllm-worker
```

**Step 3 — Start node on Machine B**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --no-vllm-worker
```

Because `--advertise-port` is omitted (defaults to `--bind-port`), the transport mode is automatically set to `hole_punch`.

**What to observe:**
1. Node logs (early) — STUN probe output: external IP and port discovered, NAT type reported
2. Coordinator logs — both nodes signal with their STUN-discovered external addresses
3. Node logs — `AXON-PUNCH` send/receive loops start for each peer
4. Node logs — hole punch succeeds within 30s; QUIC handshake completes
5. Node logs — `dry_run` execution completes without errors

### Inference (vllm_slice, GPU required)

**Step 1 — Start coordinator**
```bash
./coordinator \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --execution-mode vllm_slice
```

**Step 2 — Start node on Machine A**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --vllm-gpu-memory-utilization 0.85 \
  --vllm-max-model-len 4096
```

**Step 3 — Start node on Machine B**
```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --bind-host 0.0.0.0 \
  --bind-port 9000 \
  --vllm-gpu-memory-utilization 0.85 \
  --vllm-max-model-len 4096
```

**Step 4 — Wait for inference-ready, then send a request**
```bash
# Poll coordinator until inference_ready: true
curl http://<COORDINATOR_IP>:8000/status

# Send a completion request through the coordinator proxy
curl http://<COORDINATOR_IP>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 32
  }'
```

**What to observe:**
1. Node logs (early) — STUN probe, NAT type detection
2. Node logs — hole punch succeeds; QUIC handshake completes
3. Node logs — vLLM loads model weights across both nodes
4. Coordinator status — `inference_ready: true`
5. Completion response — tokens returned through coordinator proxy

---

## Troubleshooting

### Connectivity issues

| Symptom | Likely cause |
|---------|--------------|
| Node cannot reach coordinator | Coordinator port 8000 not reachable; wrong coordinator IP |
| QUIC connection timeout (port forward) | Port forward not applied, or wrong external IP |
| One node connects but the other doesn't | Asymmetric port forward — check both routers |
| STUN discovery fails | No internet access or STUN servers unreachable |
| "Symmetric NAT detected" | Hole punching will fail; switch to port forward path |
| Punch timeout (30s) | NAT drops unsolicited UDP; try port forwarding instead |
| QUIC handshake fails after punch | Check logs for TLS/ALPN error |
| Nodes see each other on LAN but fail externally | Both nodes behind same NAT; use LAN IPs directly |

### Inference issues

| Symptom | Likely cause |
|---------|--------------|
| vLLM fails to start | GPU driver issue; check `CUDA_VISIBLE_DEVICES` and driver version |
| OOM during model load | Reduce `--vllm-gpu-memory-utilization` or use a smaller/quantized model |
| `pipeline_ready` never reached | Check node logs for vLLM health check failure |
| Slow TTFT at long prompts | Expected on residential links; FP8 wire casting is on by default (`AXON_WIRE_DTYPE=fp8`) |
| Coordinator proxy returns 503 | `inference_ready` not yet true; poll `/status` and wait |

---

## Quick Reference

### Flag summary

| Flag | Required for | Default |
|------|-------------|---------|
| `--coordinator-url` | All modes | — |
| `--bind-port` | All modes | 9000 |
| `--advertise-host` | Port forward | auto-detected LAN IP |
| `--advertise-port` | Port forward (must differ from bind-port) | same as bind-port |
| `--no-vllm-worker` | dry_run | off |
| `--vllm-gpu-memory-utilization` | vllm_slice | 0.72 |
| `--vllm-max-model-len` | vllm_slice | 1024 |
| `--vllm-dtype` | vllm_slice | float16 |
| `--cluster-token` | When coordinator sets `--cluster-token` | — |

### Path comparison

| | Port Forwarding | UDP Hole Punching |
|--|----------------|-------------------|
| Router config needed | Yes — UDP port forward on each machine | No |
| Works with symmetric NAT | Yes | No |
| Transport mode in logs | `port_forward` | `hole_punch` |
| STUN probes run | No | Yes |
| Extra `--advertise-port` flag | Required (must differ from bind-port) | Must be omitted |
| Punch timeout applies | No | Yes (30s) |
