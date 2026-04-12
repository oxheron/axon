# Axon: Barebones Distributed vLLM Inference

This repo provides a minimal three-service setup for static, pipeline-parallel inference across multiple machines:

- `src/axon/coordinator.py`: node registration + startup broadcast
- `src/axon/node_agent.py`: node VRAM detection + Ray join + optional local vLLM worker launch
- `src/axon/server.py`: OpenAI-compatible `/v1/chat/completions` API on coordinator

Docker assets:

- `docker/coordinator.Dockerfile`
- `docker/node-agent.Dockerfile`
- `docker/server.Dockerfile`
- `scripts/test_12gb_gpu.sh` (single-GPU smoke test)

## Assumptions

- One GPU per machine
- All machines have the same model available locally
- Static cluster membership (no dynamic resize)
- Ray + vLLM installed on each machine

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Start Coordinator (on head node)

```bash
PYTHONPATH=src python -m axon.coordinator \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --autostart-ray-head
```

Useful endpoints:

- `GET /status`
- `GET /config` (available after startup)
- `POST /register` (used by node agents)

## 2) Start Node Agent (on each worker node)

```bash
PYTHONPATH=src python -m axon.node_agent \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --host 0.0.0.0 \
  --port 9000 \
  --advertise-host <THIS_NODE_REACHABLE_IP> \
  --vllm-gpu-memory-utilization 0.72 \
  --vllm-max-model-len 1024 \
  --vllm-dtype float16
```

When enough nodes register, the coordinator broadcasts `/startup`. The node then:

1. Joins Ray cluster at coordinator-provided head address
2. Starts a local vLLM process using the coordinator's `pipeline_parallel_size`

If you only want Ray join (and no local vLLM API on workers), pass `--no-vllm-worker`.

## 3) Start Inference Server (on coordinator node)

```bash
PYTHONPATH=src python -m axon.server \
  --coordinator-url http://127.0.0.1:8000 \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.72 \
  --max-model-len 1024 \
  --dtype float16
```

The server waits for coordinator readiness, then initializes `AsyncLLMEngine` with:

- `pipeline_parallel_size = number_of_registered_nodes`
- `distributed_executor_backend = "ray"`

## Test Inference

```bash
curl http://<COORDINATOR_IP>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

Streaming mode:

```bash
curl http://<COORDINATOR_IP>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Docker (per-service images)

Build images:

```bash
docker build -f docker/coordinator.Dockerfile -t axon-coordinator:test .
docker build -f docker/node-agent.Dockerfile -t axon-node-agent:test .
docker build -f docker/server.Dockerfile -t axon-server:test .
```

## 10-12GB GPU smoke test script

This script is designed for a single machine with a max ~16GB GPU and runs a 10-12GB friendly test profile by default:

```bash
./scripts/test_12gb_gpu.sh
```

Optional environment variables:

- `MODEL` (default: `Qwen/Qwen2.5-3B-Instruct`)
- `COORD_PORT` (default: `8000`)
- `SERVER_PORT` (default: `8080`)
- `TIMEOUT_SEC` (default: `420`)
- `SERVER_GPU_MEMORY_UTILIZATION` (default: `0.72`)
- `SERVER_MAX_MODEL_LEN` (default: `1024`)
- `SERVER_DTYPE` (default: `float16`)
- `KEEP_CONTAINERS=1` to keep stack running after the test
