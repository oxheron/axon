#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
COORD_PORT="${COORD_PORT:-8000}"
SERVER_PORT="${SERVER_PORT:-8080}"
NETWORK_NAME="${NETWORK_NAME:-axon-test-net}"
TIMEOUT_SEC="${TIMEOUT_SEC:-420}"
KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.72}"
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-1024}"
SERVER_DTYPE="${SERVER_DTYPE:-float16}"

COORD_IMAGE="${COORD_IMAGE:-axon-coordinator:test}"
NODE_IMAGE="${NODE_IMAGE:-axon-node-agent:test}"
SERVER_IMAGE="${SERVER_IMAGE:-axon-server:test}"

COORD_CONTAINER="axon-coordinator"
NODE_CONTAINER="axon-node"
SERVER_CONTAINER="axon-server"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

print_logs() {
  echo ""
  echo "===== coordinator logs ====="
  docker logs --tail 120 "$COORD_CONTAINER" 2>/dev/null || true
  echo ""
  echo "===== node logs ====="
  docker logs --tail 120 "$NODE_CONTAINER" 2>/dev/null || true
  echo ""
  echo "===== server logs ====="
  docker logs --tail 120 "$SERVER_CONTAINER" 2>/dev/null || true
}

cleanup() {
  if [[ "$KEEP_CONTAINERS" == "1" ]]; then
    echo "KEEP_CONTAINERS=1, leaving containers running."
    return
  fi
  docker rm -f "$SERVER_CONTAINER" "$NODE_CONTAINER" "$COORD_CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
}

on_error() {
  echo ""
  echo "Test failed. Showing recent logs..."
  print_logs
}

trap cleanup EXIT
trap on_error ERR

require_cmd docker
require_cmd curl
require_cmd python3
require_cmd nvidia-smi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable." >&2
  exit 1
fi

GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ -z "${GPU_MEM_MB}" ]]; then
  echo "Could not read GPU memory from nvidia-smi." >&2
  exit 1
fi

echo "Detected GPU memory: ${GPU_MEM_MB} MiB"
if (( GPU_MEM_MB > 17000 )); then
  echo "Note: GPU is larger than 16GB. Script still runs a 10-12GB-friendly model."
fi
if (( GPU_MEM_MB < 10000 )); then
  echo "Warning: GPU has <10GB VRAM. Model may still work, but OOM risk is higher."
fi

echo "Cleaning previous test stack..."
docker rm -f "$SERVER_CONTAINER" "$NODE_CONTAINER" "$COORD_CONTAINER" >/dev/null 2>&1 || true
docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
docker network create "$NETWORK_NAME" >/dev/null

echo "Building coordinator image..."
docker build -f docker/coordinator.Dockerfile -t "$COORD_IMAGE" .
echo "Building node image..."
docker build -f docker/node-agent.Dockerfile -t "$NODE_IMAGE" .
echo "Building server image..."
docker build -f docker/server.Dockerfile -t "$SERVER_IMAGE" .

echo "Starting coordinator..."
docker run -d \
  --name "$COORD_CONTAINER" \
  --network "$NETWORK_NAME" \
  -p "${COORD_PORT}:8000" \
  "$COORD_IMAGE" \
  --host 0.0.0.0 \
  --port 8000 \
  --min-nodes 1 \
  --model-name "$MODEL" \
  --ray-port 6379 \
  --ray-head-address "${COORD_CONTAINER}:6379" \
  --autostart-ray-head >/dev/null

echo "Starting node agent..."
docker run -d \
  --name "$NODE_CONTAINER" \
  --network "$NETWORK_NAME" \
  "$NODE_IMAGE" \
  --coordinator-url "http://${COORD_CONTAINER}:8000" \
  --host 0.0.0.0 \
  --port 9000 \
  --advertise-host "$NODE_CONTAINER" \
  --no-vllm-worker >/dev/null

echo "Starting inference server..."
docker run -d \
  --name "$SERVER_CONTAINER" \
  --network "$NETWORK_NAME" \
  --gpus all \
  -p "${SERVER_PORT}:8080" \
  "$SERVER_IMAGE" \
  --coordinator-url "http://${COORD_CONTAINER}:8000" \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$SERVER_MAX_MODEL_LEN" \
  --dtype "$SERVER_DTYPE" >/dev/null

echo "Waiting for inference server readiness..."
start_ts="$(date +%s)"
while true; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if (( elapsed > TIMEOUT_SEC )); then
    echo "Timed out waiting for server readiness after ${TIMEOUT_SEC}s" >&2
    exit 1
  fi

  health_json="$(curl -fsS "http://127.0.0.1:${SERVER_PORT}/healthz" 2>/dev/null || true)"
  if [[ -n "$health_json" ]]; then
    if python3 -c 'import json,sys; raise SystemExit(0 if json.loads(sys.stdin.read()).get("ok") else 1)' <<<"$health_json"; then
      echo "Server is ready."
      break
    fi
  fi
  sleep 5
done

echo "Sending smoke inference request..."
response="$(
  curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: axon\"}],
      \"max_tokens\": 32
    }"
)"

assistant_msg="$(python3 -c 'import json,sys; data=json.loads(sys.stdin.read()); print(data["choices"][0]["message"]["content"].strip())' <<<"$response")"
echo "Model response: ${assistant_msg}"

if [[ -z "$assistant_msg" ]]; then
  echo "Empty assistant response." >&2
  exit 1
fi

echo ""
echo "Smoke test passed."
echo "Coordinator API: http://127.0.0.1:${COORD_PORT}"
echo "Inference API:   http://127.0.0.1:${SERVER_PORT}/v1/chat/completions"
