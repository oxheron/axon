#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
COORD_PORT="${COORD_PORT:-8000}"
SERVER_PORT="${SERVER_PORT:-8080}"
NETWORK_NAME="${NETWORK_NAME:-axon-test-net}"
TIMEOUT_SEC="${TIMEOUT_SEC:-420}"
KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"
COORD_IMAGE="${COORD_IMAGE:-axon-coordinator:test}"
NODE_IMAGE="${NODE_IMAGE:-axon-node-agent:test}"
SERVER_IMAGE="${SERVER_IMAGE:-axon-server:test}"

# GPU stack: auto (detect), nvidia, or amd. Sets default VLLM_BASE_IMAGE and Docker device flags.
GPU_VENDOR="${GPU_VENDOR:-auto}"

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

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable." >&2
  exit 1
fi

if [[ "${GPU_VENDOR}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
  elif command -v rocm-smi >/dev/null 2>&1; then
    GPU_VENDOR="amd"
  else
    echo "Could not detect GPU: need working nvidia-smi (NVIDIA) or rocm-smi (AMD)." >&2
    echo "Set GPU_VENDOR=nvidia or GPU_VENDOR=amd explicitly if detection fails." >&2
    exit 1
  fi
fi

if [[ -z "${VLLM_BASE_IMAGE:-}" ]]; then
  if [[ "${GPU_VENDOR}" == "amd" ]]; then
    VLLM_BASE_IMAGE="vllm/vllm-openai-rocm:latest"
  else
    VLLM_BASE_IMAGE="vllm/vllm-openai:latest"
  fi
fi

if [[ "${GPU_VENDOR}" == "nvidia" ]]; then
  require_cmd nvidia-smi
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ -z "${GPU_MEM_MB}" ]]; then
    echo "Could not read GPU memory from nvidia-smi." >&2
    exit 1
  fi
  if [[ -n "${AXON_DOCKER_SERVER_GPU_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_SERVER_GPU_FLAGS=( ${AXON_DOCKER_SERVER_GPU_FLAGS} )
  else
    DOCKER_SERVER_GPU_FLAGS=(--gpus all)
  fi
elif [[ "${GPU_VENDOR}" == "amd" ]]; then
  require_cmd rocm-smi
  GPU_MEM_MB="$(
    python3 - <<'PY'
import re, subprocess, sys
r = subprocess.run(
    ["rocm-smi", "--showmeminfo", "vram", "-d", "0"],
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
)
text = (r.stdout or "") + (r.stderr or "")
m = re.search(r"VRAM Total Memory \(MiB\):\s*([0-9]+)", text)
if m:
    print(m.group(1))
    sys.exit(0)
m = re.search(r"VRAM Total Memory \(B\):\s*([0-9]+)", text)
if m:
    print(int(int(m.group(1)) / (1024 * 1024)))
    sys.exit(0)
sys.exit(1)
PY
  )"
  if [[ -z "${GPU_MEM_MB}" ]]; then
    echo "Could not read GPU memory from rocm-smi output." >&2
    exit 1
  fi
  if [[ -n "${AXON_DOCKER_SERVER_GPU_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_SERVER_GPU_FLAGS=( ${AXON_DOCKER_SERVER_GPU_FLAGS} )
  else
    # Matches vLLM ROCm deployment docs (https://docs.vllm.ai/en/stable/deployment/docker.html).
    DOCKER_SERVER_GPU_FLAGS=(
      --device=/dev/kfd
      --device=/dev/dri
      --group-add=video
      --ipc=host
      --cap-add=SYS_PTRACE
      --security-opt=seccomp=unconfined
    )
  fi
else
  echo "GPU_VENDOR must be auto, nvidia, or amd (got: ${GPU_VENDOR})." >&2
  exit 1
fi

echo "Detected GPU vendor: ${GPU_VENDOR} (container base: ${VLLM_BASE_IMAGE})"
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

echo "Building coordinator image (VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE})..."
docker build -f docker/coordinator.Dockerfile --build-arg "VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE}" -t "$COORD_IMAGE" .
echo "Building node image..."
docker build -f docker/node-agent.Dockerfile --build-arg "VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE}" -t "$NODE_IMAGE" .
echo "Building server image..."
docker build -f docker/server.Dockerfile --build-arg "VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE}" -t "$SERVER_IMAGE" .

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
  "${DOCKER_SERVER_GPU_FLAGS[@]}" \
  -p "${SERVER_PORT}:8080" \
  "$SERVER_IMAGE" \
  --coordinator-url "http://${COORD_CONTAINER}:8000" \
  --host 0.0.0.0 \
  --port 8080 >/dev/null

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
