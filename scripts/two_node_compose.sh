#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
AXON_EXECUTION_MODE="${AXON_EXECUTION_MODE:-dry_run}"
COORD_PORT="${COORD_PORT:-8000}"
SERVER_PORT="${SERVER_PORT:-8080}"
COMPOSE_FILE="${COMPOSE_FILE:-docker/two-node.compose.yaml}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
KEEP_RUNNING="${KEEP_RUNNING:-0}"

# Docker Compose V2 is `docker compose`; older installs use `docker-compose`.
docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "Docker Compose is required: install the Compose V2 plugin (try: docker compose version)" >&2
    echo "or install standalone docker-compose on PATH." >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_docker_daemon() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "Cannot reach the Docker daemon (docker info failed)." >&2
  echo "Typical fixes on Linux:" >&2
  echo "  - Start the service:  sudo systemctl start docker" >&2
  echo "  - Or enable on boot:  sudo systemctl enable --now docker" >&2
  echo "  - If you use rootless Docker, set DOCKER_HOST (see docker context ls)." >&2
  echo "  - If /var/run/docker.sock is missing, the daemon is not running or not installed." >&2
  exit 1
}

cleanup() {
  if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo "KEEP_RUNNING=1, leaving compose stack running."
    return
  fi
  docker_compose -f "$COMPOSE_FILE" down --remove-orphans >/dev/null 2>&1 || true
}

trap cleanup EXIT

require_cmd docker
require_cmd curl
require_cmd python3
require_docker_daemon

echo "Bringing up two-node Axon stack (mode=${AXON_EXECUTION_MODE})..."
MODEL="$MODEL" AXON_EXECUTION_MODE="$AXON_EXECUTION_MODE" docker_compose -f "$COMPOSE_FILE" up -d --build

echo "Waiting for coordinator readiness..."
start_ts="$(date +%s)"
while true; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if (( elapsed > TIMEOUT_SEC )); then
    echo "Timed out waiting for coordinator readiness after ${TIMEOUT_SEC}s" >&2
    docker_compose -f "$COMPOSE_FILE" logs --tail 80 coordinator node-a node-b user || true
    exit 1
  fi

  status_json="$(curl -fsS "http://127.0.0.1:${COORD_PORT}/status" 2>/dev/null || true)"
  if [[ -n "$status_json" ]]; then
    if python3 -c '
import json, os, sys
d = json.loads(sys.stdin.read())
mode = os.environ["AXON_EXECUTION_MODE"]
cluster_ok = d.get("cluster_ready") and d.get("all_nodes_ready") and d.get("entry_node_ready")
if mode == "dry_run":
    ok = cluster_ok and not d.get("inference_ready")
else:
    ok = cluster_ok and d.get("backend_ready") and d.get("inference_ready")
raise SystemExit(0 if ok else 1)
' <<<"$status_json"; then
      echo "Coordinator reports expected readiness."
      break
    fi
  fi
  sleep 5
done

echo ""
echo "Coordinator: http://127.0.0.1:${COORD_PORT}/status"
echo "User API:    http://127.0.0.1:${SERVER_PORT}/v1/chat/completions"

if [[ "$AXON_EXECUTION_MODE" == "dry_run" ]]; then
  echo "Dry-run mode completed: control plane and topology are ready, but inference is intentionally disabled."
  exit 0
fi

echo "Sending smoke inference request..."
curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: axon\"}],
    \"max_tokens\": 32
  }"
echo ""
