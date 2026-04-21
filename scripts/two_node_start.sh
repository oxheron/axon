#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Axon two-node local stack (no Docker) — same idea as scripts/start.sh:
#   go run coordinator + two python node agents + user service.
#
# Usage:
#   bash scripts/two_node_start.sh
#
# Environment variables:
#   MODEL                 Model name/path (default: Qwen/Qwen2.5-3B-Instruct)
#   COORD_PORT            Coordinator port (default: 8000)
#   NODE_A_PORT / NODE_B_PORT   Node control ports (default: 9000 / 9001)
#   VLLM_A_PORT / VLLM_B_PORT   vLLM worker ports (default: 8100 / 8101)
#   SERVER_PORT           User service port (default: 8080)
#   AXON_EXECUTION_MODE   Coordinator mode (default: vllm_ray_pipeline)
#   STARTUP_TIMEOUT       Seconds to wait for readiness (default: 600)
#   AUTOSTART_RAY_HEAD    Set to 0 to skip --autostart-ray-head (default: 1 for
#                         non-dry_run; 0 when AXON_EXECUTION_MODE=dry_run)
#   AXON_TWO_NODE_LOCK_FILE   Lock path (default: /tmp/axon-two-node.${COORD_PORT}.lock)
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
COORD_PORT="${COORD_PORT:-8000}"
NODE_A_PORT="${NODE_A_PORT:-9000}"
NODE_B_PORT="${NODE_B_PORT:-9001}"
VLLM_A_PORT="${VLLM_A_PORT:-8100}"
VLLM_B_PORT="${VLLM_B_PORT:-8101}"
SERVER_PORT="${SERVER_PORT:-8080}"
AXON_EXECUTION_MODE="${AXON_EXECUTION_MODE:-vllm_ray_pipeline}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"

if [[ ! -v AUTOSTART_RAY_HEAD ]]; then
  if [[ "$AXON_EXECUTION_MODE" == "dry_run" ]]; then
    AUTOSTART_RAY_HEAD=0
  else
    AUTOSTART_RAY_HEAD=1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_SRC_DIR="$PROJECT_DIR/user/src"
NODE_SRC_DIR="$PROJECT_DIR/node/src"
LOG_DIR="$PROJECT_DIR"

LOCK_FILE="${AXON_TWO_NODE_LOCK_FILE:-/tmp/axon-two-node.${COORD_PORT}.lock}"
exec {AXON_LOCK_FD}>"$LOCK_FILE"
if ! flock -n "$AXON_LOCK_FD"; then
  echo "ERROR: Lock $LOCK_FILE is busy — another two_node_start.sh is already running" >&2
  echo "  with COORD_PORT=${COORD_PORT}, or a previous run did not exit cleanly." >&2
  echo "  Stop the other process, or use: COORD_PORT=8001 bash $0" >&2
  exit 1
fi

export MODEL AXON_EXECUTION_MODE

COORD_BIN=""
PIDS=()

cleanup() {
  echo ""
  echo "Shutting down all services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ "$AUTOSTART_RAY_HEAD" == "1" ]]; then
    ray stop --force 2>/dev/null || true
  fi
  wait "${PIDS[@]}" 2>/dev/null || true
  [[ -n "$COORD_BIN" ]] && rm -f "$COORD_BIN" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

show_tcp_listeners() {
  local port="$1"
  echo "" >&2
  echo "--- What is listening on TCP port ${port}? ---" >&2
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -F ":${port} " || ss -ltn 2>/dev/null | grep -F ":${port} " || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  fi
}

require_tcp_port_free() {
  local port="$1"
  local label="$2"
  if ! python3 -c "
import socket, sys
p = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('0.0.0.0', p))
except OSError as e:
    print(f'bind 0.0.0.0:{p} failed: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    s.close()
" "$port"; then
    echo "ERROR: ${label} (${port}) is not free." >&2
    show_tcp_listeners "$port"
    echo "  Stop the process above (or reboot stale Ray/coordinator), or choose another port." >&2
    exit 1
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local timeout="${3:-60}"
  local start elapsed
  start="$(date +%s)"
  echo "Waiting for $label..."
  while true; do
    elapsed="$(( $(date +%s) - start ))"
    if (( elapsed > timeout )); then
      echo "ERROR: Timed out waiting for $label after ${timeout}s" >&2
      exit 1
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "  $label is up."
      return 0
    fi
    sleep 2
  done
}

wait_for_coordinator_two_node_ready() {
  local url="$1"
  local timeout="${2:-300}"
  local start elapsed resp
  start="$(date +%s)"
  echo "Waiting for two-node coordinator readiness (mode=${AXON_EXECUTION_MODE})..."
  while true; do
    elapsed="$(( $(date +%s) - start ))"
    if (( elapsed > timeout )); then
      echo "ERROR: Coordinator not ready after ${timeout}s." >&2
      echo "  Logs: ${LOG_DIR}/coordinator.log node-a.log node-b.log user.log" >&2
      echo "  Hint: curl -s http://127.0.0.1:${COORD_PORT}/status | python3 -m json.tool" >&2
      exit 1
    fi
    resp="$(curl -fsS "$url" 2>/dev/null || true)"
    if [[ -n "$resp" ]]; then
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
' <<<"$resp" 2>/dev/null; then
        echo "  Coordinator reports expected readiness."
        return 0
      fi
    fi
    sleep 5
  done
}

echo "========================================"
echo "  Axon — two-node stack (no Docker)"
echo "  Model: $MODEL  mode: $AXON_EXECUTION_MODE"
echo "========================================"
echo ""

coord_health_url="http://127.0.0.1:${COORD_PORT}/healthz"
require_tcp_port_free "$COORD_PORT" "Coordinator HTTP port"

coord_extra=()
if [[ "$AUTOSTART_RAY_HEAD" == "1" ]]; then
  coord_extra+=(--autostart-ray-head)
fi

# 1. Coordinator (waits for two registrations before startup broadcast)
COORD_BIN="$(mktemp -t axon-coordinator-XXXXXX)"
echo "Building coordinator..."
(cd "$PROJECT_DIR/coordinator" && go build -o "$COORD_BIN" ./src) || {
  echo "ERROR: Failed to build coordinator" >&2
  exit 1
}
"$COORD_BIN" \
  --host 0.0.0.0 \
  --port "$COORD_PORT" \
  --min-nodes 2 \
  --model-name "$MODEL" \
  --execution-mode "$AXON_EXECUTION_MODE" \
  "${coord_extra[@]}" \
  >"$LOG_DIR/coordinator.log" 2>&1 &
PIDS+=($!)
wait_for_url "$coord_health_url" "coordinator (port $COORD_PORT)" 30
if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
  echo "ERROR: Coordinator process exited (bind race or crash). Last lines of coordinator.log:" >&2
  tail -30 "$LOG_DIR/coordinator.log" >&2
  show_tcp_listeners "$COORD_PORT"
  exit 1
fi

# 2a. First node (becomes entry when registered first)
python3 "$NODE_SRC_DIR/agent.py" \
  --coordinator-url "http://127.0.0.1:${COORD_PORT}" \
  --host 0.0.0.0 \
  --port "$NODE_A_PORT" \
  --node-id node-a \
  --advertise-host 127.0.0.1 \
  --vllm-worker-port "$VLLM_A_PORT" \
  >"$LOG_DIR/node-a.log" 2>&1 &
PIDS+=($!)
wait_for_url "http://127.0.0.1:${NODE_A_PORT}/healthz" "node-a (port $NODE_A_PORT)" 30

# 2b. Second node
python3 "$NODE_SRC_DIR/agent.py" \
  --coordinator-url "http://127.0.0.1:${COORD_PORT}" \
  --host 0.0.0.0 \
  --port "$NODE_B_PORT" \
  --node-id node-b \
  --advertise-host 127.0.0.1 \
  --vllm-worker-port "$VLLM_B_PORT" \
  >"$LOG_DIR/node-b.log" 2>&1 &
PIDS+=($!)
wait_for_url "http://127.0.0.1:${NODE_B_PORT}/healthz" "node-b (port $NODE_B_PORT)" 30

# 3. User service
python3 "$USER_SRC_DIR/server.py" \
  --coordinator-url "http://127.0.0.1:${COORD_PORT}" \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  >"$LOG_DIR/user.log" 2>&1 &
PIDS+=($!)
wait_for_url "http://127.0.0.1:${SERVER_PORT}/healthz" "user service (port $SERVER_PORT)" 30

# 4. Wait for coordinator topology + vLLM (dry_run skips inference; other modes require it)
wait_for_coordinator_two_node_ready "http://127.0.0.1:${COORD_PORT}/status" "$STARTUP_TIMEOUT"

echo ""
echo "========================================"
echo "  Two-node stack ready (coordinator status)"
echo "========================================"
echo ""
echo "  Coordinator: http://127.0.0.1:${COORD_PORT}/status"
echo "  User API:    http://127.0.0.1:${SERVER_PORT}/v1/chat/completions"
echo ""
if [[ "$AXON_EXECUTION_MODE" == "dry_run" ]]; then
  echo "  dry_run: control plane only; use vllm_ray_pipeline (default) for real inference."
else
  echo "  Inference: curl the user API or run bash scripts/test.sh once status is ready."
fi
echo ""
echo "  Logs: coordinator.log  node-a.log  node-b.log  user.log"
echo "  Press Ctrl+C to stop all services."
echo ""

wait "${PIDS[@]}"
