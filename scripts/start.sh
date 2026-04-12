#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Axon single-machine startup script
# Starts coordinator, node agent, and inference server in the correct order.
#
# Usage:
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/start.sh
#
# Environment variables:
#   MODEL             Model name/path  (default: Qwen/Qwen2.5-3B-Instruct)
#   COORD_PORT        Coordinator port (default: 8000)
#   NODE_PORT         Node agent port  (default: 9000)
#   SERVER_PORT       Inference server port (default: 8080)
#   VLLM_PORT         vLLM worker port (default: 8100)
#   STARTUP_TIMEOUT   Seconds to wait for model to load (default: 600)
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
COORD_PORT="${COORD_PORT:-8000}"
NODE_PORT="${NODE_PORT:-9000}"
SERVER_PORT="${SERVER_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8100}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
LOG_DIR="$PROJECT_DIR"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for a URL to return HTTP 200.
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

# Wait until /healthz reports pipeline_ready=true.
wait_for_pipeline_ready() {
    local url="$1"
    local timeout="${2:-600}"
    local start elapsed resp
    start="$(date +%s)"
    echo "Waiting for model to load (this may take several minutes)..."
    while true; do
        elapsed="$(( $(date +%s) - start ))"
        if (( elapsed > timeout )); then
            echo "ERROR: Pipeline not ready after ${timeout}s. Check server.log for details." >&2
            exit 1
        fi
        resp="$(curl -fsS "$url" 2>/dev/null || true)"
        if [[ -n "$resp" ]]; then
            if python3 -c \
                'import json,sys; d=json.loads(sys.stdin.read()); sys.exit(0 if d.get("pipeline_ready") else 1)' \
                <<<"$resp" 2>/dev/null; then
                echo "  Pipeline is ready!"
                return 0
            fi
        fi
        sleep 5
    done
}

echo "========================================"
echo "  Axon — starting all services"
echo "  Model: $MODEL"
echo "========================================"
echo ""

# 1. Coordinator
PYTHONPATH="$SRC_DIR" python3 -m axon.coordinator \
    --host 0.0.0.0 \
    --port "$COORD_PORT" \
    --min-nodes 1 \
    --model-name "$MODEL" \
    >"$LOG_DIR/coordinator.log" 2>&1 &
PIDS+=($!)
wait_for_url "http://localhost:${COORD_PORT}/healthz" "coordinator (port $COORD_PORT)" 30

# 2. Node agent
PYTHONPATH="$SRC_DIR" python3 -m axon.node_agent \
    --coordinator-url "http://localhost:${COORD_PORT}" \
    --host 0.0.0.0 \
    --port "$NODE_PORT" \
    --vllm-worker-port "$VLLM_PORT" \
    >"$LOG_DIR/node_agent.log" 2>&1 &
PIDS+=($!)
wait_for_url "http://localhost:${NODE_PORT}/healthz" "node agent (port $NODE_PORT)" 30

# 3. Inference server
PYTHONPATH="$SRC_DIR" python3 -m axon.server \
    --coordinator-url "http://localhost:${COORD_PORT}" \
    --vllm-worker-url "http://localhost:${VLLM_PORT}" \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    >"$LOG_DIR/server.log" 2>&1 &
PIDS+=($!)
wait_for_url "http://localhost:${SERVER_PORT}/healthz" "inference server (port $SERVER_PORT)" 30

# 4. Wait for model to finish loading
wait_for_pipeline_ready "http://localhost:${SERVER_PORT}/healthz" "$STARTUP_TIMEOUT"

echo ""
echo "========================================"
echo "  System ready!"
echo "========================================"
echo ""
echo "  Test with:"
echo ""
echo "    curl http://localhost:${SERVER_PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 64}'"
echo ""
echo "  Or run:  bash scripts/test.sh"
echo ""
echo "  Logs:  coordinator.log  node_agent.log  server.log"
echo "  Press Ctrl+C to stop all services."
echo ""

# Keep script alive so the trap fires on Ctrl+C.
wait "${PIDS[@]}"
