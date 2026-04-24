#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run a single Axon node agent as a foreground process.
#
# Usage:
#   COORDINATOR_URL=http://<coordinator-ip>:8000 bash scripts/run_node.sh
#
# Environment variables:
#   COORDINATOR_URL       Full URL of the coordinator (required)
#   NODE_PORT             Node control API bind port (default: 9000)
#   VLLM_PORT             vLLM worker subprocess port (default: 8100)
#   ADVERTISE_HOST        IP/hostname this node tells the coordinator to use
#                         for callbacks and worker URLs. Must be reachable FROM
#                         the coordinator machine. Defaults to auto-detected
#                         local IP — set this explicitly on multi-machine setups.
#   ADVERTISE_PORT        Advertised control-API port (default: NODE_PORT).
#                         Override when using NAT/port-forwarding.
#   NODE_ID               Unique node identifier (default: <hostname>-<uuid8>)
#   VLLM_GPU_MEM_UTIL     --gpu-memory-utilization for vLLM (default: 0.72)
#   VLLM_MAX_MODEL_LEN    --max-model-len for vLLM (default: 1024)
#   VLLM_DTYPE            --dtype for vLLM (default: float16)
#   NO_VLLM               Set to 1 to skip vLLM launch and only join Ray
#                         (useful for pipeline nodes that are not the entry)
#   LOG_FILE              Path for log output (default: node-<NODE_ID>.log in
#                         project root). Set to "" to log to stdout.
# ---------------------------------------------------------------------------

COORDINATOR_URL="${COORDINATOR_URL:-}"
NODE_PORT="${NODE_PORT:-9000}"
VLLM_PORT="${VLLM_PORT:-8100}"
ADVERTISE_HOST="${ADVERTISE_HOST:-}"
ADVERTISE_PORT="${ADVERTISE_PORT:-}"
NODE_ID="${NODE_ID:-}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.72}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1024}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
NO_VLLM="${NO_VLLM:-0}"
LOG_FILE="${LOG_FILE:-}"   # resolved below after NODE_ID is known

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE_SRC="$PROJECT_DIR/node/src/agent.py"

if [[ -z "$COORDINATOR_URL" ]]; then
    echo "ERROR: COORDINATOR_URL is required." >&2
    echo "  Usage: COORDINATOR_URL=http://<coordinator-ip>:8000 bash $0" >&2
    exit 1
fi

# Resolve a display name for NODE_ID before using it in LOG_FILE
NODE_ID_DISPLAY="${NODE_ID:-<auto>}"

# Default LOG_FILE (resolved after NODE_ID may be known)
if [[ ! -v LOG_FILE ]]; then
    if [[ -n "$NODE_ID" ]]; then
        LOG_FILE="$PROJECT_DIR/node-${NODE_ID}.log"
    else
        LOG_FILE="$PROJECT_DIR/node.log"
    fi
fi

echo "========================================"
echo "  Axon Node Agent"
echo "  Coordinator:    $COORDINATOR_URL"
echo "  Control port:   $NODE_PORT"
echo "  vLLM port:      $VLLM_PORT"
echo "  Node ID:        $NODE_ID_DISPLAY"
[[ -n "$ADVERTISE_HOST" ]] && echo "  Advertise host: $ADVERTISE_HOST"
[[ -n "$ADVERTISE_PORT" ]] && echo "  Advertise port: $ADVERTISE_PORT"
[[ "$NO_VLLM" == "1" ]] && echo "  vLLM launch:    disabled (Ray join only)"
[[ -n "$LOG_FILE" ]] && echo "  Log:            $LOG_FILE"
echo "========================================"
echo ""
echo "Port-forward requirements (inbound on this machine):"
echo "  ${NODE_PORT}        Node control API    (coordinator sends /startup callback)"
echo "  ${VLLM_PORT}        vLLM worker API     (coordinator proxies inference here)"
echo "  10000-19999  Ray worker traffic   (inter-node tensor passing, multi-node only)"
echo ""
echo "Connecting to coordinator at $COORDINATOR_URL ..."
echo ""

build_args=(
    --coordinator-url "$COORDINATOR_URL"
    --host 0.0.0.0
    --port "$NODE_PORT"
    --vllm-worker-port "$VLLM_PORT"
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
    --vllm-dtype "$VLLM_DTYPE"
)

[[ -n "$ADVERTISE_HOST" ]] && build_args+=(--advertise-host "$ADVERTISE_HOST")
[[ -n "$ADVERTISE_PORT" ]] && build_args+=(--advertise-port "$ADVERTISE_PORT")
[[ -n "$NODE_ID" ]]        && build_args+=(--node-id "$NODE_ID")
[[ "$NO_VLLM" == "1" ]]    && build_args+=(--no-vllm-worker)

run() {
    python3 "$NODE_SRC" "${build_args[@]}"
}

if [[ -n "$LOG_FILE" ]]; then
    echo "Logging to $LOG_FILE  (also tailed below)"
    echo ""
    run >"$LOG_FILE" 2>&1 &
    NODE_PID=$!
    trap 'kill "$NODE_PID" 2>/dev/null; wait "$NODE_PID" 2>/dev/null; exit' INT TERM EXIT
    tail -f "$LOG_FILE" &
    TAIL_PID=$!
    trap 'kill "$NODE_PID" "$TAIL_PID" 2>/dev/null; wait "$NODE_PID" 2>/dev/null; exit' INT TERM EXIT
    wait "$NODE_PID"
else
    run
fi
