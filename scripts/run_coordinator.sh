#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run the Axon coordinator as a foreground process.
#
# Usage:
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_coordinator.sh
#
# Environment variables:
#   MODEL                 Model name/path (required — no default)
#   COORD_PORT            HTTP bind port (default: 8000)
#   MIN_NODES             Node count required before startup broadcast (default: 1)
#   EXECUTION_MODE        One of: single_node | vllm_ray_pipeline |
#                         slice_loaded_pipeline | dry_run
#                         (default: auto — single_node when MIN_NODES=1,
#                          vllm_ray_pipeline when MIN_NODES>1)
#   AUTOSTART_RAY_HEAD    Set to 1 to run `ray start --head` before serving.
#                         Defaults to 1 when MIN_NODES > 1 and mode is not
#                         dry_run; 0 otherwise.
#   RAY_PORT              Ray GCS port (default: 6379)
#   RAY_NODE_IP           IP passed to `ray start --node-ip-address`.
#                         Defaults to auto-detected local IP. Set this to your
#                         public/LAN IP when nodes connect from other machines.
#   LOG_FILE              Path for log output (default: coordinator.log in
#                         project root). Set to "" to log to stdout.
# ---------------------------------------------------------------------------

MODEL="${MODEL:-}"
COORD_PORT="${COORD_PORT:-8000}"
MIN_NODES="${MIN_NODES:-1}"
EXECUTION_MODE="${EXECUTION_MODE:-}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_NODE_IP="${RAY_NODE_IP:-}"
LOG_FILE="${LOG_FILE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL is required." >&2
    echo "  Usage: MODEL=Qwen/Qwen2.5-3B-Instruct bash $0" >&2
    exit 1
fi

# Default EXECUTION_MODE based on MIN_NODES
if [[ -z "$EXECUTION_MODE" ]]; then
    if (( MIN_NODES > 1 )); then
        EXECUTION_MODE="vllm_ray_pipeline"
    else
        EXECUTION_MODE="single_node"
    fi
fi

# Default AUTOSTART_RAY_HEAD
if [[ ! -v AUTOSTART_RAY_HEAD ]]; then
    if (( MIN_NODES > 1 )) && [[ "$EXECUTION_MODE" != "dry_run" ]]; then
        AUTOSTART_RAY_HEAD=1
    else
        AUTOSTART_RAY_HEAD=0
    fi
fi

# Default LOG_FILE to project root
if [[ ! -v LOG_FILE ]]; then
    LOG_FILE="$PROJECT_DIR/coordinator.log"
fi

echo "========================================"
echo "  Axon Coordinator"
echo "  Model:      $MODEL"
echo "  Port:       $COORD_PORT"
echo "  Min nodes:  $MIN_NODES"
echo "  Mode:       $EXECUTION_MODE"
if [[ "$AUTOSTART_RAY_HEAD" == "1" ]]; then
    echo "  Ray GCS:    $RAY_PORT  (autostart-ray-head=true)"
fi
[[ -n "$LOG_FILE" ]] && echo "  Log:        $LOG_FILE"
echo "========================================"
echo ""
echo "Nodes should register at:  http://<this-host>:${COORD_PORT}/register"
echo ""
if [[ "$AUTOSTART_RAY_HEAD" == "1" ]]; then
    echo "Port-forward requirements (inbound on this machine):"
    echo "  ${COORD_PORT}        HTTP control plane  (nodes + user service)"
    echo "  ${RAY_PORT}        Ray GCS              (nodes joining the Ray cluster)"
    echo "  10000-19999  Ray worker traffic   (inter-node tensor passing)"
    echo ""
else
    echo "Port-forward requirements (inbound on this machine):"
    echo "  ${COORD_PORT}  HTTP control plane  (nodes + user service)"
    echo ""
fi

build_args=(
    --host 0.0.0.0
    --port "$COORD_PORT"
    --min-nodes "$MIN_NODES"
    --model-name "$MODEL"
    --execution-mode "$EXECUTION_MODE"
    --ray-port "$RAY_PORT"
)

[[ -n "$RAY_NODE_IP" ]] && build_args+=(--ray-node-ip "$RAY_NODE_IP")
[[ "$AUTOSTART_RAY_HEAD" == "1" ]] && build_args+=(--autostart-ray-head)

run() {
    cd "$PROJECT_DIR/coordinator" && go run ./src "${build_args[@]}"
}

if [[ -n "$LOG_FILE" ]]; then
    echo "Logging to $LOG_FILE  (also tailed below)"
    echo ""
    run >"$LOG_FILE" 2>&1 &
    COORD_PID=$!
    trap 'kill "$COORD_PID" 2>/dev/null; wait "$COORD_PID" 2>/dev/null; exit' INT TERM EXIT
    tail -f "$LOG_FILE" &
    TAIL_PID=$!
    trap 'kill "$COORD_PID" "$TAIL_PID" 2>/dev/null; wait "$COORD_PID" 2>/dev/null; exit' INT TERM EXIT
    wait "$COORD_PID"
else
    run
fi
