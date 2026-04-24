#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run the Axon user service (OpenAI-compatible API proxy) as a foreground
# process.
#
# Usage:
#   COORDINATOR_URL=http://<coordinator-ip>:8000 bash scripts/run_user.sh
#
# Environment variables:
#   COORDINATOR_URL   Full URL of the coordinator (required)
#   SERVER_PORT       User API bind port (default: 8080)
#   LOG_FILE          Path for log output (default: user.log in project root).
#                     Set to "" to log to stdout.
# ---------------------------------------------------------------------------

COORDINATOR_URL="${COORDINATOR_URL:-}"
SERVER_PORT="${SERVER_PORT:-8080}"
LOG_FILE="${LOG_FILE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_SRC="$PROJECT_DIR/user/src/server.py"

if [[ -z "$COORDINATOR_URL" ]]; then
    echo "ERROR: COORDINATOR_URL is required." >&2
    echo "  Usage: COORDINATOR_URL=http://<coordinator-ip>:8000 bash $0" >&2
    exit 1
fi

# Default LOG_FILE to project root
if [[ ! -v LOG_FILE ]]; then
    LOG_FILE="$PROJECT_DIR/user.log"
fi

echo "========================================"
echo "  Axon User Service"
echo "  Coordinator:  $COORDINATOR_URL"
echo "  Listen port:  $SERVER_PORT"
[[ -n "$LOG_FILE" ]] && echo "  Log:          $LOG_FILE"
echo "========================================"
echo ""
echo "No port-forwarding required — this service only makes outbound connections"
echo "to the coordinator. Port ${SERVER_PORT} only needs to be opened if your client"
echo "is on a different machine from this one."
echo ""
echo "Connecting to coordinator at $COORDINATOR_URL ..."
echo ""

run() {
    python3 "$USER_SRC" \
        --coordinator-url "$COORDINATOR_URL" \
        --host 0.0.0.0 \
        --port "$SERVER_PORT"
}

if [[ -n "$LOG_FILE" ]]; then
    echo "Logging to $LOG_FILE  (also tailed below)"
    echo ""
    run >"$LOG_FILE" 2>&1 &
    SVC_PID=$!
    trap 'kill "$SVC_PID" 2>/dev/null; wait "$SVC_PID" 2>/dev/null; exit' INT TERM EXIT
    tail -f "$LOG_FILE" &
    TAIL_PID=$!
    trap 'kill "$SVC_PID" "$TAIL_PID" 2>/dev/null; wait "$SVC_PID" 2>/dev/null; exit' INT TERM EXIT
    wait "$SVC_PID"
else
    run
fi
