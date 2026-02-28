#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

set -a && source "$SCRIPT_DIR/.env" && set +a

WALL_CLOCK_LIMIT=${WALL_CLOCK_LIMIT:-43200}
DEADLINE_BUFFER=${DEADLINE_BUFFER:-300}
START_EPOCH=$(date +%s)

SMOKE_TASK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE_TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: ./run-opencode.sh [--smoke <task_id>]" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$SMOKE_TASK" ]]; then
    echo "=== ARC-AGI Smoke Test (OpenCode): $SMOKE_TASK ==="
    OPENCODE_TASKS="$SMOKE_TASK"
    OPENCODE_AGENTS=5
    OPENCODE_MAX_ITERATIONS=5
    OPENCODE_CONCURRENCY=5
    RUN_NAME="SMOKE_${SMOKE_TASK}"
else
    echo "=== Final ARC-AGI Run (OpenCode) ==="
    OPENCODE_TASKS="all"
    OPENCODE_AGENTS=12
    OPENCODE_MAX_ITERATIONS=10
    OPENCODE_CONCURRENCY=132
    RUN_NAME="FINAL_RUN"
fi

echo "Script dir: $SCRIPT_DIR"
echo ""

    echo "--- Starting OpenCode CLI solver (Docker) ---"
    echo "[opencode] $OPENCODE_AGENTS agents, max-iterations $OPENCODE_MAX_ITERATIONS"
    echo ""

    cd "$SCRIPT_DIR"
    uv run python run.py \
        --tasks "$OPENCODE_TASKS" \
        --num-agents "$OPENCODE_AGENTS" \
        --max-iterations "$OPENCODE_MAX_ITERATIONS" \
        --concurrency "${OPENCODE_CONCURRENCY:-40}" \
        --name "$RUN_NAME" &
PID_SOLVER=$!

echo "Solver PID: $PID_SOLVER"
echo ""

_cleanup() {
    echo ""
    echo "--- SIGTERM/SIGINT received — cleaning up ---"
    kill "$PID_SOLVER" 2>/dev/null
    sleep 5
    kill -9 "$PID_SOLVER" 2>/dev/null
    wait "$PID_SOLVER" 2>/dev/null
    [[ -n "${PID_WATCHDOG:-}" ]] && kill "$PID_WATCHDOG" 2>/dev/null
    echo "--- Producing submission.json from partial results ---"
    cd "$SCRIPT_DIR"
    python3 submission.py --solver opencode || true
    echo "=== Done (via signal handler) ==="
    exit 1
}
trap _cleanup SIGTERM SIGINT

SLEEP_SECS=$(( WALL_CLOCK_LIMIT - DEADLINE_BUFFER ))
if (( SLEEP_SECS > 0 )); then
    (
        sleep "$SLEEP_SECS"
        ELAPSED=$(( $(date +%s) - START_EPOCH ))
        echo ""
        echo "--- WATCHDOG: ${ELAPSED}s elapsed, killing solver (${DEADLINE_BUFFER}s before ${WALL_CLOCK_LIMIT}s limit) ---"
        kill "$PID_SOLVER" 2>/dev/null
        sleep 15
        kill -9 "$PID_SOLVER" 2>/dev/null
    ) &
    PID_WATCHDOG=$!
    echo "Watchdog PID: $PID_WATCHDOG (will fire in ${SLEEP_SECS}s)"
else
    echo "Warning: WALL_CLOCK_LIMIT ($WALL_CLOCK_LIMIT) <= DEADLINE_BUFFER ($DEADLINE_BUFFER), no watchdog started"
fi
echo ""

set +e
wait "$PID_SOLVER"
SOLVER_EXIT=$?
set -e

[[ -n "${PID_WATCHDOG:-}" ]] && kill "$PID_WATCHDOG" 2>/dev/null && wait "$PID_WATCHDOG" 2>/dev/null || true

echo ""
echo "--- Solver result ---"
if [[ $SOLVER_EXIT -eq 0 ]]; then
    echo "[opencode] SUCCESS"
else
    echo "[opencode] FAILED (exit code $SOLVER_EXIT)"
fi
echo ""

echo "--- Building submission ---"
cd "$SCRIPT_DIR"
python3 submission.py --solver opencode

echo ""
if [[ $SOLVER_EXIT -ne 0 ]]; then
    echo "=== Done (with solver errors) ==="
    exit 1
else
    echo "=== Done! ==="
fi
