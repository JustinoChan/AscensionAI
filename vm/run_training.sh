#!/bin/bash
# AscensionAI Training Runner — launches N headless STS workers
#
# Usage:
#   ./run_training.sh                     # 8 workers, 12 hours, 50 games/restart
#   ./run_training.sh --workers 6 --hours 24 --restart-every 30
#   ./run_training.sh --workers 10 --hours 8 --restart-every 40
#
# Each worker runs inside its own xvfb-run display, cycling through games
# until the time limit expires. Workers restart every N games (configurable)
# to prevent JVM memory leaks.
#
# The trainer (train_offline.py) runs alongside, consuming rollouts as they
# arrive. When the timer expires, all workers and the trainer stop gracefully.

set -e

# ─── Defaults ───────────────────────────────────────────────────────────────
WORKERS=8
HOURS=12
RESTART_EVERY=50
MODEL="models/ppo_sts.pt"
ROLLOUT_DIR="rollouts_shared"
RUN_TRAINER=true
BATCH_GAMES=8
MAX_ROLLOUT_LAG=4

# ─── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)      WORKERS="$2"; shift 2;;
        --hours)        HOURS="$2"; shift 2;;
        --restart-every) RESTART_EVERY="$2"; shift 2;;
        --model)        MODEL="$2"; shift 2;;
        --no-trainer)   RUN_TRAINER=false; shift;;
        --batch-games)  BATCH_GAMES="$2"; shift 2;;
        --max-lag)      MAX_ROLLOUT_LAG="$2"; shift 2;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --workers N        Number of STS instances (default: 8)"
            echo "  --hours N          Run duration in hours (default: 12)"
            echo "  --restart-every N  Games per worker before JVM restart (default: 50)"
            echo "  --model PATH       Model checkpoint path (default: models/ppo_sts.pt)"
            echo "  --no-trainer       Don't run the offline trainer (workers only)"
            echo "  --batch-games N    Rollouts per PPO update (default: 8)"
            echo "  --max-lag N        Max rollout lag before rejection (default: 4)"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

PROJECT_DIR="$HOME/ascension"
GAME_DIR="$PROJECT_DIR/game"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
VENV="$PROJECT_DIR/.venv/bin/activate"
PIDS_FILE="$PROJECT_DIR/logs/.training_pids"
STOP_FILE="$PROJECT_DIR/logs/.stop_training"

# ─── Preflight checks ───────────────────────────────────────────────────────
if [ ! -f "$GAME_DIR/desktop-1.0.jar" ] && [ ! -f "$GAME_DIR/SlayTheSpire.jar" ]; then
    echo "ERROR: STS game files not found in $GAME_DIR"
    echo "Copy your game directory first. See vm/setup.sh instructions."
    exit 1
fi

if [ ! -f "$PROJECT_DIR/$MODEL" ]; then
    echo "ERROR: Model not found at $PROJECT_DIR/$MODEL"
    echo "Sync your model first:  vm/sync.sh push"
    exit 1
fi

source "$VENV"
rm -f "$STOP_FILE"
mkdir -p "$PROJECT_DIR/$ROLLOUT_DIR" "$PROJECT_DIR/logs"

DURATION_SECS=$((HOURS * 3600))
END_TIME=$(($(date +%s) + DURATION_SECS))

echo "=== AscensionAI Training ==="
echo "Workers:        $WORKERS"
echo "Duration:       ${HOURS}h (until $(date -d @$END_TIME '+%Y-%m-%d %H:%M'))"
echo "Restart every:  $RESTART_EVERY games per worker"
echo "Model:          $MODEL"
echo "Trainer:        $RUN_TRAINER"
echo ""

# ─── Instance config generation ─────────────────────────────────────────────
# Each worker instance needs its own CommunicationMod config pointing to
# the correct python command with --id flag

generate_config() {
    local id=$1
    local instance_dir="$PROJECT_DIR/instances/worker_$id"
    mkdir -p "$instance_dir/mods/CommunicationMod"

    cat > "$instance_dir/mods/CommunicationMod/config.properties" << EOF
command=python3 $SCRIPTS_DIR/rollout_worker.py --model $PROJECT_DIR/$MODEL --out $PROJECT_DIR/$ROLLOUT_DIR --id $id --restart-every $RESTART_EVERY
runAtGameStart=true
EOF
}

# ─── Worker launcher ────────────────────────────────────────────────────────
launch_worker() {
    local id=$1
    generate_config "$id"
    local instance_dir="$PROJECT_DIR/instances/worker_$id"
    local log_file="$PROJECT_DIR/logs/worker_${id}.log"

    while [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; do
        echo "[$(date '+%H:%M:%S')] Worker $id: starting STS instance"

        # Launch STS headlessly with xvfb
        xvfb-run -a \
            java -Xmx512m -Xms256m \
            -cp "$GAME_DIR/desktop-1.0.jar:$GAME_DIR/*" \
            --add-opens java.base/java.lang=ALL-UNNAMED \
            com.megacrit.cardcrawl.desktop.DesktopLauncher \
            --mods CommunicationMod \
            --mods-dir "$instance_dir/mods" \
            >> "$log_file" 2>&1 || true

        # Worker exited (restart-every or crash) — brief pause then restart
        if [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; then
            echo "[$(date '+%H:%M:%S')] Worker $id: restarting after game cycle"
            sleep 3
        fi
    done
    echo "[$(date '+%H:%M:%S')] Worker $id: time limit reached, stopping"
}

# ─── Offline trainer ────────────────────────────────────────────────────────
run_trainer() {
    local log_file="$PROJECT_DIR/logs/trainer.log"
    echo "[$(date '+%H:%M:%S')] Trainer: starting offline trainer"

    while [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; do
        python3 "$SCRIPTS_DIR/train_offline.py" \
            --model "$PROJECT_DIR/$MODEL" \
            --data "$PROJECT_DIR/$ROLLOUT_DIR" \
            --delete-consumed \
            --batch-games "$BATCH_GAMES" \
            --lr 3e-5 \
            --bc-coef 0.001 \
            --max-rollout-lag "$MAX_ROLLOUT_LAG" \
            --ent-coef 0.001 \
            --auto-tune \
            --max-rollout-lag 9999 \
            >> "$log_file" 2>&1 || true

        if [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; then
            sleep 10
        fi
    done
    echo "[$(date '+%H:%M:%S')] Trainer: time limit reached, stopping"
}

# ─── Launch everything ──────────────────────────────────────────────────────
echo "Starting $WORKERS workers..."
PIDS=()

for i in $(seq 1 $WORKERS); do
    launch_worker "$i" &
    PIDS+=($!)
    sleep 2  # stagger launches to avoid disk contention
done

if [ "$RUN_TRAINER" = true ]; then
    # Wait for first rollouts to appear before starting trainer
    echo "Waiting 60s for initial rollouts before starting trainer..."
    sleep 60
    run_trainer &
    PIDS+=($!)
fi

# Save PIDs for stop script
printf '%s\n' "${PIDS[@]}" > "$PIDS_FILE"

echo ""
echo "All processes launched. PIDs saved to $PIDS_FILE"
echo "Monitor: tail -f $PROJECT_DIR/logs/trainer.log"
echo "Stop:    ./stop.sh"
echo ""
echo "Training will auto-stop at $(date -d @$END_TIME '+%Y-%m-%d %H:%M')"

# Wait for time limit
wait
echo ""
echo "=== Training session complete ==="
echo "Total duration: ${HOURS}h"
echo "Pull results:  vm/sync.sh pull   (from your local machine)"
