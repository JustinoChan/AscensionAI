#!/bin/bash
# AscensionAI Training Runner — launches N headless STS workers
#
# Usage:
#   ./run_training.sh                     # 8 workers, 12 hours, 25 games/restart
#   ./run_training.sh --workers 6 --hours 24 --restart-every 30
#   ./run_training.sh --workers 10 --hours 8 --restart-every 40
#
# Each worker runs inside its own Xvfb display, cycling through games
# until the time limit expires. Workers restart every N games (configurable)
# to prevent JVM memory leaks / OOM (heap grows over a long session).
#
# The trainer (train_offline.py) runs alongside, consuming rollouts as they
# arrive. When the timer expires, all workers and the trainer stop gracefully.

set -e

# ─── Defaults ───────────────────────────────────────────────────────────────
WORKERS=8
HOURS=12
RESTART_EVERY=25
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
            echo "  --restart-every N  Games per worker before JVM restart (default: 25)"
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
if [ ! -f "$GAME_DIR/desktop-1.0.jar" ]; then
    echo "ERROR: desktop-1.0.jar not found in $GAME_DIR"
    echo "Copy your game files first. See vm/quickstart.sh"
    exit 1
fi

if [ ! -f "$GAME_DIR/ModTheSpire.jar" ]; then
    echo "ERROR: ModTheSpire.jar not found in $GAME_DIR"
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

# Set up mods directory — ModTheSpire looks for mods/ relative to game dir
mkdir -p "$GAME_DIR/mods"
for jar in BaseMod.jar CommunicationMod.jar SuperFastMode.jar; do
    if [ -f "$GAME_DIR/$jar" ] && [ ! -f "$GAME_DIR/mods/$jar" ]; then
        ln -sf "$GAME_DIR/$jar" "$GAME_DIR/mods/$jar"
    fi
done

DURATION_SECS=$((HOURS * 3600))
END_TIME=$(($(date +%s) + DURATION_SECS))

# Each worker gets its own Xvfb display to avoid rendering contention.
# Shared display causes 100x+ slowdown with multiple OpenGL windows.
export LIBGL_ALWAYS_SOFTWARE=1

echo "=== AscensionAI Training ==="
echo "Workers:        $WORKERS"
echo "Duration:       ${HOURS}h (until $(date -d @$END_TIME '+%Y-%m-%d %H:%M'))"
echo "Restart every:  $RESTART_EVERY games per worker"
echo "Model:          $MODEL"
echo "Trainer:        $RUN_TRAINER"
echo ""

# ─── Instance config generation ─────────────────────────────────────────────
# CommunicationMod ignores XDG_CONFIG_HOME — all workers share ~/.config/.
# Worker ID defaults to PID, so each spawned python process gets a unique ID.

generate_config() {
    local commmod_dir="$HOME/.config/ModTheSpire/CommunicationMod"
    local sfm_dir="$HOME/.config/ModTheSpire/SuperFastMode"
    mkdir -p "$commmod_dir" "$sfm_dir"

    cat > "$commmod_dir/config.properties" << EOF
command=python3 $SCRIPTS_DIR/rollout_worker.py --model $PROJECT_DIR/$MODEL --out $PROJECT_DIR/$ROLLOUT_DIR --restart-every $RESTART_EVERY --verbose
runAtGameStart=true
EOF

    cat > "$sfm_dir/SuperFastModeConfig.properties" << EOF
isDeltaMultiplied=true
deltaMultiplier=4.999997
isInstantLerp=true
EOF
}

# ─── Worker launcher ────────────────────────────────────────────────────────
launch_worker() {
    local id=$1
    local log_file="$PROJECT_DIR/logs/worker_${id}.log"
    local tmpdir="/tmp/sts_worker_${id}"
    local display_num=$((99 + id))
    mkdir -p "$tmpdir"

    # Each worker gets its own Xvfb — small resolution reduces software rendering cost
    Xvfb :$display_num -screen 0 320x240x16 -ac +extension GLX +extension RANDR &>/dev/null &
    local xvfb_pid=$!
    export DISPLAY=:$display_num
    sleep 1

    while [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; do
        echo "[$(date '+%H:%M:%S')] Worker $id: starting STS instance (display :$display_num)"

        cd "$GAME_DIR"
        DISPLAY=:$display_num java -Xmx2048m -Xms512m \
            -Dorg.lwjgl.openal.libname=/usr/lib/x86_64-linux-gnu/libopenal.so.1 \
            -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true \
            -Djava.io.tmpdir="$tmpdir" \
            -jar ModTheSpire.jar \
            --skip-launcher \
            --mods basemod,CommunicationMod,superfastmode \
            >> "$log_file" 2>&1 || true

        if [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; then
            echo "[$(date '+%H:%M:%S')] Worker $id: restarting after game cycle"
            sleep 3
        fi
    done
    echo "[$(date '+%H:%M:%S')] Worker $id: time limit reached, stopping"
    kill $xvfb_pid 2>/dev/null
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
            --max-rollout-lag 9999 \
            --ent-coef 0.001 \
            --auto-tune \
            >> "$log_file" 2>&1 || true

        if [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; then
            sleep 10
        fi
    done
    echo "[$(date '+%H:%M:%S')] Trainer: time limit reached, stopping"
}

# ─── Launch everything ──────────────────────────────────────────────────────
generate_config

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
