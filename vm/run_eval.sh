#!/bin/bash
# AscensionAI Eval Runner — run a fixed-seed evaluation on the VM
#
# Usage:
#   ./run_eval.sh                           # 200 games, current model
#   ./run_eval.sh --games 200 --tag my_eval
#   ./run_eval.sh --model models/ppo_sts.pt --games 100 --policy heuristic
#   ./run_eval.sh --instances 4 --games 200  # Split across 4 parallel STS instances
#
# Runs greedy (no exploration) evaluation using the same eval_model.py harness.
# Results go to logs/eval_stats.csv, same as local.

set -e

# ─── Defaults ───────────────────────────────────────────────────────────────
GAMES=200
INSTANCES=1
MODEL="models/ppo_sts.pt"
POLICY="model"
SEED_FILE="seeds/eval_200.txt"
RUN_TAG="ppo_current_200_$(date +%Y%m%d_%H%M%S)"
RESTART_EVERY=50

# ─── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --games)         GAMES="$2"; shift 2;;
        --instances)     INSTANCES="$2"; shift 2;;
        --model)         MODEL="$2"; shift 2;;
        --policy)        POLICY="$2"; shift 2;;
        --seed-file)     SEED_FILE="$2"; shift 2;;
        --tag)           RUN_TAG="$2"; shift 2;;
        --restart-every) RESTART_EVERY="$2"; shift 2;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --games N          Total eval games (default: 200)"
            echo "  --instances N      Parallel STS instances (default: 1)"
            echo "  --model PATH       Model to evaluate (default: models/ppo_sts.pt)"
            echo "  --policy TYPE      'model' or 'heuristic' (default: model)"
            echo "  --seed-file PATH   Seed file (default: seeds/eval_200.txt)"
            echo "  --tag NAME         Run tag for CSV logging"
            echo "  --restart-every N  Games per instance restart (default: 50)"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

PROJECT_DIR="$HOME/ascension"
GAME_DIR="$PROJECT_DIR/game"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
VENV="$PROJECT_DIR/.venv/bin/activate"

source "$VENV"

# Set up mods directory
mkdir -p "$GAME_DIR/mods"
for jar in BaseMod.jar CommunicationMod.jar SuperFastMode.jar; do
    if [ -f "$GAME_DIR/$jar" ] && [ ! -f "$GAME_DIR/mods/$jar" ]; then
        ln -sf "$GAME_DIR/$jar" "$GAME_DIR/mods/$jar"
    fi
done

# Start a shared virtual display using Xorg with dummy driver
DISPLAY_NUM=99
if [ -f /etc/X11/xorg-dummy.conf ]; then
    sudo Xorg :$DISPLAY_NUM -config /etc/X11/xorg-dummy.conf -noreset -ac +extension GLX +extension RANDR +extension RENDER &
else
    Xvfb :$DISPLAY_NUM -screen 0 1280x720x24 -ac +extension GLX +extension RANDR &
fi
XORG_PID=$!
export DISPLAY=:$DISPLAY_NUM
export LIBGL_ALWAYS_SOFTWARE=1
sleep 2

echo "=== AscensionAI Evaluation ==="
echo "Games:     $GAMES"
echo "Instances: $INSTANCES"
echo "Model:     $MODEL"
echo "Policy:    $POLICY"
echo "Seed file: $SEED_FILE"
echo "Run tag:   $RUN_TAG"
echo ""

# For single instance, just run directly
if [ "$INSTANCES" -eq 1 ]; then
    INSTANCE_DIR="$PROJECT_DIR/instances/eval_1"
    # CommunicationMod ignores XDG_CONFIG_HOME on Linux, so write to the real ~/.config.
    CONFIG_DIR="$HOME/.config/ModTheSpire"
    mkdir -p "$CONFIG_DIR/CommunicationMod"
    mkdir -p "$CONFIG_DIR/SuperFastMode"

    EVAL_CMD="python3 $SCRIPTS_DIR/eval_model.py --model $PROJECT_DIR/$MODEL --games $GAMES --policy $POLICY --seed-file $PROJECT_DIR/$SEED_FILE --run-tag $RUN_TAG --restart-every $RESTART_EVERY --resume-run"

    cat > "$CONFIG_DIR/CommunicationMod/config.properties" << EOF
command=$EVAL_CMD
runAtGameStart=true
EOF

    cat > "$CONFIG_DIR/SuperFastMode/SuperFastModeConfig.properties" << EOF
isDeltaMultiplied=true
deltaMultiplier=4.999997
isInstantLerp=true
EOF

    echo "Starting eval (single instance, $GAMES games)..."
    echo "Monitor: tail -f $PROJECT_DIR/logs/eval_debug.log"

    while true; do
        cd "$GAME_DIR"
        # Large heap: a long eval runs many games per JVM and 512m OOMs after
        # ~35 games (silently restarting mid-run). 8 GB gives ample margin
        # for a full 200-game eval; the VM has 86 GB RAM.
        java -Xmx8192m -Xms512m \
            -Dorg.lwjgl.openal.libname=/usr/lib/x86_64-linux-gnu/libopenal.so.1 \
            -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true \
            -Djava.io.tmpdir=/tmp/sts_eval \
            -jar ModTheSpire.jar \
            --skip-launcher \
            --mods basemod,CommunicationMod,superfastmode \
            >> "$PROJECT_DIR/logs/eval_1.log" 2>&1 || true

        # Check if eval is done by looking at the CSV
        COMPLETED=$(python3 -c "
import csv, os
csv_path = '$PROJECT_DIR/logs/eval_stats.csv'
if not os.path.exists(csv_path):
    print(0)
else:
    count = sum(1 for r in csv.DictReader(open(csv_path)) if r.get('run') == '$RUN_TAG')
    print(count)
" 2>/dev/null)

        if [ "$COMPLETED" -ge "$GAMES" ]; then
            echo "Eval complete: $COMPLETED/$GAMES games"
            break
        fi
        echo "Eval progress: $COMPLETED/$GAMES games — restarting instance..."
        sleep 3
    done

else
    # Multi-instance: split games across instances
    GAMES_PER_INSTANCE=$(( (GAMES + INSTANCES - 1) / INSTANCES ))
    echo "Splitting $GAMES games across $INSTANCES instances ($GAMES_PER_INSTANCE each)"
    echo "NOTE: Multi-instance eval uses different run tags per instance."
    echo ""

    PIDS=()
    for i in $(seq 1 $INSTANCES); do
        INSTANCE_TAG="${RUN_TAG}_part${i}"
        INSTANCE_DIR="$PROJECT_DIR/instances/eval_$i"
        CONFIG_DIR="$INSTANCE_DIR/config/ModTheSpire"
        mkdir -p "$CONFIG_DIR/CommunicationMod"
        mkdir -p "$CONFIG_DIR/SuperFastMode"

        EVAL_CMD="python3 $SCRIPTS_DIR/eval_model.py --model $PROJECT_DIR/$MODEL --games $GAMES_PER_INSTANCE --policy $POLICY --seed-file $PROJECT_DIR/$SEED_FILE --run-tag $INSTANCE_TAG --restart-every $RESTART_EVERY --resume-run"

        cat > "$CONFIG_DIR/CommunicationMod/config.properties" << EOF
command=$EVAL_CMD
runAtGameStart=true
EOF

        cat > "$CONFIG_DIR/SuperFastMode/SuperFastModeConfig.properties" << EOF
isDeltaMultiplied=true
deltaMultiplier=4.999997
isInstantLerp=true
EOF

        (
            while true; do
                cd "$GAME_DIR"
                # Large heap (see single-instance note above): avoid mid-eval OOM.
                XDG_CONFIG_HOME="$INSTANCE_DIR/config" \
                    java -Xmx8192m -Xms512m \
                    -Dorg.lwjgl.openal.libname=/usr/lib/x86_64-linux-gnu/libopenal.so.1 \
                    -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true \
                    -jar ModTheSpire.jar \
                    --skip-launcher \
                    --mods basemod,CommunicationMod,superfastmode \
                    >> "$PROJECT_DIR/logs/eval_${i}.log" 2>&1 || true

                COMPLETED=$(python3 -c "
import csv, os
csv_path = '$PROJECT_DIR/logs/eval_stats.csv'
if not os.path.exists(csv_path):
    print(0)
else:
    count = sum(1 for r in csv.DictReader(open(csv_path)) if r.get('run') == '$INSTANCE_TAG')
    print(count)
" 2>/dev/null)
                [ "$COMPLETED" -ge "$GAMES_PER_INSTANCE" ] && break
                sleep 3
            done
        ) &
        PIDS+=($!)
        sleep 2
    done

    echo "All eval instances launched. Waiting for completion..."
    wait "${PIDS[@]}"
    echo "Eval complete."
fi

echo ""
echo "Results in: $PROJECT_DIR/logs/eval_stats.csv"
echo "Pull to local: vm/sync.sh pull"
