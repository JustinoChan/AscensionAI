#!/bin/bash
# Collect fresh heuristic BC demos for the PPO BC anchor — headless, no training.
#
#   ./collect_bc.sh --workers 4 --minutes 25 --games-per-cycle 15
#
# Runs behavior_clone.py --collect-only through N headless STS instances,
# writing 717-d demo .npz files into bc_demos_shared/. It does NOT train any
# model — it only gathers (observation, heuristic action, mask) tuples that
# train_offline.py loads as the BC anchor. Mirrors run_training.sh's per-worker
# Xvfb + tmpdir + log-staleness watchdog.
set -e

WORKERS=4
MINUTES=25
GAMES_PER_CYCLE=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)         WORKERS="$2"; shift 2;;
        --minutes)         MINUTES="$2"; shift 2;;
        --games-per-cycle) GAMES_PER_CYCLE="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

PROJECT_DIR="$HOME/ascension"
GAME_DIR="$PROJECT_DIR/game"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
VENV="$PROJECT_DIR/.venv/bin/activate"
DEMO_DIR="$PROJECT_DIR/bc_demos_shared"
STOP_FILE="$PROJECT_DIR/logs/.stop_bc_collect"

source "$VENV"
rm -f "$STOP_FILE"
# Start from a clean demo dir so no stale-dimension demos pollute the anchor.
mkdir -p "$DEMO_DIR" "$PROJECT_DIR/logs"
rm -f "$DEMO_DIR"/*.npz

mkdir -p "$GAME_DIR/mods"
for jar in BaseMod.jar CommunicationMod.jar SuperFastMode.jar; do
    [ -f "$GAME_DIR/$jar" ] && ln -sf "$GAME_DIR/$jar" "$GAME_DIR/mods/$jar"
done

END_TIME=$(($(date +%s) + MINUTES * 60))
export LIBGL_ALWAYS_SOFTWARE=1

# CommunicationMod runs the heuristic collector. Each finished cycle of
# GAMES_PER_CYCLE games saves a timestamped demo file into bc_demos_shared/.
COMMMOD_DIR="$HOME/.config/ModTheSpire/CommunicationMod"
SFM_DIR="$HOME/.config/ModTheSpire/SuperFastMode"
mkdir -p "$COMMMOD_DIR" "$SFM_DIR"
cat > "$COMMMOD_DIR/config.properties" << EOF
command=python3 $SCRIPTS_DIR/behavior_clone.py --collect-only --demo-dir $DEMO_DIR --games $GAMES_PER_CYCLE
runAtGameStart=true
EOF
cat > "$SFM_DIR/SuperFastModeConfig.properties" << EOF
isDeltaMultiplied=true
deltaMultiplier=4.999997
isInstantLerp=true
EOF

launch_collector() {
    local id=$1
    local log_file="$PROJECT_DIR/logs/bc_collect_${id}.log"
    local tmpdir="/tmp/bc_worker_${id}"
    local display_num=$((120 + id))
    local stale_limit=120 boot_grace=150
    mkdir -p "$tmpdir"
    Xvfb :$display_num -screen 0 320x240x16 -ac +extension GLX +extension RANDR &>/dev/null &
    local xvfb_pid=$!
    export DISPLAY=:$display_num
    sleep 1

    while [ $(date +%s) -lt $END_TIME ] && [ ! -f "$STOP_FILE" ]; do
        echo "[$(date '+%H:%M:%S')] Collector $id: starting STS (display :$display_num)"
        cd "$GAME_DIR"
        DISPLAY=:$display_num java -Xmx2048m -Xms512m \
            -Dorg.lwjgl.openal.libname=/usr/lib/x86_64-linux-gnu/libopenal.so.1 \
            -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true \
            -Djava.io.tmpdir="$tmpdir" \
            -jar ModTheSpire.jar --skip-launcher \
            --mods basemod,CommunicationMod,superfastmode \
            >> "$log_file" 2>&1 &
        local java_pid=$!
        local launch_ts=$(date +%s)
        while kill -0 "$java_pid" 2>/dev/null; do
            sleep 20
            if [ $(date +%s) -ge $END_TIME ] || [ -f "$STOP_FILE" ]; then
                kill "$java_pid" 2>/dev/null || true; sleep 2; kill -9 "$java_pid" 2>/dev/null || true; break
            fi
            local now=$(date +%s)
            [ $((now - launch_ts)) -lt $boot_grace ] && continue
            local mtime=$(stat -c %Y "$log_file" 2>/dev/null || echo "$now")
            if [ $((now - mtime)) -gt $stale_limit ]; then
                echo "[$(date '+%H:%M:%S')] Collector $id: WEDGED — killing JVM to relaunch"
                kill -9 "$java_pid" 2>/dev/null || true; break
            fi
        done
        wait "$java_pid" 2>/dev/null || true
        sleep 2
    done
    echo "[$(date '+%H:%M:%S')] Collector $id: time limit reached"
    kill "$java_pid" 2>/dev/null || true
    kill "$xvfb_pid" 2>/dev/null || true
}

echo "=== BC demo collection: $WORKERS workers, ${MINUTES}min, $GAMES_PER_CYCLE games/cycle -> $DEMO_DIR ==="
for i in $(seq 1 $WORKERS); do
    launch_collector "$i" &
    sleep 2
done
wait
echo ""
echo "=== Collection complete ==="
echo "Demo files: $(ls -1 "$DEMO_DIR"/*.npz 2>/dev/null | wc -l)"
