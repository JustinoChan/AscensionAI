#!/usr/bin/env bash
# ============================================================================
# DOES NOT WORK -- kept as a documented dead end. DO NOT USE for real evals.
#
# Intent: parallel greedy eval by giving each instance its own HOME so
# CommunicationMod would read a per-instance ~/.config (different seed subset +
# run-tag), letting N instances split a 200-game eval.
#
# Why it fails: CommunicationMod resolves its config from the *real* user home
# (Java user.home / hard-coded ~/.config), NOT the HOME env var. So every
# instance reads /home/<user>/.config and runs the SAME command, duplicating
# one eval N times (deduped on write -> no speedup, wasted CPU). Per-instance
# HOME was verified set in /proc/<pid>/environ, yet rows still landed under the
# real-home config's run-tag. Training parallelizes only because all workers
# intentionally run the *identical* command; eval needs distinct seed subsets,
# which this architecture cannot deliver. Eval is therefore single-stream
# (see vm/run_eval.sh --instances 1).
#
# Original design notes (the parts that ARE sound): all instances append to the
# shared logs/eval_stats.csv safely (atomic <4KB O_APPEND, dedup keyed on
# (run_tag, game)); the seed split and per-Xvfb display setup work fine.
# ============================================================================
set -u

PROJECT_DIR="$HOME/ascension"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
GAME_DIR="$PROJECT_DIR/game"
MODEL="models/ppo_sts.pt"
SEED_FILE="$PROJECT_DIR/seeds/eval_200.txt"
RUN_PREFIX="${RUN_PREFIX:-deckvec717p}"
INSTANCES="${1:-8}"

source "$PROJECT_DIR/.venv/bin/activate"

# 1. Round-robin split seeds into INSTANCES chunks (balances seed difficulty).
mkdir -p /tmp/eval_seeds
python3 - "$SEED_FILE" "$INSTANCES" <<'PY'
import sys, os
seed_file, k = sys.argv[1], int(sys.argv[2])
seeds = [l.strip() for l in open(seed_file)
         if l.strip() and not l.lstrip().startswith("#")]
os.makedirs("/tmp/eval_seeds", exist_ok=True)
chunks = [[] for _ in range(k)]
for i, s in enumerate(seeds):
    chunks[i % k].append(s)
for i, c in enumerate(chunks, 1):
    with open(f"/tmp/eval_seeds/chunk_{i}.txt", "w") as f:
        f.write("\n".join(c) + "\n")
    print(f"chunk_{i}: {len(c)} seeds")
print("total seeds:", len(seeds))
PY

export LIBGL_ALWAYS_SOFTWARE=1
PIDS=()
for i in $(seq 1 "$INSTANCES"); do
    HOMEDIR="/tmp/evalhome_$i"
    CONFIG_DIR="$HOMEDIR/.config/ModTheSpire"
    mkdir -p "$CONFIG_DIR/CommunicationMod" "$CONFIG_DIR/SuperFastMode"
    CHUNK="/tmp/eval_seeds/chunk_$i.txt"
    NGAMES=$(grep -c . "$CHUNK")
    RUNTAG="${RUN_PREFIX}_$i"

    cat > "$CONFIG_DIR/CommunicationMod/config.properties" <<EOF
command=python3 $SCRIPTS_DIR/eval_model.py --model $PROJECT_DIR/$MODEL --games $NGAMES --policy model --seed-file $CHUNK --run-tag $RUNTAG --restart-every 25 --resume-run --log-file $PROJECT_DIR/logs/eval_p$i.log
runAtGameStart=true
EOF
    cat > "$CONFIG_DIR/SuperFastMode/SuperFastModeConfig.properties" <<EOF
isDeltaMultiplied=true
deltaMultiplier=4.999997
isInstantLerp=true
EOF

    DISP=$((160 + i))
    TMPD="/tmp/eval_tmp_$i"
    mkdir -p "$TMPD"
    Xvfb ":$DISP" -screen 0 320x240x16 -ac +extension GLX +extension RANDR >/dev/null 2>&1 &
    sleep 1

    (
        export HOME="$HOMEDIR"
        export DISPLAY=":$DISP"
        cd "$GAME_DIR"
        while true; do
            java -Xmx4096m -Xms512m \
                -Dorg.lwjgl.openal.libname=/usr/lib/x86_64-linux-gnu/libopenal.so.1 \
                -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true \
                -Djava.io.tmpdir="$TMPD" \
                -jar ModTheSpire.jar --skip-launcher \
                --mods basemod,CommunicationMod,superfastmode \
                >> "$PROJECT_DIR/logs/eval_jvm_$i.log" 2>&1 || true
            DONE=$(python3 -c "import csv,os;p='$PROJECT_DIR/logs/eval_stats.csv';print(sum(1 for r in csv.DictReader(open(p)) if r.get('run')=='$RUNTAG') if os.path.exists(p) else 0)" 2>/dev/null)
            [ "${DONE:-0}" -ge "$NGAMES" ] && break
            sleep 3
        done
    ) &
    PIDS+=($!)
    sleep 2
done

echo "launched $INSTANCES eval instances (pids ${PIDS[*]})"
wait "${PIDS[@]}"
echo "ALL EVAL INSTANCES DONE"
