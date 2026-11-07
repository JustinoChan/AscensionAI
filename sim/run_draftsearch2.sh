#!/bin/bash
# Resume-safe draft-teacher runner: skips seeds whose output already exists
# non-empty, so a spot-preemption relaunch continues where it left off.
# QLAND/QLAND_DIR pass through from the caller's environment.
# Usage: run_draftsearch2.sh <i_start> <i_count> <outdir> [DRAFT_LOG=1] [TIMEOUT=2700]
set -u
I0=$1; N=$2; OUT=$3; LOG=${4:-1}; TO=${5:-2700}
PY="$HOME/sts_gym_spike/.venv/bin/python"
cd "$HOME/sts_gym_spike"
mkdir -p "$OUT"
export PY OUT LOG TO I0
seq 0 $((N - 1)) | xargs -P 12 -I{} bash -c '
  i=$(( I0 + {} ))
  seed=$(( 10007 + i * 7919 ))
  out="$OUT/seed_$seed.json"
  [ -s "$out" ] && exit 0
  DRAFT_SEARCH=1 DRAFT_LOG=$LOG timeout -s KILL $TO \
    "$PY" eval_fullrun_analytics_v2.py $seed macro > "$out" 2>/dev/null \
    || echo "{\"seed\":$seed,\"win\":0,\"floor\":0,\"timeout\":1}" > "$out"
'
