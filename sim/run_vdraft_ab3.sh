#!/bin/bash
# A/B over the 150 eval seeds: A = mainline_v3, B = +V_DRAFT (draft_v3.npz, enriched).
set -u
PY="$HOME/sts_gym_spike/.venv/bin/python"
cd "$HOME/sts_gym_spike"
run() {  # $1=arm_env  $2=outdir
  mkdir -p "$2"; export ARMENV="$1" OUTD="$2" PY
  seq 0 149 | xargs -P 12 -I{} bash -c '
    seed=$(( 10007 + {} * 7919 ))
    out="$OUTD/seed_$seed.json"
    [ -s "$out" ] && exit 0
    env $ARMENV timeout -s KILL 600 "$PY" eval_fullrun_analytics_v2.py $seed macro \
      > "$out" 2>/dev/null \
      || echo "{\"seed\":$seed,\"win\":0,\"floor\":0,\"timeout\":1}" > "$out"
  '
}
run "" sim_logs/vdraft3_A
run "V_DRAFT=1 DRAFT_V_NPZ=draft_v3.npz" sim_logs/vdraft3_B
echo "=== A (mainline_v3)   ==="; "$PY" reduce_draftsearch.py sim_logs/vdraft3_A
echo "=== B (+V_DRAFT v3)   ==="; "$PY" reduce_draftsearch.py sim_logs/vdraft3_B
