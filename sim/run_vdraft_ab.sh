#!/bin/bash
# A/B over the 150 eval seeds: A = mainline_v3 (V_DRAFT off), B = +V_DRAFT.
set -u
PY="$HOME/sts_gym_spike/.venv/bin/python"
cd "$HOME/sts_gym_spike"
run() {  # $1=arm_env  $2=outdir
  mkdir -p "$2"; export ARMENV="$1" OUTD="$2" PY
  seq 0 149 | xargs -P 12 -I{} bash -c '
    seed=$(( 10007 + {} * 7919 ))
    env $ARMENV timeout -s KILL 600 "$PY" eval_fullrun_analytics_v2.py $seed macro \
      > "$OUTD/seed_$seed.json" 2>/dev/null \
      || echo "{\"seed\":$seed,\"win\":0,\"floor\":0,\"timeout\":1}" > "$OUTD/seed_$seed.json"
  '
}
rm -rf sim_logs/vdraft_A sim_logs/vdraft_B
run "" sim_logs/vdraft_A
run "V_DRAFT=1" sim_logs/vdraft_B
echo "=== A (mainline_v3) ==="; "$PY" reduce_draftsearch.py sim_logs/vdraft_A
echo "=== B (+V_DRAFT)   ==="; "$PY" reduce_draftsearch.py sim_logs/vdraft_B
