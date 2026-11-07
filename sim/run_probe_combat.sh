#!/bin/bash
# Phase-0 probe: full runs, mainline_v3 combat search vs student Q-net.
set -u
PY="$HOME/sts_gym_spike/.venv/bin/python"
cd "$HOME/sts_gym_spike"
run() {
  mkdir -p "$2"; export ARMENV="$1" OUTD="$2" PY
  seq 150 199 | xargs -P 12 -I{} bash -c '
    seed=$(( 10007 + {} * 7919 ))
    out="$OUTD/seed_$seed.json"
    [ -s "$out" ] && exit 0
    env $ARMENV timeout -s KILL 900 "$PY" eval_fullrun_analytics_v2.py $seed macro \
      > "$out" 2>/dev/null \
      || echo "{\"seed\":$seed,\"win\":0,\"floor\":0,\"timeout\":1}" > "$out"
  '
}
run "" sim_logs/probe_search
run "COMBAT_NET=combat_q_student.npz" sim_logs/probe_student
"$PY" probe_report.py sim_logs/probe_search sim_logs/probe_student
