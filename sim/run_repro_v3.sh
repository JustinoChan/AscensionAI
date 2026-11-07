#!/bin/bash
# 150-seed mainline_v3 reproduction (mirrors run_vdraft_ab3.sh arm A; -P 14, 1800s kill timeout)
set -u
cd "$HOME/sts"
export PY="$HOME/sts/.venv/bin/python" OUTD=sim_logs/repro_v3
mkdir -p "$OUTD"
date +%s > "$OUTD/.start"
seq 0 149 | xargs -P 14 -I{} bash -c '
  seed=$(( 10007 + {} * 7919 )); out="$OUTD/seed_$seed.json"
  timeout -s KILL 1800 "$PY" eval_fullrun_analytics_v2.py $seed macro > "$out" 2>/dev/null \
    || echo "{\"seed\":$seed,\"win\":0,\"floor\":0,\"timeout\":1}" > "$out"
'
date +%s > "$OUTD/.end"
"$PY" reduce_draftsearch.py "$OUTD" > "$OUTD/SUMMARY.txt" 2>&1
python3 - <<'PY' >> "$OUTD/SUMMARY.txt"
import json,glob
n=w=f=eq=0
for p in glob.glob('sim_logs/repro_v3/seed_*.json'):
    a=json.load(open(p)); r=json.load(open(p.replace('repro_v3','vdraft3_A')))
    n+=1; w+=a.get('win',0); f+=a.get('floor',0); eq+= (a==r)
print(f"n={n} wins={w} avg_floor={f/n:.2f} exact_json_match_vs_vdraft3_A={eq}/{n}")
PY
