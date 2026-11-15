#!/bin/bash
# POTION_VALUE sweep on the first 150 eval500 seeds with the best policy (5d), then extend the best value to all 500.
cd "$HOME/sts"
E="HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz TRACE=1"
for v in 5 10 20; do
  grep -q "^n=150 \|^n=500 " sim_logs/honest/eval500_rl5d_pv$v/SUMMARY.txt 2>/dev/null || \
    SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl5d_pv$v 150 7200 "$E POTION_VALUE=$v" "5d + POTION_VALUE=$v (sweep, first 150 eval500 seeds)"
done
python3 ledger.py ingest > /dev/null
BEST=$(python3 - <<'PY'
import sqlite3
c = sqlite3.connect("sim_logs/results.db")
base = {r[0]: (r[1], r[2]) for r in c.execute("select seed, win, floor from seeds where run='eval500_rl5d_K64'")}
best, bk = None, None
for v in (5, 10, 20):
    rows = c.execute("select seed, win, floor from seeds where run=?", ("eval500_rl5d_pv%d" % v,)).fetchall()
    w = sum(r[1] for r in rows); bw = sum(base[r[0]][0] for r in rows); f = sum(r[2] for r in rows) / max(1, len(rows))
    print("PV=%d: %d wins vs 5d %d on the same %d seeds, floor %.2f" % (v, w, bw, len(rows), f), flush=True)
    if bk is None or (w, f) > bk:
        best, bk = v, (w, f)
print(best)
PY
)
echo "$BEST" | tee sim_logs/POTVAL_SWEEP.txt
V=$(echo "$BEST" | tail -1)
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl5d_pv$V 500 7200 "$E POTION_VALUE=$V" "5d + POTION_VALUE=$V (best of sweep), full eval500"
python3 ledger.py ingest > /dev/null
{ echo "best POTION_VALUE=$V"; python3 ledger.py compare eval500_rl5d_K64 eval500_rl5d_pv$V | sed -n 1,20p
  python3 ledger.py compare eval500_rl5d_potsave eval500_rl5d_pv$V | sed -n 4,6p
  python3 boss_report.py eval500_rl5d_K64 eval500_rl5d_pv$V | grep -E "fight|BOSS"; } | tee sim_logs/GATE_potval_eval500.txt
