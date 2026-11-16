#!/bin/bash
# best policy (round 6 if it beat 5d on eval500 wins, else 5d) + POTION_VALUE=10, eval500 vs 5d+PV10 / 5d; then holdout-150.
cd "$HOME/sts"
python3 ledger.py ingest > /dev/null
read POL EXTRA TAG <<< $(python3 - <<'PY'
import sqlite3
c = sqlite3.connect("sim_logs/results.db")
def w(a):
    n, x = c.execute("select count(*), sum(win) from seeds where run=?", (a,)).fetchone()
    return x if n == 500 else -1
if w("eval500_rl6_K64") > w("eval500_rl5d_K64"):
    print("nets/rl6/pol_gate.npz MACRO_MAPFEAT=1,MACRO_ENCFEAT=1 rl6")
else:
    print("nets/rl5d/pol_gate.npz MACRO_MAPFEAT=1 rl5d")
PY
)
EX=$(echo $EXTRA | tr ',' ' ')
echo "combo: $TAG + POTION_VALUE=10" | tee sim_logs/COMBO_PV.txt
if [ "$TAG" = rl6 ]; then
  grep -q "^n=500 " sim_logs/honest/eval500_rl6_pv10/SUMMARY.txt 2>/dev/null || \
    SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl6_pv10 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $EX MACRO_POLICY=$POL POTION_VALUE=10 TRACE=1" "round 6 + POTION_VALUE=10, eval500"
  python3 ledger.py ingest > /dev/null
  { python3 ledger.py compare eval500_rl5d_pv10 eval500_rl6_pv10 | sed -n 1,8p; python3 ledger.py compare eval500_rl6_K64 eval500_rl6_pv10 | sed -n 4,6p; } | tee -a sim_logs/COMBO_PV.txt
  ARM=holdout_rl6_pv10
else
  ARM=holdout_rl5d_pv10
fi
POL=$POL ARM=$ARM EXTRA_ENV="$EX POTION_VALUE=10" ./holdout_pol.sh
python3 ledger.py compare holdout_rl5d_K64 $ARM | sed -n 4,6p | tee -a sim_logs/COMBO_PV.txt
