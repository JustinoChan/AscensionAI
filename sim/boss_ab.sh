#!/bin/bash
# Boss-budget A/B on eval500: pick the better RL policy (rl2 gate vs it099 by eval500 wins, then floor; tie -> it099),
# run it with HONEST_K_BOSS=256 HONEST_K_ELITE=128 (K=64 elsewhere), paired vs the same policy at plain K=64.
cd "$HOME/sts"
python3 ledger.py ingest > /dev/null
read BASEARM POL <<< $(python3 - <<'PY'
import sqlite3
c = sqlite3.connect("sim_logs/results.db")
def s(a):
    r = c.execute("select count(*), sum(win), avg(floor) from seeds where run=?", (a,)).fetchone()
    return (r[1] or 0, r[2] or 0) if r[0] == 500 else (-1, 0)
a, b = s("eval500_rl_K64"), s("eval500_rl2_K64")
print("eval500_rl2_K64 nets/rl2/pol_gate.npz" if b > a and b[0] >= a[0] else "eval500_rl_K64 nets/rl/pol_gate.npz")
PY
)
OUT=sim_logs/GATE_boss_budget.txt
echo "policy: $POL (plain-K64 arm $BASEARM); boss budget: HONEST_K_BOSS=256 HONEST_K_ELITE=128, K=64 elsewhere" | tee $OUT
grep -q "^n=500 " sim_logs/honest/eval500_rlboss_K64/SUMMARY.txt 2>/dev/null || \
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rlboss_K64 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 HONEST_K_BOSS=256 HONEST_K_ELITE=128 MACRO_POLICY=$POL TRACE=1" "boss-budget A/B: $POL, K_BOSS=256 K_ELITE=128, K=64 elsewhere"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare $BASEARM eval500_rlboss_K64 | sed -n 1,20p; python3 boss_report.py $BASEARM eval500_rlboss_K64; } | tee -a $OUT
