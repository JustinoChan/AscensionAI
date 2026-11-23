#!/bin/bash
# Under the new mainline (5d + PV10 + exi1 root prior c=5 + exi1 play-outs): (1) holdout-150 baseline for future gates;
# (2) re-tune POTION_VALUE (5 / 20 vs 10) on the first 150 eval500 seeds, since combat got stronger; a value with more
# wins than PV10 on those seeds is extended to the full eval500 vs 216.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; X="MACRO_MAPFEAT=1 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1"
POL=nets/rl5d/pol_gate.npz ARM=holdout_main2 EXTRA_ENV="$X POTION_VALUE=10" ./holdout_pol.sh
python3 ledger.py compare holdout_rl5d_pv10 holdout_main2 | sed -n 3,6p | tee sim_logs/MAIN2_CHECKS.txt
E="HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_POLICY=nets/rl5d/pol_gate.npz $X TRACE=1"
for v in 5 20; do
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_main2_pv$v 150 7200 "$E POTION_VALUE=$v" "new mainline with POTION_VALUE=$v (first 150 eval500 seeds)"
done
python3 ledger.py ingest > /dev/null
BEST=$(python3 - <<'PY'
import sqlite3
c = sqlite3.connect("sim_logs/results.db")
base = dict(c.execute("select seed, win from seeds where run='eval500_prior_c5roll'").fetchall())
best, bw = 10, None
for v in (5, 20):
    rows = dict(c.execute("select seed, win from seeds where run=?", ("eval500_main2_pv%d" % v,)).fetchall())
    w = sum(rows.values()); b = sum(base[s] for s in rows)
    print("PV=%d: %d wins vs PV10 %d on the same %d seeds" % (v, w, b, len(rows)))
    if w > b and (bw is None or w - b > bw):
        best, bw = v, w - b
print(best)
PY
)
echo "$BEST" | tee -a sim_logs/MAIN2_CHECKS.txt
V=$(echo "$BEST" | tail -1); [ "$V" = 10 ] && exit 0
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_main2_pv$V 500 7200 "$E POTION_VALUE=$V" "new mainline with POTION_VALUE=$V, full eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_prior_c5roll eval500_main2_pv$V | sed -n 1,8p | tee -a sim_logs/MAIN2_CHECKS.txt
