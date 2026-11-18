#!/bin/bash
# Policy ensemble (mean candidate logits) under the mainline (PV10): members 5d + round 7 (+ round 8 if its eval500 >= 192).
# eval500 vs 5d+PV10 (192/500), then holdout-150 of round 7 + PV10 and of the ensemble vs mainline 63/150.
cd "$HOME/sts"
python3 ledger.py ingest > /dev/null
W8=$(python3 -c "
import sqlite3; c=sqlite3.connect('sim_logs/results.db'); n,w=c.execute(\"select count(*),sum(win) from seeds where run='eval500_rl8_pv10'\").fetchone(); print(w if n==500 else -1)")
ENS=nets/rl7/pol_gate.npz; TAG=ens57
[ "$W8" -ge 192 ] && { ENS=$ENS,nets/rl8/pol_gate.npz; TAG=ens578; }
echo "ensemble $TAG: nets/rl5d/pol_gate.npz + $ENS (round 8 eval500 wins: $W8)" | tee sim_logs/GATE_$TAG.txt
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_${TAG}_pv10 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz MACRO_POLICY_ENS=$ENS POTION_VALUE=10 TRACE=1" "ensemble $TAG + PV10, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_${TAG}_pv10 | sed -n 1,8p; python3 ledger.py compare eval500_rl7_pv10 eval500_${TAG}_pv10 | sed -n 4,6p; } | tee -a sim_logs/GATE_$TAG.txt
POL=nets/rl7/pol_gate.npz ARM=holdout_rl7_pv10 EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10" ./holdout_pol.sh
python3 ledger.py compare holdout_rl5d_pv10 holdout_rl7_pv10 | sed -n 4,6p | tee -a sim_logs/GATE_$TAG.txt
POL=nets/rl5d/pol_gate.npz ARM=holdout_${TAG}_pv10 EXTRA_ENV="MACRO_MAPFEAT=1 MACRO_POLICY_ENS=$ENS POTION_VALUE=10" ./holdout_pol.sh
python3 ledger.py compare holdout_rl5d_pv10 holdout_${TAG}_pv10 | sed -n 4,6p | tee -a sim_logs/GATE_$TAG.txt
