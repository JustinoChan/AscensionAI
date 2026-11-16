#!/bin/bash
# Is POTION_VALUE=10 on 5d real? holdout-150 (vs 5d 57/150), then a fresh 500-seed set (eval500b) for 5d and 5d+PV10,
# so the paired test has n=1000 with eval500.
cd "$HOME/sts"
B="HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz TRACE=1"
POL=nets/rl5d/pol_gate.npz ARM=holdout_rl5d_pv10 EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10" ./holdout_pol.sh
python3 ledger.py compare holdout_rl5d_K64 holdout_rl5d_pv10 | sed -n 1,8p | tee sim_logs/PV10_CONFIRM.txt
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_rl5d_pv10 500 7200 "$B POTION_VALUE=10" "5d + POTION_VALUE=10, eval500b (fresh seeds)"
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_rl5d_K64 500 7200 "$B" "5d, eval500b (fresh seeds)"
python3 ledger.py ingest > /dev/null
python3 ledger.py compare eval500b_rl5d_K64 eval500b_rl5d_pv10 | sed -n 1,8p | tee -a sim_logs/PV10_CONFIRM.txt
