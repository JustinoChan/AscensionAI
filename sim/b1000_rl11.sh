#!/bin/bash
# 1000-seed paired test: round 11 deck policy + mainline combat vs mainline (c5roll: 216 + 209 = 425/1000).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_rl11_main 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1" "round 11 + mainline combat, eval500b"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500b_prior_c5roll eval500b_rl11_main | sed -n 3,6p; python3 b1000.py prior_c5roll rl11_main; } | tee sim_logs/B1000_rl11.txt
