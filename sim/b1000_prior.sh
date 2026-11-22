#!/bin/bash
# 1000-seed paired test for the root policy prior (c=5 + exi1 play-outs) vs mainline 5d+PV10 (383/1000): eval500b run.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_prior_c5roll 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1" "root prior c5roll, eval500b (fresh seeds)"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500b_rl5d_pv10 eval500b_prior_c5roll | sed -n 3,6p; python3 b1000.py rl5d_pv10 prior_c5roll; } | tee sim_logs/B1000_prior.txt
