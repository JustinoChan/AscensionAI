#!/bin/bash
# holdout-150 baseline for the round 11 mainline (previous mainline c5roll: 70/150).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
POL=nets/rl11/pol_gate.npz ARM=holdout_rl11_main EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1" ./holdout_pol.sh
python3 ledger.py compare holdout_main2 holdout_rl11_main | sed -n 3,6p | tee sim_logs/HOLDOUT_rl11.txt
