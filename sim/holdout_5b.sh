#!/bin/bash
cd "$HOME/sts"
SEED0=20011 SEEDSTEP=104729 PAR=4 ./run_honest.sh holdout_rl5b_K64 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5b/pol_gate.npz TRACE=1" "holdout-150: round 5b (critic PPO, it149) + honest K=64"
python3 ledger.py ingest >/dev/null
{ python3 ledger.py compare holdout_K64 holdout_rl5b_K64 | sed -n 1,20p; python3 ledger.py compare holdout_rl5_K64 holdout_rl5b_K64 | sed -n 4,6p; } | tee sim_logs/GATE_holdout_rl5b_K64.txt
