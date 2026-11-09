#!/bin/bash
# holdout-150 confirmation of the gated RL out-of-combat policy (nets/rl/pol_gate.npz = it099), honest K=64.
cd "$HOME/sts"
grep -q "^n=150 " sim_logs/honest/holdout_rl_K64/SUMMARY.txt 2>/dev/null || \
  SEED0=20011 SEEDSTEP=104729 PAR=14 ./run_honest.sh holdout_rl_K64 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_POLICY=nets/rl/pol_gate.npz TRACE=1" "holdout-150 confirmation: RL out-of-combat policy it099 + honest K=64"
python3 ledger.py ingest >/dev/null; python3 ledger.py compare holdout_K64 holdout_rl_K64 | tee sim_logs/GATE_rl_holdout.txt
