#!/bin/bash
# holdout-150 of an RL policy at honest K=64.  env: POL=<npz> ARM=<name> EXTRA_ENV="..."
cd "$HOME/sts"
grep -q "^n=150 " sim_logs/honest/$ARM/SUMMARY.txt 2>/dev/null || \
  SEED0=20011 SEEDSTEP=104729 PAR=14 ./run_honest.sh $ARM 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_POLICY=$POL TRACE=1${EXTRA_ENV:+ $EXTRA_ENV}" "holdout-150: $POL + honest K=64"
python3 ledger.py ingest >/dev/null
{ python3 ledger.py compare holdout_K64 $ARM | sed -n 1,20p; python3 ledger.py compare holdout_rl_K64 $ARM | sed -n 4,6p; } | tee sim_logs/GATE_$ARM.txt
