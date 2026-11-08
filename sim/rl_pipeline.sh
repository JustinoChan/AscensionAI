#!/bin/bash
# Phase 3 step 3.2 pipeline: BC data (1000 K=8 runs logging honest candidate features) -> BC -> PPO (10 workers).
cd "$HOME/sts"
grep -q "^n=1000 " sim_logs/honest/rl_bcdata/SUMMARY.txt 2>/dev/null || \
  SEED0=800000011 SEEDSTEP=19 PAR=${PAR_BC:-10} ./run_honest.sh rl_bcdata 1000 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 MACRO_FEAT_LOG=sim_logs/rl/bcdata" "Phase 3 3.2 BC data: mainline macro + honest candidate features, K=8, bc seeds 800000011+i*19"
[ -f nets/rl/pol_bc.npz ] || nice -n 5 .venv_train/bin/python rl_macro_train.py bc > nets/rl_bc.log 2>&1
tail -3 nets/rl_bc.log
RL_WORKERS=${RL_WORKERS:-10} nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${RL_ITERS:-150} >> nets/rl/rl.log 2>&1
tail -5 nets/rl/rl.log
