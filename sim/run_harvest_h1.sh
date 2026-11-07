#!/bin/bash
# Phase 3 step 1a: honest data harvest with mainline_h1 (1-ply C++ K=64) on TRAINING seeds
# 10007 + i*7919, i = 150..1749 (eval set is i < 150). Resumable (finished seeds skipped).
cd "$HOME/sts"
export SEED0=1197857 SEEDSTEP=7919
nice -n 5 ./run_honest.sh harvest_h1 1600 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 HARVEST=sim_logs/harvest/h1" "Phase 3 harvest: mainline_h1 (K=64) on training seeds i=150..1749"
.venv/bin/python harvest_status.py sim_logs/harvest/h1 | tee sim_logs/harvest/h1/STATUS.txt
echo HARVEST_DONE
