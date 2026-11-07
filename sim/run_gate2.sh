#!/bin/bash
# Phase 2 gate: confirm on the 150-seed HOLDOUT (20011 + i*104729), paired against the honest K=8 baseline.
cd "$HOME/sts"
export SEED0=20011 SEEDSTEP=104729
./run_honest.sh holdout_K8 150 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8" "GATE holdout-150: honest K=8 baseline"
./run_honest.sh holdout_K32 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=32" "GATE holdout-150: honest K=32 (chosen engine)"
./run_honest.sh holdout_K64 150 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64" "GATE holdout-150: honest K=64 (cost check)"
python3 ledger.py compare holdout_K8 holdout_K32 > sim_logs/honest/GATE_phase2.txt
echo >> sim_logs/honest/GATE_phase2.txt
python3 ledger.py compare holdout_K8 holdout_K64 >> sim_logs/honest/GATE_phase2.txt
echo GATE2_DONE
