#!/bin/bash
# Phase 2 queue A: engines at 150 seeds (starts after the potion audit finishes).
cd "$HOME/sts"
while [ ! -f sim_logs/honest/potion_audit_K8/SUMMARY.txt ]; do sleep 20; done
R=./run_honest.sh
$R cpp_1ply_K8 150 1800 "HONEST=1 HONEST_K=8 HONEST_CPP=1" "1-ply honest search K=8, native batched C++ rollouts (should equal combat_K8)"
$R mcts_K8_S700 150 1800 "HONEST=1 HONEST_ENGINE=mcts HONEST_K=8 MCTS_SIMS=700" "ensemble MCTS: 8 determinized trees x 700 sims (~= 1-ply K=8 wall-clock)"
$R mcts_K16_S1000 150 1800 "HONEST=1 HONEST_ENGINE=mcts HONEST_K=16 MCTS_SIMS=1000" "ensemble MCTS: 16 determinized trees x 1000 sims (~3x budget)"
echo PHASE2A_DONE
