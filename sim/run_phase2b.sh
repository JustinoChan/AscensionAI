#!/bin/bash
# Phase 2 queue B: more samples, adaptive K by room, potion conservation (150 seeds each).
cd "$HOME/sts"
R=./run_honest.sh
$R cpp_1ply_K32 150 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=32" "1-ply native K=32 everywhere"
$R cpp_adapt_8_32_64 150 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 HONEST_K_ELITE=32 HONEST_K_BOSS=64" "1-ply native, adaptive K: hallway 8 / elite 32 / boss 64"
$R cpp_adapt_potsave 150 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 HONEST_K_ELITE=32 HONEST_K_BOSS=64 POTION_SAVE=1" "adaptive K 8/32/64 + potion conservation"
$R mcts_K16_S1000_potsave 150 3600 "HONEST=1 HONEST_ENGINE=mcts HONEST_K=16 MCTS_SIMS=1000 POTION_SAVE=1" "ensemble MCTS 16x1000 + potion conservation"
echo PHASE2B_DONE
