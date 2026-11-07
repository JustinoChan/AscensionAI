#!/bin/bash
# Phase 2 queue C: hybrid engine (1-ply in Act 1, ensemble MCTS from Act 2), starts after queue B.
cd "$HOME/sts"
while pgrep -f run_phase2b.sh > /dev/null; do sleep 20; done
R=./run_honest.sh
$R hybrid_adapt_mcts 150 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 HONEST_K_ELITE=32 HONEST_K_BOSS=64 HONEST_ENGINE_BY_ACT=1ply,mcts,mcts MCTS_TREES=16 MCTS_SIMS=1000" "hybrid: Act1 1-ply adaptive 8/32/64, Act2+ MCTS 16x1000"
$R hybrid_adapt_mcts_potsave 150 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 HONEST_K_ELITE=32 HONEST_K_BOSS=64 HONEST_ENGINE_BY_ACT=1ply,mcts,mcts MCTS_TREES=16 MCTS_SIMS=1000 POTION_SAVE=1" "hybrid + potion conservation"
echo PHASE2C_DONE
