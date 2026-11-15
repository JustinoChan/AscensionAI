#!/bin/bash
# eval500 of the best policy (5d) with the boss engine that won the held-out gate (skips if k64 stayed best)
cd "$HOME/sts"
B=$(tail -1 sim_logs/GATE_boss_engine.txt | awk '{print $2}')
case "$B" in mcts16) T=16; SIM=1000;; mcts32) T=32; SIM=2000;; mcts64) T=64; SIM=4000;; *) echo "gate kept k64 -> skip"; exit 0;; esac
grep -q "^n=500 " sim_logs/honest/eval500_rl5d_$B/SUMMARY.txt 2>/dev/null || \
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl5d_$B 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz HONEST_ENGINE_BOSS=mcts MCTS_TREES_BOSS=$T MCTS_SIMS_BOSS=$SIM TRACE=1" "5d policy + honest MCTS in boss fights ($B), K=64 elsewhere, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5d_K64 eval500_rl5d_$B | sed -n 1,20p; python3 boss_report.py eval500_rl5d_K64 eval500_rl5d_$B | grep -E "fight|BOSS"; } | tee sim_logs/GATE_boss_engine_eval500.txt
