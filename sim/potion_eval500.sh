#!/bin/bash
# eval500: best policy (5d) + POTION_SAVE=1 (no potions in hallway/event fights unless HP <= 25% or visible lethal) vs 5d
cd "$HOME/sts"
grep -q "^n=500 " sim_logs/honest/eval500_rl5d_potsave/SUMMARY.txt 2>/dev/null || \
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl5d_potsave 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_SAVE=1 TRACE=1" "5d policy + POTION_SAVE (potions kept for elites/bosses), K=64, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5d_K64 eval500_rl5d_potsave | sed -n 1,20p; python3 boss_report.py eval500_rl5d_K64 eval500_rl5d_potsave | grep -E "fight|BOSS|ELITE"; } | tee sim_logs/GATE_potsave_eval500.txt
