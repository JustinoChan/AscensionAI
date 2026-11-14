#!/bin/bash
# paired eval500 of the best policy with the boss-RL net as rollout policy (HONEST_ROLLOUT_NET), only if the search gate passed
cd "$HOME/sts"
tail -1 sim_logs/GATE_boss1_search.txt 2>/dev/null | grep -q PASS || { echo "boss search gate not passed (or not done): skipping eval500"; exit 0; }
NEW=${NEW:-nets/boss1/val_it059.npz}
grep -q "^n=500 " sim_logs/honest/eval500_rl5b_boss_K64/SUMMARY.txt 2>/dev/null || \
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl5b_boss_K64 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5b/pol_gate.npz HONEST_ROLLOUT_NET=$NEW TRACE=1" "rl5b policy + boss-RL rollout net ($NEW), K=64, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5b_K64 eval500_rl5b_boss_K64 | sed -n 1,20p; python3 boss_report.py eval500_rl5b_K64 eval500_rl5b_boss_K64 | grep -E "fight|BOSS"; } | tee sim_logs/GATE_boss1_eval500.txt
