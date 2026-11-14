#!/bin/bash
# held-out boss fights inside honest K=64 search: varN vs boss-RL net as rollout policy (4 workers x 40 fresh seeds)
cd "$HOME/sts"
NEW=${NEW:-nets/boss1/val_it059.npz}; OUT=sim_logs/boss_gate1
for w in 0 1 2 3; do
  MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5b/pol_gate.npz GATE_SEEDSTEP=124 nice -n 5 .venv/bin/python boss_gate.py $NEW $((975000001 + w*31)) 40 $OUT >> $OUT.log 2>/dev/null &
done; wait
python3 boss_gate_report.py | tee sim_logs/GATE_boss1_search.txt
