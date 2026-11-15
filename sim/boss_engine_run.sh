#!/bin/bash
# held-out boss fights: honest 1-ply K=64 vs honest ensemble MCTS variants, identical starts; 14 workers x 12 fresh seeds
cd "$HOME/sts"; OUT=sim_logs/boss_engine_gate
for w in $(seq 0 13); do
  MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_engine_gate.py $((985000003 + w*31)) 12 $OUT >> $OUT.log 2>/dev/null &
done; wait
python3 boss_engine_report.py | tee sim_logs/GATE_boss_engine.txt
