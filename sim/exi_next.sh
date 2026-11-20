#!/bin/bash
# After expert iteration 1: PASS -> iteration 2 (teacher and play-out policy = exi1 net, data aggregated over iterations,
# gate vs exi1); FAIL -> exi1b: retrain on the same data with sharper targets (EXI_TAU=0.5) and gate again vs varN.
cd "$HOME/sts"
if grep -q "^PASS" sim_logs/GATE_exi1.txt; then
  ./exi_iter.sh 2 nets/exi1/rollout.npz
else
  mkdir -p nets/exi1b
  EXI_TAU=0.5 .venv_train/bin/python exi_train.py policy_varN.npz nets/exi1b/rollout.npz sim_logs/exi1/data 2>&1 | tee nets/exi1b/train.log
  G=sim_logs/exi1b_gate
  for w in $(seq 0 13); do
    MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10 GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_gate.py nets/exi1b/rollout.npz $((979000003 + 1000000 + w*31)) 9 $G >> $G.log 2>/dev/null &
  done; wait
  python3 boss_gate_report.py $G | tee sim_logs/GATE_exi1b.txt
fi
