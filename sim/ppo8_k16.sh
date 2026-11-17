#!/bin/bash
# Round 8: PPO from 5d under the mainline (POTION_VALUE=10) with stronger training combat (K=16 instead of 8),
# critic on, 50 iterations; eval500 gate with PV10 vs 5d+PV10 (192/500).
cd "$HOME/sts"
export MACRO_MAPFEAT=1 POTION_VALUE=10
mkdir -p nets/rl8; [ -f nets/rl8/pol_bc.npz ] || cp nets/rl5d/pol_gate.npz nets/rl8/pol_bc.npz
RL_CRITIC=1 RL_OUT=nets/rl8 RL_PROG=sim_logs/honest/rl_macro8 RL_TMP=sim_logs/rl8 RL_SEED0=900800047 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-50} >> nets/rl8/rl.log 2>&1
NETDIR=nets/rl8 ARM=eval500_rl8_pv10 OUTF=sim_logs/GATE_rl8_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_rl8_pv10 | sed -n 1,20p; python3 boss_report.py eval500_rl5d_pv10 eval500_rl8_pv10 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl8_eval500.txt
