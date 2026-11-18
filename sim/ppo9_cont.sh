#!/bin/bash
# Round 9: continue round 8's direction from its gated checkpoint (it039, 201/500): PPO with PV10 + K=16 training
# combat, critic warm-started from round 8, 50 iterations, fresh training seeds; eval500 gate with PV10 vs 5d+PV10 (192)
# and vs round 8 (201).
cd "$HOME/sts"
export MACRO_MAPFEAT=1 POTION_VALUE=10
mkdir -p nets/rl9
[ -f nets/rl9/pol_bc.npz ] || cp nets/rl8/pol_gate.npz nets/rl9/pol_bc.npz
[ -f nets/rl9/critic.pt ] || cp nets/rl8/critic.pt nets/rl9/critic.pt
RL_CRITIC=1 RL_OUT=nets/rl9 RL_PROG=sim_logs/honest/rl_macro9 RL_TMP=sim_logs/rl9 RL_SEED0=900900053 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-50} >> nets/rl9/rl.log 2>&1
NETDIR=nets/rl9 ARM=eval500_rl9_pv10 OUTF=sim_logs/GATE_rl9_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_rl9_pv10 | sed -n 1,20p; python3 ledger.py compare eval500_rl8_pv10 eval500_rl9_pv10 | sed -n 4,6p
  python3 boss_report.py eval500_rl5d_pv10 eval500_rl9_pv10 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl9_eval500.txt
