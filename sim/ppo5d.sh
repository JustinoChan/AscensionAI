#!/bin/bash
# Round 5d: critic PPO from 5c (eval500 172/500) with BIGGER batches (200 games/iteration) and a WIDER critic (512, fresh,
# 5-iteration warm-up); 80 iterations on 14 workers; then eval500 gate vs 5c.
cd "$HOME/sts"
export MACRO_MAPFEAT=1
mkdir -p nets/rl5d; [ -f nets/rl5d/pol_bc.npz ] || cp nets/rl5c/pol_gate.npz nets/rl5d/pol_bc.npz
RL_CRITIC=1 RL_CRITIC_W=512 RL_GAMES=200 RL_OUT=nets/rl5d RL_PROG=sim_logs/honest/rl_macro5d RL_TMP=sim_logs/rl5d RL_SEED0=860500027 \
  RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=8 \
  nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-80} >> nets/rl5d/rl.log 2>&1
NETDIR=nets/rl5d ARM=eval500_rl5d_K64 OUTF=sim_logs/GATE_rl5d_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5c_K64 eval500_rl5d_K64 | sed -n 1,20p; python3 ledger.py compare eval500_rl5b_K64 eval500_rl5d_K64 | sed -n 4,6p; python3 boss_report.py eval500_rl5c_K64 eval500_rl5d_K64 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl5d_eval500.txt
