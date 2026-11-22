#!/bin/bash
# Round 11: deck-building PPO under the NEW mainline combat (exi1 root prior c=5 + exi1 play-outs + PV10) with K=16
# training combat, from 5d, critic on, 50 iterations; eval500 gate with the mainline vs eval500_prior_c5roll (216/500).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
export MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1
mkdir -p nets/rl11; [ -f nets/rl11/pol_bc.npz ] || cp nets/rl5d/pol_gate.npz nets/rl11/pol_bc.npz
RL_CRITIC=1 RL_OUT=nets/rl11 RL_PROG=sim_logs/honest/rl_macro11 RL_TMP=sim_logs/rl11 RL_SEED0=901100067 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-50} >> nets/rl11/rl.log 2>&1
NETDIR=nets/rl11 ARM=eval500_rl11_main OUTF=sim_logs/GATE_rl11_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1" ./rl_gate.sh
python3 ledger.py compare eval500_prior_c5roll eval500_rl11_main | sed -n 1,20p | tee -a sim_logs/GATE_rl11_eval500.txt
