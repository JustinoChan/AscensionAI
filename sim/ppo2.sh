#!/bin/bash
# continue PPO from the gated checkpoint (it099) with stronger K=16 combat in training, 14 workers, new RL seeds.
cd "$HOME/sts"
mkdir -p nets/rl2; [ -f nets/rl2/pol_bc.npz ] || cp nets/rl/val_it099.npz nets/rl2/pol_bc.npz
RL_OUT=nets/rl2 RL_PROG=sim_logs/honest/rl_macro2 RL_TMP=sim_logs/rl2 RL_SEED0=620000003 RL_COMBAT_K=16 RL_WORKERS=14 RL_LR=5e-5 \
  nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-80} >> nets/rl2/rl.log 2>&1
tail -3 nets/rl2/rl.log
