#!/bin/bash
# continue critic PPO from 5d (eval500 181/500) with its 512 critic and 200 games/it; resumable (morning gate separately)
cd "$HOME/sts"; export MACRO_MAPFEAT=1
mkdir -p nets/rl5e; [ -f nets/rl5e/pol_bc.npz ] || cp nets/rl5d/pol_gate.npz nets/rl5e/pol_bc.npz
[ -f nets/rl5e/critic.pt ] || cp nets/rl5d/critic.pt nets/rl5e/critic.pt
RL_CRITIC=1 RL_CRITIC_W=512 RL_GAMES=200 RL_OUT=nets/rl5e RL_PROG=sim_logs/honest/rl_macro5e RL_TMP=sim_logs/rl5e RL_SEED0=880500031 \
  RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=8 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-80} >> nets/rl5e/rl.log 2>&1
