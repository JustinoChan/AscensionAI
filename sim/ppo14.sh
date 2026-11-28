#!/bin/bash
# Round 14: lower-noise deck PPO from the round 11 mainline policy (rounds 12/13 plain continuations stalled):
# 200 games/iteration, LR 2e-5, 200-game validation every 5 iterations, critic warm from round 11, 25 iterations;
# eval500 gate vs round 11 (232).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
export MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1
mkdir -p nets/rl14; [ -f nets/rl14/pol_bc.npz ] || cp nets/rl11/pol_gate.npz nets/rl14/pol_bc.npz
[ -f nets/rl14/critic.pt ] || cp nets/rl11/critic.pt nets/rl14/critic.pt
RL_CRITIC=1 RL_OUT=nets/rl14 RL_PROG=sim_logs/honest/rl_macro14 RL_TMP=sim_logs/rl14 RL_SEED0=901400079 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_GAMES=200 RL_VAL_N=200 RL_BOSS_BONUS=0.3 RL_LR=2e-5 RL_EVAL_EVERY=5 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-25} >> nets/rl14/rl.log 2>&1
NETDIR=nets/rl14 ARM=eval500_rl14_main OUTF=sim_logs/GATE_rl14_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1" ./rl_gate.sh
python3 ledger.py compare eval500_rl11_main eval500_rl14_main | sed -n 1,8p | tee -a sim_logs/GATE_rl14_eval500.txt
