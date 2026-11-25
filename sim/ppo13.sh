#!/bin/bash
# Round 13: continue round 12 (deck PPO under the new mainline combat) from its gated checkpoint it049, critic warm,
# fresh seeds, 50 iterations, K=16; eval500 gate vs mainline round 11 (232) and round 12 (237).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
export MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1
mkdir -p nets/rl13; [ -f nets/rl13/pol_bc.npz ] || cp nets/rl12/pol_gate.npz nets/rl13/pol_bc.npz
[ -f nets/rl13/critic.pt ] || cp nets/rl12/critic.pt nets/rl13/critic.pt
RL_CRITIC=1 RL_OUT=nets/rl13 RL_PROG=sim_logs/honest/rl_macro13 RL_TMP=sim_logs/rl13 RL_SEED0=901300073 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-50} >> nets/rl13/rl.log 2>&1
NETDIR=nets/rl13 ARM=eval500_rl13_main OUTF=sim_logs/GATE_rl13_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl11_main eval500_rl13_main | sed -n 1,8p; python3 ledger.py compare eval500_rl12_main eval500_rl13_main | sed -n 3,6p; } | tee -a sim_logs/GATE_rl13_eval500.txt
