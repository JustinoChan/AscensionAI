#!/bin/bash
# Round 12: continue round 11 (deck PPO under the new mainline combat) from its gated checkpoint it049, critic warm,
# fresh seeds, 50 iterations, K=16; eval500 gate vs mainline (216) and round 11 (232).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz
export MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1
mkdir -p nets/rl12; [ -f nets/rl12/pol_bc.npz ] || cp nets/rl11/pol_gate.npz nets/rl12/pol_bc.npz
[ -f nets/rl12/critic.pt ] || cp nets/rl11/critic.pt nets/rl12/critic.pt
RL_CRITIC=1 RL_OUT=nets/rl12 RL_PROG=sim_logs/honest/rl_macro12 RL_TMP=sim_logs/rl12 RL_SEED0=901200071 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-50} >> nets/rl12/rl.log 2>&1
NETDIR=nets/rl12 ARM=eval500_rl12_main OUTF=sim_logs/GATE_rl12_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10 HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1" ./rl_gate.sh
{ python3 ledger.py compare eval500_prior_c5roll eval500_rl12_main | sed -n 1,8p; python3 ledger.py compare eval500_rl11_main eval500_rl12_main | sed -n 3,6p; } | tee -a sim_logs/GATE_rl12_eval500.txt
