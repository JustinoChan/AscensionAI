#!/bin/bash
# Round 10: lower-noise PPO. Rounds 8/9 (PV10 + K=16 training) give ~+1.5 points that don't reach significance over
# 1000 seeds, and their 100-game validation is too noisy to pick checkpoints. From round 9's gated checkpoint (209/500):
# 3x games per iteration (300), LR 2e-5, 20 iterations, critic warm from round 9, 200-game validation every 5 iterations;
# eval500 gate with PV10 vs 5d+PV10 (192) and round 9 (209).
cd "$HOME/sts"
export MACRO_MAPFEAT=1 POTION_VALUE=10
mkdir -p nets/rl10
[ -f nets/rl10/pol_bc.npz ] || cp nets/rl9/pol_gate.npz nets/rl10/pol_bc.npz
[ -f nets/rl10/critic.pt ] || cp nets/rl9/critic.pt nets/rl10/critic.pt
RL_CRITIC=1 RL_OUT=nets/rl10 RL_PROG=sim_logs/honest/rl_macro10 RL_TMP=sim_logs/rl10 RL_SEED0=901000061 RL_COMBAT_K=16 RL_WORKERS=14 \
  RL_GAMES=300 RL_VAL_N=200 RL_BOSS_BONUS=0.3 RL_LR=2e-5 RL_EVAL_EVERY=5 nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-20} >> nets/rl10/rl.log 2>&1
NETDIR=nets/rl10 ARM=eval500_rl10_pv10 OUTF=sim_logs/GATE_rl10_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_rl10_pv10 | sed -n 1,20p; python3 ledger.py compare eval500_rl9_pv10 eval500_rl10_pv10 | sed -n 4,6p; } | tee -a sim_logs/GATE_rl10_eval500.txt
