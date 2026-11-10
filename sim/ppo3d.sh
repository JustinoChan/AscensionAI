#!/bin/bash
# continue round-3 PPO from the current best (nets/rl3c/pol_gate.npz = rl3c it099; eval500 53/500), same flags, 150 iters, then eval500 gate
cd "$HOME/sts"
export MACRO_MAPFEAT=1
mkdir -p nets/rl3d; [ -f nets/rl3d/pol_bc.npz ] || cp nets/rl3c/pol_gate.npz nets/rl3d/pol_bc.npz
RL_OUT=nets/rl3d RL_PROG=sim_logs/honest/rl_macro3d RL_TMP=sim_logs/rl3d RL_SEED0=720000019 RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0.3 RL_LR=5e-5 \
  nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-150} >> nets/rl3d/rl.log 2>&1
NETDIR=nets/rl3d ARM=eval500_rl3d_K64 OUTF=sim_logs/GATE_rl3d_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl3c_K64 eval500_rl3d_K64 | sed -n 1,20p; python3 boss_report.py eval500_rl3c_K64 eval500_rl3d_K64 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl3d_eval500.txt
