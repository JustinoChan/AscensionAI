#!/bin/bash
# Round 7: boss-readiness shaping. (1) offline gate: readiness net AUC on held-out runs (ready_train.py);
# (2) if PASS: PPO fine-tune from 5d, critic on, RL_READY potential + GAE(0.9), 80 it; (3) eval500 gate with
# POTION_VALUE=10 vs 5d+PV10 (192/500).
cd "$HOME/sts"
[ -f nets/ready/data.npz ] || .venv/bin/python ready_data.py nets/ready/data.npz sim_logs/rl5b sim_logs/rl5c sim_logs/rl5d sim_logs/rl5e sim_logs/rl6 > nets/ready/data.log 2>&1
[ -f nets/ready/ready.npz ] || .venv_train/bin/python ready_train.py nets/ready/data.npz nets/ready/ready.npz | tee sim_logs/GATE_ready.txt
grep -q "GATE PASS" sim_logs/GATE_ready.txt || { echo "readiness net failed its offline gate; no PPO"; exit 0; }
export MACRO_MAPFEAT=1
mkdir -p nets/rl7; [ -f nets/rl7/pol_bc.npz ] || cp nets/rl5d/pol_gate.npz nets/rl7/pol_bc.npz
RL_CRITIC=1 RL_READY=nets/ready/ready.npz RL_READY_W=0.5 RL_LAMBDA=0.9 RL_OUT=nets/rl7 RL_PROG=sim_logs/honest/rl_macro7 RL_TMP=sim_logs/rl7 \
  RL_SEED0=900700043 RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0.3 RL_LR=3e-5 RL_EVAL_EVERY=10 \
  nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-80} >> nets/rl7/rl.log 2>&1
NETDIR=nets/rl7 ARM=eval500_rl7_pv10 OUTF=sim_logs/GATE_rl7_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1 POTION_VALUE=10" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_rl7_pv10 | sed -n 1,20p; python3 boss_report.py eval500_rl5d_pv10 eval500_rl7_pv10 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl7_eval500.txt
