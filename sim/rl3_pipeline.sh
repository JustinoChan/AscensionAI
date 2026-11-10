#!/bin/bash
# RL round 3: out-of-combat candidate scorer WITH visible-map features (MACRO_MAPFEAT=1) and a boss-kill bonus
# (RL_BOSS_BONUS=0.3): fresh BC data (mainline heuristics, K=8) -> BC -> PPO (K=8, 14 workers) -> eval500 gate.
cd "$HOME/sts"
export MACRO_MAPFEAT=1
mkdir -p nets/rl3 sim_logs/rl3
grep -q "^n=1000 " sim_logs/honest/rl3_bcdata/SUMMARY.txt 2>/dev/null || \
  SEED0=820000013 SEEDSTEP=19 PAR=14 ./run_honest.sh rl3_bcdata 1000 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 MACRO_MAPFEAT=1 MACRO_FEAT_LOG=sim_logs/rl3/bcdata" "RL round 3 BC data: mainline macro + candidate features incl. visible map, K=8"
[ -f nets/rl3/pol_bc.npz ] || BCDATA=sim_logs/rl3/bcdata RL_OUT=nets/rl3 nice -n 5 .venv_train/bin/python rl_macro_train.py bc > nets/rl3/bc.log 2>&1
tail -2 nets/rl3/bc.log
RL_OUT=nets/rl3 RL_PROG=sim_logs/honest/rl_macro3 RL_TMP=sim_logs/rl3 RL_SEED0=640000007 RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=0.3 \
  nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-150} >> nets/rl3/rl.log 2>&1
tail -3 nets/rl3/rl.log
NETDIR=nets/rl3 ARM=eval500_rl3_K64 OUTF=sim_logs/GATE_rl3_eval500.txt EXTRA_ENV="MACRO_MAPFEAT=1" ./rl_gate.sh
python3 ledger.py compare eval500_rl_K64 eval500_rl3_K64 | sed -n 1,20p | tee -a sim_logs/GATE_rl3_eval500.txt
python3 boss_report.py eval500_rl_K64 eval500_rl3_K64 | tee -a sim_logs/GATE_rl3_eval500.txt
