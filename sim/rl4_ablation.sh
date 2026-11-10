#!/bin/bash
# Round 4 single-variable ablations vs round 3 (eval500_rl3_K64: 51/500, it069; map features + bonus 0.3, from scratch BC):
#   VARIANT=a : + MACRO_BOSSFEAT (deck-role summary now/after), bonus 0.3   (fresh BC data: input size changes)
#   VARIANT=b : bonus 0.5 instead of 0.3, map features only                 (reuses round 3's BC policy)
# Each: PPO 150 it at K=8 (14 workers) from BC, then eval500 gate of the best checkpoint, paired vs round 3 and 017.
cd "$HOME/sts"
export MACRO_MAPFEAT=1
V=${VARIANT:?}
if [ "$V" = a ]; then
  export MACRO_BOSSFEAT=1; D=nets/rl4a; T=sim_logs/rl4a; BONUS=0.3; SEED=740000023; EXTRA="MACRO_MAPFEAT=1 MACRO_BOSSFEAT=1"
  mkdir -p $D $T
  grep -q "^n=1000 " sim_logs/honest/rl4a_bcdata/SUMMARY.txt 2>/dev/null || \
    SEED0=840000017 SEEDSTEP=19 PAR=14 ./run_honest.sh rl4a_bcdata 1000 3600 "HONEST=1 HONEST_CPP=1 HONEST_K=8 MACRO_MAPFEAT=1 MACRO_BOSSFEAT=1 MACRO_FEAT_LOG=$T/bcdata" "RL round 4a BC data: + deck-role (boss) features, K=8"
  [ -f $D/pol_bc.npz ] || BCDATA=$T/bcdata RL_OUT=$D nice -n 5 .venv_train/bin/python rl_macro_train.py bc > $D/bc.log 2>&1
else
  D=nets/rl4b; T=sim_logs/rl4b; BONUS=0.5; SEED=760000027; EXTRA="MACRO_MAPFEAT=1"
  mkdir -p $D $T; [ -f $D/pol_bc.npz ] || cp nets/rl3/pol_bc.npz $D/pol_bc.npz
fi
RL_OUT=$D RL_PROG=sim_logs/honest/rl_macro4$V RL_TMP=$T RL_SEED0=$SEED RL_COMBAT_K=8 RL_WORKERS=14 RL_BOSS_BONUS=$BONUS \
  nice -n 5 .venv_train/bin/python rl_macro_train.py rl ${ITERS:-150} >> $D/rl.log 2>&1
NETDIR=$D ARM=eval500_rl4${V}_K64 OUTF=sim_logs/GATE_rl4${V}_eval500.txt EXTRA_ENV="$EXTRA" ./rl_gate.sh
{ python3 ledger.py compare eval500_rl3_K64 eval500_rl4${V}_K64 | sed -n 1,20p; python3 boss_report.py eval500_rl3_K64 eval500_rl4${V}_K64 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_rl4${V}_eval500.txt
