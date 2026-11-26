#!/bin/bash
# Combat learner step 5: afterstate root prior. exi4 (fresh obs-only learner) failed, and obs-only nets sit at 5.5x the
# label noise (regret 3.3 vs 0.6). Give the prior what each move leads to: collect 6 h of mainline live root states with
# per-action afterstates (each legal action played once in a sampled/determinized copy; honest) and honest K=64 Q labels;
# train exi_train_as.py (MLP on [s, a_s - s], CE + expected regret); pair boss gate exi1 prior vs afterstate prior (both c=5,
# exi1 play-outs) on fresh held-out A2+A3 bosses; PASS -> eval500 vs round 11 mainline (232).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"
D=sim_logs/exi5; N=nets/exi5; mkdir -p $D $N
for w in $(seq 0 13); do
  env $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 EXI_AFTER=1 EXI_ACT=search EXI_K=64 EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434 \
    timeout 6h nice -n 5 .venv/bin/python exi_collect.py $P1 $((999900001 + w*31)) 500 $D/data >> $D/collect.log 2>/dev/null &
done; wait
rm -f $D/data/*.tmp.npz
EXI_TAU=1 EXI_REGRET_W=0.2 EXI_EPOCHS=20 .venv_train/bin/python exi_train_as.py $N/as.npz $D/data 2>&1 | tee $N/train.log
for ACT in 2 3; do
  OUT=sim_logs/boss_exi5_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_BASE_NET=$P1 PRIOR_AS=$N/as.npz GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999950013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_exi5_a2 sim_logs/boss_exi5_a3 | tee sim_logs/GATE_exi5.txt
[ "$(tail -1 sim_logs/GATE_exi5.txt | awk '{print $2}')" = c5roll ] || { echo "afterstate prior did not beat exi1" | tee -a sim_logs/GATE_exi5.txt; exit 0; }
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_exi5as 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_AS=$N/as.npz HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1" "round 11 + afterstate root prior, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_exi5as | sed -n 1,8p | tee -a sim_logs/GATE_exi5.txt
