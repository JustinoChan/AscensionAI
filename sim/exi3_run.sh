#!/bin/bash
# Combat round on the round 11 mainline (alternating loops): collect 6 h of the mainline's own LIVE ROOT states in Act 2/3
# elite+boss fights (episodes follow the search's choice, Q + 5 log pi_exi1, in determinized worlds), every state labelled
# with honest K=64 Q from exi1 play-outs; train from exi1 on exi3 + earlier data; pair gate exi1 vs exi3 (both c=5 +
# play-outs) on fresh held-out A2+A3 bosses; PASS -> eval500 vs the mainline (eval500_rl11_main, 232). FAIL -> stop
# (the queue's backfill takes over).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"
D=sim_logs/exi3; N=nets/exi3; mkdir -p $D $N
for w in $(seq 0 13); do
  env $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 EXI_ACT=search EXI_K=64 EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434 \
    timeout 6h nice -n 5 .venv/bin/python exi_collect.py $P1 $((997000001 + w*31)) 500 $D/data >> $D/collect.log 2>/dev/null &
done; wait
rm -f $D/data/*.tmp.npz
EXI_TAU=1 EXI_EPOCHS=20 .venv_train/bin/python exi_train.py $P1 $N/rollout.npz $D/data sim_logs/exi2/data sim_logs/exi1/data sim_logs/exi1c/data $( [ -d sim_logs/exi_bf/data ] && echo sim_logs/exi_bf/data ) 2>&1 | tee $N/train.log
for ACT in 2 3; do
  OUT=sim_logs/boss_exi3_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$N/rollout.npz PRIOR_BASE_NET=$P1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((998000013 + ACT*100000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_exi3_a2 sim_logs/boss_exi3_a3 | tee sim_logs/GATE_exi3.txt
[ "$(tail -1 sim_logs/GATE_exi3.txt | awk '{print $2}')" = c5roll ] || { echo "exi3 did not beat exi1" | tee -a sim_logs/GATE_exi3.txt; exit 0; }
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_exi3 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$N/rollout.npz HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$N/rollout.npz TRACE=1" "round 11 + exi3 combat net, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_exi3 | sed -n 1,8p | tee -a sim_logs/GATE_exi3.txt
