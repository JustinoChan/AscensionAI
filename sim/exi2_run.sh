#!/bin/bash
# Expert iteration 2 with the integration that passed (root prior c=5 + play-out net = exi1):
#  collect 8 h: run played by mainline + exi1 prior/play-outs; Act 2/3 elite+boss fights played greedily by exi1 in
#  determinized worlds; every visited state labelled with honest K=64 Q using exi1 play-outs (the improved search);
#  train from exi1 on exi1+exi1c+exi2 data; gate exi1 vs exi2 (both c=5 + play-outs) on fresh held-out A2+A3 bosses;
#  PASS -> eval500 + PV10 with exi2 vs mainline (192) and vs exi1 prior (eval500_prior_c5roll).
cd "$HOME/sts"
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"; P1=nets/exi1/rollout.npz
D=sim_logs/exi2; N=nets/exi2; mkdir -p $D $N
for w in $(seq 0 13); do
  env $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 EXI_K=64 EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434 \
    timeout 8h nice -n 5 .venv/bin/python exi_collect.py $P1 $((990000001 + w*31)) 200 $D/data >> $D/collect.log 2>/dev/null &
done; wait
EXI_TAU=1 EXI_EPOCHS=20 .venv_train/bin/python exi_train.py $P1 $N/rollout.npz $D/data sim_logs/exi1/data sim_logs/exi1c/data 2>&1 | tee $N/train.log
for ACT in 2 3; do
  OUT=sim_logs/boss_exi2_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$N/rollout.npz PRIOR_BASE_NET=$P1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((991000013 + ACT*100000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_exi2_a2 sim_logs/boss_exi2_a3 | tee sim_logs/GATE_exi2.txt
[ "$(tail -1 sim_logs/GATE_exi2.txt | awk '{print $2}')" = c5roll ] || { echo "exi2 did not beat exi1" | tee -a sim_logs/GATE_exi2.txt; exit 0; }
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_exi2_c5roll 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$N/rollout.npz HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$N/rollout.npz TRACE=1" "expert iteration 2 (root prior c=5 + play-outs) + mainline, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_exi2_c5roll | sed -n 1,8p; python3 ledger.py compare eval500_prior_c5roll eval500_exi2_c5roll | sed -n 3,6p; } | tee -a sim_logs/GATE_exi2.txt
