#!/bin/bash
# Expert iteration 1c: double the data (another 8 h of varN play-out states labelled by honest K=64, fresh seeds), retrain
# from varN on all ~300k states (EXI_TAU=1, 20 epochs), boss gate vs varN, eval500 + PV10 on PASS.
cd "$HOME/sts"
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"; D=sim_logs/exi1c; N=nets/exi1c; mkdir -p $D $N
for w in $(seq 0 13); do
  env $MAIN EXI_K=64 EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434 timeout 8h nice -n 5 .venv/bin/python exi_collect.py policy_varN.npz $((982000001 + w*31)) 200 $D/data >> $D/collect.log 2>/dev/null &
done; wait
EXI_TAU=1 EXI_EPOCHS=20 .venv_train/bin/python exi_train.py policy_varN.npz $N/rollout.npz sim_logs/exi1/data $D/data 2>&1 | tee $N/train.log
G=sim_logs/exi1c_gate
for w in $(seq 0 13); do
  env $MAIN GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_gate.py $N/rollout.npz $((979000003 + 1000000 + w*31)) 9 $G >> $G.log 2>/dev/null &
done; wait
python3 boss_gate_report.py $G | tee sim_logs/GATE_exi1c.txt
grep -q "^PASS" sim_logs/GATE_exi1c.txt || exit 0
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_exi1c_pv10 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROLLOUT_NET=$N/rollout.npz TRACE=1" "expert iteration 1c play-out net + mainline, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl5d_pv10 eval500_exi1c_pv10 | sed -n 1,8p | tee -a sim_logs/GATE_exi1c.txt
