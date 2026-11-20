#!/bin/bash
# Combat expert iteration, one iteration: ./exi_iter.sh ITER PREV_NET
#  1) collect (<= EXI_HOURS h): 14 workers, fresh seeds, mainline run policy (5d + PV10), Act 2/3 elite+boss fights played
#     greedily by PREV_NET in determinized worlds, every visited state labelled by honest K=64 search Q (exi_collect.py)
#  2) train the play-out net on soft Q targets, warm start PREV_NET (exi_train.py, GPU)
#  3) gate inside search: held-out Act 2/3 boss fights, identical starts and samples, PREV_NET vs new (boss_gate.py)
#  4) if PASS: eval500 with HONEST_ROLLOUT_NET=new + PV10 vs 5d+PV10 (192/500)
cd "$HOME/sts"
IT=$1; PREV=${2:-policy_varN.npz}; H=${EXI_HOURS:-8}
D=sim_logs/exi$IT; N=nets/exi$IT; mkdir -p $D $N
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"
for w in $(seq 0 13); do
  env $MAIN EXI_K=64 EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434 timeout ${H}h nice -n 5 .venv/bin/python exi_collect.py $PREV $((978000001 + IT*1000000 + w*31)) 200 $D/data >> $D/collect.log 2>/dev/null &
done; wait
.venv_train/bin/python exi_train.py $PREV $N/rollout.npz $(ls -d sim_logs/exi*/data) 2>&1 | tee $N/train.log
G=sim_logs/exi${IT}_gate
for w in $(seq 0 13); do
  env $MAIN GATE_BASE_NET=$PREV GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_gate.py $N/rollout.npz $((979000003 + IT*1000000 + w*31)) 9 $G >> $G.log 2>/dev/null &
done; wait
python3 boss_gate_report.py $G | tee sim_logs/GATE_exi$IT.txt
grep -q "^PASS" sim_logs/GATE_exi$IT.txt || { echo "exi$IT: failed the boss-fight gate; no eval500" | tee -a sim_logs/GATE_exi$IT.txt; exit 0; }
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_exi${IT}_pv10 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROLLOUT_NET=$N/rollout.npz TRACE=1" "combat expert iteration $IT (play-out net) + mainline, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_exi${IT}_pv10 | sed -n 1,20p; python3 boss_report.py eval500_rl5d_pv10 eval500_exi${IT}_pv10 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_exi$IT.txt
