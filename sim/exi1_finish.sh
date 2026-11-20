#!/bin/bash
# Finish expert iteration 1 after its driver was detached: wait for the boss-gate workers, report, eval500 on PASS.
cd "$HOME/sts"
while pgrep -f "boss_gate.py nets/exi1/rollout.npz" > /dev/null; do sleep 60; done
G=sim_logs/exi1_gate; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"
python3 boss_gate_report.py $G | tee sim_logs/GATE_exi1.txt
grep -q "^PASS" sim_logs/GATE_exi1.txt || { echo "exi1: failed the boss-fight gate; no eval500" | tee -a sim_logs/GATE_exi1.txt; exit 0; }
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_exi1_pv10 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROLLOUT_NET=nets/exi1/rollout.npz TRACE=1" "combat expert iteration 1 (play-out net) + mainline, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl5d_pv10 eval500_exi1_pv10 | sed -n 1,20p; python3 boss_report.py eval500_rl5d_pv10 eval500_exi1_pv10 | grep -E "fight|BOSS"; } | tee -a sim_logs/GATE_exi1.txt
