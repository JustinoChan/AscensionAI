#!/bin/bash
# EXPERT_BOSS gate: held-out Act 2 and Act 3 boss fights (fresh seed range), rl6 + POTION_VALUE=10 mainline, 6 variants
# each from identical starts. If a rule variant wins with p<0.25 over both acts: full eval500 with it vs rl6+PV10.
cd "$HOME/sts"
POL="MACRO_MAPFEAT=1 MACRO_ENCFEAT=1 MACRO_POLICY=nets/rl6/pol_gate.npz POTION_VALUE=10"
for ACT in 2 3; do
  OUT=sim_logs/boss_expert_a$ACT; N=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $POL GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_expert_gate.py $((987000011 + ACT*100000 + w*31)) $N $OUT >> $OUT.log 2>/dev/null &
  done; wait
  python3 boss_expert_report.py $OUT | tee sim_logs/GATE_boss_expert_a$ACT.txt
done
python3 boss_expert_report.py sim_logs/boss_expert_a2 sim_logs/boss_expert_a3 | tee sim_logs/GATE_boss_expert.txt
B=$(tail -1 sim_logs/GATE_boss_expert.txt | awk '{print $2}')
case $B in
  pvdump) X="EXPERT_BOSS_PV=0.5,0";; pvzero) X="EXPERT_BOSS_PV=0,0";; risk) X="EXPERT_BOSS_RISK=0.5";;
  both) X="EXPERT_BOSS_PV=0.5,0 EXPERT_BOSS_RISK=0.5";; *) echo "no rule passed the boss gate; no eval500"; exit 0;;
esac
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl6_pv10_$B 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $POL $X TRACE=1" "rl6 + PV10 + EXPERT_BOSS $B, eval500"
python3 ledger.py ingest > /dev/null
python3 ledger.py compare eval500_rl6_pv10 eval500_rl6_pv10_$B | sed -n 1,20p | tee -a sim_logs/GATE_boss_expert.txt
