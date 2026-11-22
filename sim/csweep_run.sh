#!/bin/bash
# c sweep around the new mainline c5roll (exi1 prior + play-outs): c=3 and c=10 vs c=5 on fresh held-out A2+A3 bosses;
# a variant that beats c=5 (p<0.25) -> eval500 vs eval500_prior_c5roll (216).
cd "$HOME/sts"
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"; P1=nets/exi1/rollout.npz
for ACT in 2 3; do
  OUT=sim_logs/boss_csweep_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_CSWEEP=1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((994000013 + ACT*100000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_csweep_a2 sim_logs/boss_csweep_a3 | tee sim_logs/GATE_csweep.txt
B=$(tail -1 sim_logs/GATE_csweep.txt | awk '{print $2}'); [ "$B" = base ] && exit 0
C=$(echo $B | sed 's/^c\([0-9.]*\)roll/\1/')
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_prior_c${C}roll 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=$C HONEST_ROLLOUT_NET=$P1 TRACE=1" "root prior c=$C + exi1 play-outs, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_prior_c5roll eval500_prior_c${C}roll | sed -n 1,8p | tee -a sim_logs/GATE_csweep.txt
