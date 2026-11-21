#!/bin/bash
# Root policy prior gate: held-out Act 2 and Act 3 boss fights (fresh seeds), mainline (5d + PV10), identical starts:
# base vs Q + c*log pi_exi1 for c in 2,5,10,20, and c=5 with exi1 also as play-out net.
# PASS (a variant beats base, p<0.25, both acts pooled) -> eval500 + PV10 vs 192.
# FAIL -> same gate with the exi1c net (more data) as the prior, Act 2 only.
cd "$HOME/sts"
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"
gate() {  # NET TAG ACTS
  for ACT in $3; do
    OUT=sim_logs/boss_prior_$2_a$ACT; N=$([ $ACT = 2 ] && echo 10 || echo 14)
    for w in $(seq 0 13); do
      env $MAIN PRIOR_NET=$1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((988000013 + ACT*100000 + w*31)) $N $OUT >> $OUT.log 2>/dev/null &
    done; wait
  done
  python3 boss_prior_report.py $(for ACT in $3; do echo sim_logs/boss_prior_$2_a$ACT; done) | tee sim_logs/GATE_boss_prior_$2.txt
}
gate nets/exi1/rollout.npz exi1 "2 3"
B=$(tail -1 sim_logs/GATE_boss_prior_exi1.txt | awk '{print $2}'); NET=nets/exi1/rollout.npz
if [ "$B" = base ] && [ -f nets/exi1c/rollout.npz ]; then
  gate nets/exi1c/rollout.npz exi1c "2"
  B=$(tail -1 sim_logs/GATE_boss_prior_exi1c.txt | awk '{print $2}'); NET=nets/exi1c/rollout.npz
fi
[ "$B" = base ] && { echo "no root-prior variant passed" | tee -a sim_logs/GATE_boss_prior_exi1.txt; exit 0; }
C=$(echo $B | sed 's/^c\([0-9.]*\).*/\1/'); X="HONEST_ROOT_NET=$NET HONEST_ROOT_C=$C"; [ "${B%roll}" != "$B" ] && X="$X HONEST_ROLLOUT_NET=$NET"
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_prior_$B 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN $X TRACE=1" "root policy prior $B ($NET) + mainline, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl5d_pv10 eval500_prior_$B | sed -n 1,20p | tee sim_logs/GATE_prior_eval500.txt
