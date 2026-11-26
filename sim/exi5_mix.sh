#!/bin/bash
# Fallback after combat step 5: if the afterstate prior alone did not beat exi1, gate the MIXED prior
# (0.5 * afterstate log-probs + 0.5 * exi1 log-probs, c=5, exi1 play-outs) vs exi1 alone on fresh held-out A2+A3 bosses;
# PASS -> eval500 vs round 11 mainline (232). Skipped if step 5 already passed.
cd "$HOME/sts"
grep -q "^BEST base" sim_logs/GATE_exi5.txt 2>/dev/null || { echo "step 5 passed or not finished; mix gate skipped"; exit 0; }
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"; AS=nets/exi5/as.npz
for ACT in 2 3; do
  OUT=sim_logs/boss_exi5mix_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_BASE_NET=$P1 PRIOR_MIX_AS=$AS GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999970013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_exi5mix_a2 sim_logs/boss_exi5mix_a3 | tee sim_logs/GATE_exi5mix.txt
[ "$(tail -1 sim_logs/GATE_exi5mix.txt | awk '{print $2}')" = c5roll ] || exit 0
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_exi5mix 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_AS=$AS HONEST_ROOT_MIX=0.5 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1" "round 11 + mixed exi1/afterstate prior, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_exi5mix | sed -n 1,8p | tee -a sim_logs/GATE_exi5mix.txt
