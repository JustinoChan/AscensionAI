#!/bin/bash
# Elite-only search horizon (after the boss horizon gate passed): first Act 2 / Act 3 ELITE fight, fresh held-out seeds,
# identical starts, mainline prior: gamma 0.97 (base) vs 0.99 vs 0.995 + cap 400 in elite fights only.
# PASS (best beats base, p<0.25) -> eval500 with boss gamma 0.99 + the elite setting vs eval500_rl11_g99 (237) and 232.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"
for ACT in 2 3; do
  OUT=sim_logs/elite_horizon_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_HORIZON_ELITE=1 GATE_ROOM=ELITE GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999996013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/elite_horizon_a2 sim_logs/elite_horizon_a3 | tee sim_logs/GATE_elite_horizon.txt
B=$(tail -1 sim_logs/GATE_elite_horizon.txt | awk '{print $2}'); [ "$B" = base ] && exit 0
case $B in g99) EX="HONEST_GAMMA_ELITE=0.99";; g995) EX="HONEST_GAMMA_ELITE=0.995 HONEST_ROLL_CAP_ELITE=400";; esac
X="HONEST=1 HONEST_CPP=1 HONEST_K=64 HONEST_GAMMA_BOSS=0.99 $EX $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1"
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_g99_el$B 500 7200 "$X" "round 11 mainline + boss gamma 0.99 + elite $B, eval500"
python3 ledger.py ingest > /dev/null
{ python3 ledger.py compare eval500_rl11_g99 eval500_rl11_g99_el$B | sed -n 1,8p; python3 ledger.py compare eval500_rl11_main eval500_rl11_g99_el$B | sed -n 3,6p; } | tee -a sim_logs/GATE_elite_horizon.txt
