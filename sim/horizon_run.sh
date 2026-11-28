#!/bin/bash
# Boss-only search horizon on the round 11 mainline (exi1 prior c=5 + play-outs): per-action discount 0.97 (base) vs
# 0.99 vs 0.995 with a 400-step play-out cap, BOSS fights only, fresh held-out A2+A3 bosses, identical starts.
# PASS (best beats base, p<0.25) -> eval500 vs round 11 (232) -> if >= 232, eval500b for the 1000-seed test vs 466.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"
for ACT in 2 3; do
  OUT=sim_logs/boss_horizon_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_HORIZON=1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999995013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_horizon_a2 sim_logs/boss_horizon_a3 | tee sim_logs/GATE_horizon.txt
B=$(tail -1 sim_logs/GATE_horizon.txt | awk '{print $2}'); [ "$B" = base ] && exit 0
case $B in g99) HX="HONEST_GAMMA_BOSS=0.99";; g995) HX="HONEST_GAMMA_BOSS=0.995 HONEST_ROLL_CAP_BOSS=400";; esac
X="HONEST=1 HONEST_CPP=1 HONEST_K=64 $HX $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1"
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_$B 500 7200 "$X" "round 11 mainline + boss horizon $B, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_$B | sed -n 1,8p | tee -a sim_logs/GATE_horizon.txt
W=$(python3 -c "import sqlite3;print(sqlite3.connect('sim_logs/results.db').execute(\"select sum(win) from seeds where run='eval500_rl11_$B'\").fetchone()[0] or 0)")
[ "$W" -ge 232 ] || exit 0
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_rl11_$B 500 7200 "$X" "round 11 mainline + boss horizon $B, eval500b"
python3 ledger.py ingest > /dev/null; python3 b1000.py rl11_main rl11_$B | tee -a sim_logs/GATE_horizon.txt
