#!/bin/bash
# Boss-fight search budget on the round 11 mainline (exi1 prior c=5 + play-outs): K=64 vs 128 vs 256 in boss fights only,
# fresh held-out A2+A3 bosses, identical starts. PASS (best beats K=64, p<0.25) -> eval500 with HONEST_K_BOSS vs 232
# -> if >= 232, eval500b for the 1000-seed test vs 466.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"
for ACT in 2 3; do
  OUT=sim_logs/boss_kboss_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_KBOSS=1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999990013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_kboss_a2 sim_logs/boss_kboss_a3 | tee sim_logs/GATE_kboss.txt
B=$(tail -1 sim_logs/GATE_kboss.txt | awk '{print $2}'); [ "$B" = base ] && exit 0
KB=${B#k}
X="HONEST=1 HONEST_CPP=1 HONEST_K=64 HONEST_K_BOSS=$KB $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1"
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_kb$KB 500 7200 "$X" "round 11 mainline + boss K=$KB, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_kb$KB | sed -n 1,8p | tee -a sim_logs/GATE_kboss.txt
W=$(python3 -c "import sqlite3;print(sqlite3.connect('sim_logs/results.db').execute(\"select sum(win) from seeds where run='eval500_rl11_kb$KB'\").fetchone()[0] or 0)")
[ "$W" -ge 232 ] || exit 0
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_rl11_kb$KB 500 7200 "$X" "round 11 mainline + boss K=$KB, eval500b"
python3 ledger.py ingest > /dev/null; python3 b1000.py rl11_main rl11_kb$KB | tee -a sim_logs/GATE_kboss.txt
