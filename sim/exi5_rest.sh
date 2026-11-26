#!/bin/bash
# Resume combat step 5 after the WSL crash (collection had finished: 476 seeds in sim_logs/exi5/data).
# Train the afterstate scorer; pair boss gate on fresh held-out A2+A3 bosses: exi1 obs prior (base) vs afterstate prior
# (c5roll, 1 sampled copy) vs afterstate prior averaged over 4 sampled copies (asK), all c=5 with exi1 play-outs.
# PASS (best non-base beats base, p<0.25) -> eval500 vs round 11 (232) -> if >= 232, eval500b for a 1000-seed test vs 466.
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"; N=nets/exi5; mkdir -p $N
[ -f $N/as.npz ] || EXI_TAU=1 EXI_REGRET_W=0.2 EXI_EPOCHS=20 .venv_train/bin/python exi_train_as.py $N/as.npz sim_logs/exi5/data 2>&1 | tee $N/train.log
for ACT in 2 3; do
  OUT=sim_logs/boss_exi5_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$P1 PRIOR_BASE_NET=$P1 PRIOR_AS=$N/as.npz PRIOR_AS_K=4 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999950013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_exi5_a2 sim_logs/boss_exi5_a3 | tee sim_logs/GATE_exi5.txt
B=$(tail -1 sim_logs/GATE_exi5.txt | awk '{print $2}')
[ "$B" = base ] && { echo "afterstate prior did not beat exi1" | tee -a sim_logs/GATE_exi5.txt; exit 0; }
K=1; [ "$B" = asK ] && K=4
X="HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$P1 HONEST_ROOT_AS=$N/as.npz HONEST_ROOT_AS_K=$K HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1"
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_exi5as 500 7200 "$X" "round 11 + afterstate root prior (K=$K), eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_exi5as | sed -n 1,8p | tee -a sim_logs/GATE_exi5.txt
W=$(python3 -c "import sqlite3;print(sqlite3.connect('sim_logs/results.db').execute(\"select sum(win) from seeds where run='eval500_rl11_exi5as'\").fetchone()[0] or 0)")
[ "$W" -ge 232 ] || exit 0
SEED0=60013 SEEDSTEP=1000033 PAR=14 ./run_honest.sh eval500b_rl11_exi5as 500 7200 "$X" "round 11 + afterstate root prior (K=$K), eval500b"
python3 ledger.py ingest > /dev/null; python3 b1000.py rl11_main rl11_exi5as | tee -a sim_logs/GATE_exi5.txt
