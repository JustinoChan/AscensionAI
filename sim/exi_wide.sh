#!/bin/bash
# Learner capacity: expert iteration stalled (exi2 = exi1: offline regret flat ~7.4, gate 173 vs 172).
# Train a wider prior net (1024-512-256, fresh init, 40 epochs, cosine LR) on all ~355k labelled states; use it ONLY as the
# root prior (c=5), keeping exi1 as the play-out net (rollout speed unchanged). Pair gate on fresh held-out A2+A3 bosses:
# exi1 c5roll vs wide prior + exi1 play-outs. PASS -> eval500 + PV10. FAIL -> c sweep (3, 10) around c5roll with exi1.
cd "$HOME/sts"
MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl5d/pol_gate.npz POTION_VALUE=10"; P1=nets/exi1/rollout.npz; N=nets/exiw; mkdir -p $N
[ -f $N/rollout.npz ] || EXI_WIDTH=1024,512,256 EXI_EPOCHS=40 EXI_LR=3e-4 EXI_TAU=1 .venv_train/bin/python exi_train.py $P1 $N/rollout.npz sim_logs/exi1/data sim_logs/exi1c/data sim_logs/exi2/data 2>&1 | tee $N/train.log
gate() {  # TAG ENV...
  T=$1; shift
  for ACT in 2 3; do
    OUT=sim_logs/boss_$T_a$ACT; OUT=sim_logs/boss_${T}_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
    for w in $(seq 0 13); do
      env $MAIN "$@" GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((993000013 + ${#T}*7 + ACT*100000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
    done; wait
  done
  python3 boss_prior_report.py sim_logs/boss_${T}_a2 sim_logs/boss_${T}_a3 | tee sim_logs/GATE_$T.txt
}
gate exiw PRIOR_NET=$N/rollout.npz PRIOR_BASE_NET=$P1 PRIOR_ROLL_NET=$P1
if [ "$(tail -1 sim_logs/GATE_exiw.txt | awk '{print $2}')" = c5roll ]; then
  SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_exiw_c5 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$N/rollout.npz HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 TRACE=1" "wide root prior + exi1 play-outs, eval500"
  python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_prior_c5roll eval500_exiw_c5 | sed -n 1,8p | tee -a sim_logs/GATE_exiw.txt
else
  gate csweep PRIOR_NET=$P1 PRIOR_CSWEEP=1
fi
