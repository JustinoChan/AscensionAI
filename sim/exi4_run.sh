#!/bin/bash
# Combat-learner rethink (three retrains stalled at the exi1 level).
# A) 1 h label-noise diagnostic: mainline live root states labelled TWICE with independent K=64 sampled worlds.
#    If exi1's regret against the 2nd labelling is within 1.5x of the labels' own regret (label vs label), the net is near
#    the noise ceiling -> B1 better labels; otherwise -> B2 better learner.
# B1) 6 h of K=256 labels (4x less noise) on live root states; train from exi1 with soft-target CE + expected-regret term.
# B2) fresh init (no warm start) on all ~500k states, CE + expected-regret term, 30 epochs.
# Then pair boss gate exi1 vs exi4 (both c=5 + play-outs) on fresh held-out A2+A3 bosses; PASS -> eval500 vs round 11 (232).
cd "$HOME/sts"
P1=nets/exi1/rollout.npz; MAIN="MACRO_MAPFEAT=1 MACRO_POLICY=nets/rl11/pol_gate.npz POTION_VALUE=10"
CMB="HONEST_ROOT_NET=$P1 HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$P1 EXI_ACT=search EXI_RUN_K=16 EXI_EPS=2 EXI_SEEDSTEP=434"
N=nets/exi4; mkdir -p $N sim_logs/exi4
for w in $(seq 0 13); do
  env $MAIN $CMB EXI_DOUBLE=1 EXI_K=64 timeout 1h nice -n 5 .venv/bin/python exi_collect.py $P1 $((999000001 + w*31)) 500 sim_logs/exi4/noise >> sim_logs/exi4/noise.log 2>/dev/null &
done; wait
.venv/bin/python exi_noise.py sim_logs/exi4/noise $P1 policy_varN.npz | tee sim_logs/EXI4_NOISE.txt
BR=$(.venv/bin/python -c "
import json; r=json.load(open('sim_logs/EXI4_NOISE.txt'))
print('B1' if r['nets/exi1/rollout.npz']['regret_vs_2nd'] <= 1.5 * r['label_regret_vs_2nd'] else 'B2')")
echo "branch $BR" | tee -a sim_logs/EXI4_NOISE.txt
if [ "$BR" = B1 ]; then
  for w in $(seq 0 13); do
    env $MAIN $CMB EXI_K=256 timeout 6h nice -n 5 .venv/bin/python exi_collect.py $P1 $((999500001 + w*31)) 500 sim_logs/exi4/k256 >> sim_logs/exi4/k256.log 2>/dev/null &
  done; wait
  rm -f sim_logs/exi4/k256/*.tmp.npz
  EXI_TAU=1 EXI_EPOCHS=20 EXI_REGRET_W=0.2 .venv_train/bin/python exi_train.py $P1 $N/rollout.npz sim_logs/exi4/k256 sim_logs/exi3/data 2>&1 | tee $N/train.log
else
  EXI_FRESH=1 EXI_TAU=1 EXI_EPOCHS=30 EXI_LR=3e-4 EXI_REGRET_W=0.2 .venv_train/bin/python exi_train.py $P1 $N/rollout.npz sim_logs/exi3/data sim_logs/exi2/data sim_logs/exi1/data sim_logs/exi1c/data 2>&1 | tee $N/train.log
fi
.venv/bin/python exi_noise.py sim_logs/exi4/noise $P1 $N/rollout.npz | tee -a sim_logs/EXI4_NOISE.txt
for ACT in 2 3; do
  OUT=sim_logs/boss_exi4_a$ACT; NS=$([ $ACT = 2 ] && echo 10 || echo 14)
  for w in $(seq 0 13); do
    env $MAIN PRIOR_NET=$N/rollout.npz PRIOR_BASE_NET=$P1 GATE_ACT=$ACT GATE_SEEDSTEP=434 nice -n 5 .venv/bin/python boss_prior_gate.py $((999800013 + ACT*10000 + w*31)) $NS $OUT >> $OUT.log 2>/dev/null &
  done; wait
done
python3 boss_prior_report.py sim_logs/boss_exi4_a2 sim_logs/boss_exi4_a3 | tee sim_logs/GATE_exi4.txt
[ "$(tail -1 sim_logs/GATE_exi4.txt | awk '{print $2}')" = c5roll ] || { echo "exi4 did not beat exi1" | tee -a sim_logs/GATE_exi4.txt; exit 0; }
SEED0=40009 SEEDSTEP=1000003 PAR=14 ./run_honest.sh eval500_rl11_exi4 500 7200 "HONEST=1 HONEST_CPP=1 HONEST_K=64 $MAIN HONEST_ROOT_NET=$N/rollout.npz HONEST_ROOT_C=5 HONEST_ROLLOUT_NET=$N/rollout.npz TRACE=1" "round 11 + exi4 combat net, eval500"
python3 ledger.py ingest > /dev/null; python3 ledger.py compare eval500_rl11_main eval500_rl11_exi4 | sed -n 1,8p | tee -a sim_logs/GATE_exi4.txt
