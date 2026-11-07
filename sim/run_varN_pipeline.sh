#!/bin/bash
# Unattended varN pipeline: train combat net on the combined (act-oversampled)
# bank, freeze it, restore the original bank, then auto-eval (per-act diagnostic
# + 150-seed full run) with the new net. Assumes the combined bank is already
# built (bank_train.jsonl) with the original backed up at bank_train.act1.bak.
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
EV=sim_logs/varN_eval.log
echo "=== varN TRAIN start $(date) ===" >> $EV
python train_search.py 1500000 12 100000 varN >> sim_logs/varN.log 2>&1
cp policy_live.npz policy_varN.npz
cp bank_train.act1.bak bank_train.jsonl
echo "=== varN TRAIN done, EVAL start $(date) ===" >> $EV
export GATE_ELITES=0
export RELIC_BUY=0
export COMBAT_NPZ=policy_varN.npz
rm -f v2_varN.jsonl
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(150)))")
for s in $SEEDS; do
  for t in 1 2 3; do
    out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
    if [ -n "$out" ]; then echo "$out" >> v2_varN.jsonl; break; fi
  done
done
{
  echo "##### varN PER-ACT COMBAT #####"
  python aggregate_combat.py v2_varN.jsonl
  echo "##### varN FULL-RUN #####"
  python aggregate_v2.py v2_varN.jsonl
  echo "PIPELINE_DONE $(date)"
} >> $EV
