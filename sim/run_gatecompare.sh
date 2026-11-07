#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(48)))")
for ARM in nogate gate; do
  if [ "$ARM" = "nogate" ]; then export GATE_ELITES=0; else export GATE_ELITES=1; fi
  rm -f v2_${ARM}.jsonl
  for s in $SEEDS; do
    for t in 1 2 3; do
      out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
      if [ -n "$out" ]; then echo "$out" >> v2_${ARM}.jsonl; break; fi
    done
  done
  echo "=== $ARM done: $(wc -l < v2_${ARM}.jsonl) seeds ==="
done
echo "##### NO-GATE (3-win baseline) #####"
python aggregate_v2.py v2_nogate.jsonl
echo "##### ELITE-GATE #####"
python aggregate_v2.py v2_gate.jsonl
echo ALLDONE
