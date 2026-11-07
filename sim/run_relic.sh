#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0
rm -f v2_relic.jsonl
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(48)))")
for s in $SEEDS; do
  for t in 1 2 3; do
    out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
    if [ -n "$out" ]; then echo "$out" >> v2_relic.jsonl; echo "seed $s ok"; break; fi
  done
done
echo "=== relic done: $(wc -l < v2_relic.jsonl) seeds ==="
echo "##### RELIC-BUYING #####"
python aggregate_v2.py v2_relic.jsonl
echo ALLDONE
