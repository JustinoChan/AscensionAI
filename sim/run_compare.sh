#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f v2_search.jsonl v2_macro.jsonl
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(48)))")
for MODE in search macro; do
  for s in $SEEDS; do
    for t in 1 2 3; do
      out=$(python eval_fullrun_analytics_v2.py $s $MODE 2>/dev/null)
      if [ -n "$out" ]; then echo "$out" >> v2_${MODE}.jsonl; break; fi
    done
  done
  echo "=== $MODE done: $(wc -l < v2_${MODE}.jsonl) seeds ==="
done
echo "##### SEARCH (baseline) #####"
python aggregate_v2.py v2_search.jsonl
echo "##### MACRO (override) #####"
python aggregate_v2.py v2_macro.jsonl
echo ALLDONE
