#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f v2_decisions.jsonl
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(48)))")
for s in $SEEDS; do
  for t in 1 2 3; do
    out=$(python eval_fullrun_analytics_v2.py $s 2>/dev/null)
    if [ -n "$out" ]; then echo "$out" >> v2_decisions.jsonl; echo "seed $s ok"; break; fi
    echo "seed $s retry $t"
  done
done
echo "=== collected: $(wc -l < v2_decisions.jsonl) seeds ==="
python aggregate_v2.py
