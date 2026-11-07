#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(150)))")
for ARM in A B; do
  if [ "$ARM" = "A" ]; then export RELIC_BUY=0; else export RELIC_BUY=1; fi
  rm -f v2_${ARM}.jsonl
  for s in $SEEDS; do
    for t in 1 2 3; do
      out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
      if [ -n "$out" ]; then echo "$out" >> v2_${ARM}.jsonl; break; fi
    done
  done
  echo "=== arm $ARM done: $(wc -l < v2_${ARM}.jsonl) seeds ==="
done
echo "##### A: deck-hygiene baseline (relics OFF) #####"
python aggregate_v2.py v2_A.jsonl
echo "##### B: deck-hygiene + relic v2 (remove-first) #####"
python aggregate_v2.py v2_B.jsonl
echo ALLDONE
