#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0
export RELIC_BUY=0
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(150)))")
for ARM in A B; do
  if [ "$ARM" = "A" ]; then export ROLE_SCORE=0; else export ROLE_SCORE=1; fi
  rm -f v2_${ARM}r.jsonl
  for s in $SEEDS; do
    for t in 1 2 3; do
      out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
      if [ -n "$out" ]; then echo "$out" >> v2_${ARM}r.jsonl; break; fi
    done
  done
  echo "=== arm $ARM done: $(wc -l < v2_${ARM}r.jsonl) seeds ==="
done
echo "##### A: varN + deck-hygiene (NO role scorer) #####"
python aggregate_v2.py v2_Ar.jsonl | head -2
python aggregate_roles.py v2_Ar.jsonl
echo "##### B: + AoE/draw role scorer #####"
python aggregate_v2.py v2_Br.jsonl | head -2
python aggregate_roles.py v2_Br.jsonl
echo "##### B per-act combat #####"
python aggregate_combat.py v2_Br.jsonl
echo ALLDONE
