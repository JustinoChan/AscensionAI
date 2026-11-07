#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f fights_deep.jsonl
# seeds i in [150, 350) -> disjoint from the 150 eval seeds (i<150)
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(150, 350)))")
done=0
for s in $SEEDS; do
  for t in 1 2 3; do
    out=$(python harvest_fights.py $s 2>/dev/null)
    if [ -n "$out" ]; then echo "$out" >> fights_deep.jsonl; done=$((done+1)); break; fi
  done
done
echo "=== harvested $done seeds ==="
echo "total fights: $(wc -l < fights_deep.jsonl)"
echo "by act/room:"
grep -o '"act": [0-9], "room": "[A-Z]*"' fights_deep.jsonl | sort | uniq -c
echo ALLDONE
