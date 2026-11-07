#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0
export RELIC_BUY=0
# COMBAT_NPZ defaults to policy_varN.npz (mainline)
rm -f v2_roles.jsonl
SEEDS=$(python -c "print(' '.join(str(10007+i*7919) for i in range(150)))")
for s in $SEEDS; do
  for t in 1 2 3; do
    out=$(python eval_fullrun_analytics_v2.py $s macro 2>/dev/null)
    if [ -n "$out" ]; then echo "$out" >> v2_roles.jsonl; break; fi
  done
done
echo "=== roles run done: $(wc -l < v2_roles.jsonl) ==="
python aggregate_roles.py v2_roles.jsonl
echo "--- overall (reference) ---"
python aggregate_v2.py v2_roles.jsonl | head -3
echo ROLES_DONE
