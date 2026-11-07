#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 92); do
  if grep -q 'ALLDONE' sim_logs/combatdiag.log 2>/dev/null; then break; fi
  sleep 6
done
echo "diag seeds: $(wc -l < v2_diag.jsonl 2>/dev/null)"
if grep -q 'ALLDONE' sim_logs/combatdiag.log 2>/dev/null; then
  tail -n 22 sim_logs/combatdiag.log
else
  echo "(still running)"
fi
