#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 92); do
  if grep -q 'ALLDONE' sim_logs/harvest.log 2>/dev/null; then break; fi
  sleep 6
done
echo "fights so far: $(wc -l < fights_deep.jsonl 2>/dev/null)"
if grep -q 'ALLDONE' sim_logs/harvest.log 2>/dev/null; then
  tail -n 16 sim_logs/harvest.log
else
  echo "(still running)"
fi
