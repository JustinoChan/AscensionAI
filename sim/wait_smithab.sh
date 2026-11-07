#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 92); do
  if grep -q 'ALLDONE' sim_logs/smithab.log 2>/dev/null; then break; fi
  sleep 6
done
echo "A: $(wc -l < v2_As.jsonl 2>/dev/null) | B: $(wc -l < v2_Bs.jsonl 2>/dev/null)"
if grep -q 'ALLDONE' sim_logs/smithab.log 2>/dev/null; then
  tail -n 40 sim_logs/smithab.log
else
  echo "(still running)"; tail -n 1 sim_logs/smithab.log
fi
