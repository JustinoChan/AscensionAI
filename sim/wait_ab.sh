#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 92); do
  if grep -q 'ALLDONE' sim_logs/ab150.log 2>/dev/null; then break; fi
  sleep 6
done
echo "A: $(wc -l < v2_A.jsonl 2>/dev/null) | B: $(wc -l < v2_B.jsonl 2>/dev/null)"
if grep -q 'ALLDONE' sim_logs/ab150.log 2>/dev/null; then
  tail -n 42 sim_logs/ab150.log
else
  echo "(still running)"; tail -n 2 sim_logs/ab150.log
fi
