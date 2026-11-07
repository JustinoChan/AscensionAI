#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 92); do
  if grep -q 'ALLDONE' sim_logs/rolescorer.log 2>/dev/null; then break; fi
  sleep 6
done
echo "A: $(wc -l < v2_Ar.jsonl 2>/dev/null) | B: $(wc -l < v2_Br.jsonl 2>/dev/null)"
if grep -q 'ALLDONE' sim_logs/rolescorer.log 2>/dev/null; then
  tail -n 44 sim_logs/rolescorer.log
else
  echo "(still running)"; tail -n 2 sim_logs/rolescorer.log
fi
