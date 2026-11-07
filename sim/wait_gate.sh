#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 92); do
  if grep -q 'ALLDONE' sim_logs/gatecmp.log 2>/dev/null; then break; fi
  sleep 6
done
echo "nogate: $(wc -l < v2_nogate.jsonl 2>/dev/null) | gate: $(wc -l < v2_gate.jsonl 2>/dev/null)"
if grep -q 'ALLDONE' sim_logs/gatecmp.log 2>/dev/null; then
  tail -n 40 sim_logs/gatecmp.log
else
  echo "(still running)"; tail -n 2 sim_logs/gatecmp.log
fi
