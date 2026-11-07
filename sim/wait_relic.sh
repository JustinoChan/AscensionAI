#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 95); do
  if grep -q 'ALLDONE' sim_logs/relic.log 2>/dev/null; then break; fi
  sleep 6
done
echo "relic seeds: $(wc -l < v2_relic.jsonl 2>/dev/null)"
tail -n 22 sim_logs/relic.log
