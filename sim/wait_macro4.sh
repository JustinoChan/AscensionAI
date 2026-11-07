#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 95); do
  if grep -q 'campfire' sim_logs/macro4.log 2>/dev/null; then break; fi
  sleep 6
done
echo "macro seeds: $(wc -l < v2_macro.jsonl 2>/dev/null)"
tail -n 26 sim_logs/macro4.log
