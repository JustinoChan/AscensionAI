#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 90); do
  if grep -q ALLDONE sim_logs/compare.log 2>/dev/null; then break; fi
  sleep 6
done
echo "search: $(wc -l < v2_search.jsonl 2>/dev/null) | macro: $(wc -l < v2_macro.jsonl 2>/dev/null)"
tail -n 48 sim_logs/compare.log
