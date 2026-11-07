#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 80); do
  if grep -q '==== search' sim_logs/fullrun_search.log 2>/dev/null; then break; fi
  sleep 10
done
echo "--- seeds done: $(grep -c '\-> floor' sim_logs/fullrun_search.log) ---"
tail -n 16 sim_logs/fullrun_search.log
