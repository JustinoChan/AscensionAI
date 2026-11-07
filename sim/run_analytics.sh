#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f sim_logs/fullrun_deaths.log
setsid nohup python eval_fullrun_analytics.py fullrun_deaths.csv 48 \
    >> sim_logs/fullrun_deaths.log 2>&1 < /dev/null &
for i in $(seq 1 110); do
  if grep -q 'SUMMARY' sim_logs/fullrun_deaths.log 2>/dev/null; then break; fi
  sleep 6
done
echo "=== seeds done: $(grep -c 'win=' sim_logs/fullrun_deaths.log) ==="
tail -n 30 sim_logs/fullrun_deaths.log
