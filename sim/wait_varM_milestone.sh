#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 110); do
  if grep -q 'MILESTONE' sim_logs/varM.log 2>/dev/null; then break; fi
  sleep 6
done
echo "trainers: $(pgrep -fc '[t]rain_search')  workers: $(pgrep -fc '[s]im_worker_search')"
grep -E 'MILESTONE|DONE' sim_logs/varM.log | head -4
echo "--- last upd ---"
tail -n 3 sim_logs/varM.log
