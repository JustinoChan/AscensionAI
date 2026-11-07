#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
for i in $(seq 1 30); do
  if grep -q 'upd ' sim_logs/varL.log 2>/dev/null; then break; fi
  sleep 3
done
echo "trainers: $(pgrep -fc '[t]rain_search' || echo 0)  workers: $(pgrep -fc '[s]im_worker_search' || echo 0)"
echo "== last log lines =="
tail -n 8 sim_logs/varL.log
