#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f .sim_autorun                       # keep the model-free watchdog out of this
pkill -f '[t]rain_search' 2>/dev/null
pkill -f '[s]im_worker_search' 2>/dev/null
sleep 2
setsid nohup python train_search.py 1500000 12 100000 varL \
    >> sim_logs/varL.log 2>&1 < /dev/null &
sleep 6
echo "trainers: $(pgrep -fc '[t]rain_search' || echo 0)  workers: $(pgrep -fc '[s]im_worker_search' || echo 0)"
echo "== first log lines =="
tail -n 5 sim_logs/varL.log
