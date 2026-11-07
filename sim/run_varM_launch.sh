#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
rm -f .sim_autorun
pkill -f '[t]rain_search' 2>/dev/null
pkill -f '[s]im_worker_search' 2>/dev/null
sleep 2
rm -f sim_logs/varM.log
setsid nohup python train_search.py 1500000 12 100000 varM \
    >> sim_logs/varM.log 2>&1 < /dev/null &
sleep 8
echo "trainers: $(pgrep -fc '[t]rain_search')  workers: $(pgrep -fc '[s]im_worker_search')"
tail -n 6 sim_logs/varM.log
