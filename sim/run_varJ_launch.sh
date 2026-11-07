#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo '200000 12 5000 varJ fightpure' > .sim_autorun
setsid nohup python train_sim_mp.py 200000 12 5000 varJ fightpure \
  >> sim_logs/varJ.log 2>&1 < /dev/null &
sleep 25
tail -n 4 sim_logs/varJ.log
pgrep -fc '[t]rain_sim' || echo "LAUNCH FAILED"
echo "autorun: $(cat .sim_autorun)"
