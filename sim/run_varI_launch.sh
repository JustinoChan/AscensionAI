#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo '200000 12 5000 varI fightpure' > .sim_autorun
setsid nohup python train_sim_mp.py 200000 12 5000 varI fightpure \
  >> sim_logs/varI.log 2>&1 < /dev/null &
sleep 25
tail -n 5 sim_logs/varI.log
pgrep -fc '[t]rain_sim' || echo "LAUNCH FAILED"
echo "autorun: $(cat .sim_autorun)"
