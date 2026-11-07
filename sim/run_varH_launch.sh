#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo '150000 12 5000 varH fightpure' > .sim_autorun
setsid nohup python train_sim_mp.py 150000 12 5000 varH fightpure \
  >> sim_logs/varH.log 2>&1 < /dev/null &
sleep 25
tail -n 8 sim_logs/varH.log
pgrep -fc '[t]rain_sim' || echo "LAUNCH FAILED"
echo "autorun: $(cat .sim_autorun)"
