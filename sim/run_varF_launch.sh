#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo '300000 12 5000 varF fightmix' > .sim_autorun
setsid nohup python train_sim_mp.py 300000 12 5000 varF fightmix \
  >> sim_logs/varF.log 2>&1 < /dev/null &
sleep 30
tail -n 6 sim_logs/varF.log
pgrep -fc '[t]rain_sim' || echo "LAUNCH FAILED"
echo "autorun: $(cat .sim_autorun)"
