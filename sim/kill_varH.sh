#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
rm -f .sim_autorun
pkill -f '[t]rain_sim_mp.py' && echo "killed trainer" || echo "no trainer running"
sleep 2
pgrep -fc '[t]rain_sim_mp.py' || echo "confirmed: 0 trainers"
echo "autorun: $(cat .sim_autorun 2>/dev/null || echo none)"
