#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== process =="
ps -C python -o etimes,%cpu,cmd --sort=etimes | grep '[t]rain_search'
echo "trainers: $(pgrep -fc '[t]rain_search')  workers: $(pgrep -fc '[s]im_worker_search')"
date +%H:%M:%S
echo
echo "== varL MILESTONE lines (greedy vs search) =="
grep -E 'MILESTONE|DONE' sim_logs/varL.log 2>/dev/null
echo
echo "== last 6 upd lines =="
tail -n 6 sim_logs/varL.log 2>/dev/null
echo
echo "== varL CSV =="
if [ -f sim_logs/sim_training_varL.csv ]; then
  head -n 1 sim_logs/sim_training_varL.csv
  tail -n +2 sim_logs/sim_training_varL.csv
else
  echo "(no varL csv yet)"
fi
