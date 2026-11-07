#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== process status =="
echo "trainers: $(pgrep -fc '[t]rain_search' || echo 0)  workers: $(pgrep -fc '[s]im_worker_search' || echo 0)"
ps -C python -o etimes,time,%cpu,cmd --sort=etimes | grep -E '[t]rain_search' | head -3
echo
echo "== log mtime / now =="
ls -la --time-style=+%H:%M:%S sim_logs/varK.log 2>/dev/null
date +%H:%M:%S
echo
echo "== MILESTONE lines (the greedy vs search readout) =="
grep -E 'MILESTONE|EVAL|g_boss|FINAL|done|complete' sim_logs/varK.log 2>/dev/null
echo
echo "== last 25 log lines =="
tail -n 25 sim_logs/varK.log 2>/dev/null
echo
echo "== CSV head + tail =="
if [ -f sim_logs/sim_training_varK.csv ]; then
  head -n 1 sim_logs/sim_training_varK.csv
  echo "... ($(wc -l < sim_logs/sim_training_varK.csv) rows total) ..."
  tail -n 12 sim_logs/sim_training_varK.csv
else
  echo "(no varK csv yet)"
  ls -la sim_logs/ | grep -i vark
fi
echo
echo "== checkpoints =="
ls -la --time-style=+%H:%M:%S sim_models_varK/ 2>/dev/null || echo "(no sim_models_varK dir)"
