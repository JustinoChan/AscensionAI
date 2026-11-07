#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== process tree (trainer + workers) =="
ps -C python -o pid,ppid,etimes,time,%cpu,cmd --sort=etimes | grep -E 'train_search|sim_worker_search' | head -20
echo "== count of search workers =="
pgrep -fc '[s]im_worker_search'
echo "== trainer CPU time (is it computing or blocked?) =="
ps -C python -o pid,time,%cpu,cmd | grep '[t]rain_search'
echo "== full varK.log =="
cat sim_logs/varK.log
echo "== policy_live.npz mtime (rewritten each update?) =="
ls -la --time-style=+%H:%M:%S policy_live.npz
date +%H:%M:%S
