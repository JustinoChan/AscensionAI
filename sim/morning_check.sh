#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== procs (trainer count) =="
pgrep -fc '[t]rain_sim' || echo "0 (no trainer running)"
echo "== flags =="
for f in .sim_autorun .sim_queue; do
  if [ -f "$f" ]; then echo "$f EXISTS: $(cat $f)"; else echo "$f absent"; fi
done
echo "== varD log: launch + promotion + done =="
grep -E 'mode=|queued run armed|DONE|relaunch' sim_logs/varD.log | tail -n 8
echo "== varD log tail =="
tail -n 6 sim_logs/varD.log
echo "== varE log: launch + done =="
grep -E 'mode=|queued run armed|DONE|relaunch' sim_logs/varE.log 2>/dev/null | tail -n 8
echo "== varE log tail =="
tail -n 6 sim_logs/varE.log 2>/dev/null || echo "no varE.log"
echo "== csv line counts =="
wc -l sim_logs/sim_training_varD.csv sim_logs/sim_training_varE.csv 2>/dev/null
echo "== relaunch totals =="
grep -c 'relaunch' sim_logs/varD.log 2>/dev/null
grep -c 'relaunch' sim_logs/varE.log 2>/dev/null
