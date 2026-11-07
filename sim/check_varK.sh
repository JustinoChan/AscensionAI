#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== updates so far =="
grep -E '^upd' sim_logs/varK.log | tail -6
echo "== milestones (greedy vs search) =="
grep 'MILESTONE' sim_logs/varK.log
echo "== procs =="
pgrep -fc '[t]rain_search' || echo "0 (not running)"
