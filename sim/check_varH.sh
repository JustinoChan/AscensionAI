#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== ev trend (every ~25th update line) =="
grep -E '^upd' sim_logs/varH.log | awk 'NR%25==0'
echo "== fight eval milestones =="
grep 'fight eval' sim_logs/varH.log
echo "== latest =="
tail -n 1 sim_logs/varH.log
