#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== ev band (every ~30th update) =="
grep -E '^upd' sim_logs/varJ.log | awk 'NR%30==0 {print $1,$2,$3, "ev="$21}'
echo "== train win-rate + greedy fight eval per milestone =="
paste <(grep 'train fight win-rate' sim_logs/varJ.log) <(grep 'fight eval' sim_logs/varJ.log) | sed 's/  */ /g'
echo "== latest =="
tail -n 1 sim_logs/varJ.log
