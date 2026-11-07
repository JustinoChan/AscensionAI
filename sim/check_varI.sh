#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
echo "== train fight win-rate + greedy fight eval by milestone =="
paste <(grep 'train fight win-rate' sim_logs/varI.log) <(grep 'fight eval' sim_logs/varI.log) | sed 's/  */ /g'
echo "== latest =="
tail -n 1 sim_logs/varI.log
