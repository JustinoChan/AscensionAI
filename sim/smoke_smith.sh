#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
for s in 10007 41683 144630; do
  echo "-- $s --"
  python eval_fullrun_analytics_v2.py $s macro 2>/dev/null
done
