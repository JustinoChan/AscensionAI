#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0
echo "-- 10007 --"; python eval_fullrun_analytics_v2.py 10007 macro 2>/dev/null
echo "-- 144630 --"; python eval_fullrun_analytics_v2.py 144630 macro 2>/dev/null
