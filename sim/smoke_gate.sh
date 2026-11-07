#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
echo "-- gate ON 10007 --"; GATE_ELITES=1 python eval_fullrun_analytics_v2.py 10007 macro 2>/dev/null
echo "-- gate OFF 10007 --"; GATE_ELITES=0 python eval_fullrun_analytics_v2.py 10007 macro 2>/dev/null
echo "-- gate ON 144630 --"; GATE_ELITES=1 python eval_fullrun_analytics_v2.py 144630 macro 2>/dev/null
