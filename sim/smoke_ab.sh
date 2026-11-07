#!/bin/bash
cd /home/justinochan16/sts_gym_spike || exit 1
source .venv/bin/activate
export GATE_ELITES=0
echo "== A relics-off 144630 =="; RELIC_BUY=0 python eval_fullrun_analytics_v2.py 144630 macro 2>/dev/null
echo "== B relic-v2 144630 =="; RELIC_BUY=1 python eval_fullrun_analytics_v2.py 144630 macro 2>/dev/null
